"""media-gen-mcp: an HTTP MCP server exposing image generation to Hermes.

Called directly from Hermes (registered as an HTTP mcp_servers entry in
~/.hermes/config.yaml -- see k8s/p3/hermes-agent.yaml). There is no OpenAI-
compatible bridge in front of ComfyUI: the OpenAI-shaped tool
(pnyxai/comfyui-openai-api) that the original plan assumed does not exist
(confirmed via WebSearch -- no such package/repo), and ComfyUI's own API
has no native /v1/images/generations endpoint on the version we run
(0.34.0 -- confirmed empirically, a direct fetch() returns 404 despite a
blog post's claim it exists since 0.3.10). So this server talks to
ComfyUI's real native API directly: POST /prompt, GET /history/{id},
GET /view, POST /free.

Concurrency model ("full swap", Eldar's explicit decision -- no VRAM
coexistence tier at all): a generation call
  1. sets a Redis busy flag (media-gen:busy) BEFORE touching anything else,
     so custom_router_hook.py (k8s/p1/litellm.yaml) can see it and redirect
     any chat request that arrives mid-generation to the OpenRouter cloud
     fallback instead of racing this pod for VRAM or queueing behind it.
  2. hits llama-swap's real unload endpoint (POST /api/models/unload,
     confirmed via llama-swap's own docs) to fully release whatever chat
     model is resident -- ComfyUI gets the whole 16GB card.
  3. submits the workflow, polls for the result, fetches the output image.
  4. calls ComfyUI-Manager's /free endpoint (confirmed empirically against
     the live pod: POST /free and /api/free both return 200 with
     {"unload_models": true, "free_memory": true}) so ComfyUI actually
     releases VRAM afterward instead of caching the model for speed --
     without this the chat model's next lazy-reload would OOM.
  5. clears the busy flag in a finally block, so a crash mid-generation
     can't wedge the router into permanent cloud-fallback mode. The Redis
     key's own TTL (set in step 1) is a second, independent safety net for
     the same failure mode (pod killed outright, no finally block runs).

Workflow JSON below is not hand-authored -- it's the exact payload
ComfyUI's own frontend sent for a real, verified-successful Z-Image Turbo
generation (captured by calling window.app.graphToPrompt() against the
live default graph in a browser tab pointed at the running ComfyUI pod,
confirmed against a real generated image). Node IDs match that capture:
  9  SaveImage            62 CLIPLoader (qwen_3_4b.safetensors)
  63 VAELoader (ae.safetensors)      65 VAEDecode
  66 UNETLoader (z_image_turbo_bf16.safetensors)
  67 CLIPTextEncode (positive)       68 EmptySD3LatentImage
  69 ModelSamplingAuraFlow           70 KSampler
  71 CLIPTextEncode (negative)
Only the fields generate_image() actually parameterizes are touched;
everything else is passed through byte-for-byte from the capture.
"""
import asyncio
import copy
import logging
import os
import random
import time

import httpx
import redis.asyncio as redis
# mcp>=2.0 renamed FastMCP -> MCPServer and moved it to mcp.server.mcpserver
# (confirmed empirically against the actually-installed package -- the
# earlier research this project did mid-build found the 1.x FastMCP name/
# import path via WebFetch of docs that turned out to describe the older
# API; `pip install mcp` today pulls 2.x, where that import raises
# ModuleNotFoundError with its own migration-guide message). json_response
# also moved: it's now a run()/run_streamable_http_async() kwarg, not a
# constructor kwarg.
from mcp.server.mcpserver import MCPServer, Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s media-gen-mcp %(levelname)s: %(message)s")
logger = logging.getLogger("media-gen-mcp")

LLAMA_SWAP_URL = os.environ.get("LLAMA_SWAP_URL", "http://llama-swap.ai-models.svc.cluster.local:8080")
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://comfyui.ai-models.svc.cluster.local:8188")
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis.ai-memory.svc.cluster.local:6379/0")

BUSY_KEY = "media-gen:busy"
BUSY_TTL_SECONDS = 300  # safety-net only -- the finally block is the primary clear path

# Poll /history for up to this long before giving up. Z-Image Turbo took
# ~18-20s end-to-end in the verified live test; 180s leaves generous room
# for a cold model load on top of that without hanging forever on a stuck job.
POLL_TIMEOUT_SECONDS = 180
POLL_INTERVAL_SECONDS = 1.0

DEFAULT_NEGATIVE_PROMPT = "low quality, bad anatomy, extra digits, missing digits, extra limbs, missing limbs"

# The exact captured, verified-working /prompt payload. Deep-copied per
# request before any field is edited.
WORKFLOW_TEMPLATE = {
    "9": {
        "inputs": {"filename_prefix": "media-gen-mcp", "images": ["65", 0]},
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"},
    },
    "62": {
        "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "default"},
        "class_type": "CLIPLoader",
        "_meta": {"title": "Load CLIP"},
    },
    "63": {
        "inputs": {"vae_name": "ae.safetensors"},
        "class_type": "VAELoader",
        "_meta": {"title": "Load VAE"},
    },
    "65": {
        "inputs": {"samples": ["70", 0], "vae": ["63", 0]},
        "class_type": "VAEDecode",
        "_meta": {"title": "VAE Decode"},
    },
    "66": {
        "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"},
        "class_type": "UNETLoader",
        "_meta": {"title": "Load Diffusion Model"},
    },
    "67": {
        "inputs": {"text": "", "clip": ["62", 0]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Prompt)"},
    },
    "68": {
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        "class_type": "EmptySD3LatentImage",
        "_meta": {"title": "EmptySD3LatentImage"},
    },
    "69": {
        "inputs": {"shift": 3, "model": ["66", 0]},
        "class_type": "ModelSamplingAuraFlow",
        "_meta": {"title": "ModelSamplingAuraFlow"},
    },
    "70": {
        "inputs": {
            "seed": 42,
            "steps": 8,
            "cfg": 1,
            "sampler_name": "res_multistep",
            "scheduler": "simple",
            "denoise": 1,
            "model": ["69", 0],
            "positive": ["67", 0],
            "negative": ["71", 0],
            "latent_image": ["68", 0],
        },
        "class_type": "KSampler",
        "_meta": {"title": "KSampler"},
    },
    "71": {
        "inputs": {"text": DEFAULT_NEGATIVE_PROMPT, "clip": ["62", 0]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Prompt)"},
    },
}

mcp = MCPServer("media-gen")
_redis = redis.from_url(REDIS_URL, decode_responses=True)


async def _unload_chat_model(client: httpx.AsyncClient) -> None:
    try:
        resp = await client.post(f"{LLAMA_SWAP_URL}/api/models/unload", timeout=30.0)
        logger.info("llama-swap unload -> %s", resp.status_code)
    except Exception as exc:  # noqa: BLE001 -- best effort, don't block generation on this
        logger.warning("llama-swap unload failed (%s: %s) -- continuing anyway", type(exc).__name__, exc)


async def _free_comfyui_vram(client: httpx.AsyncClient) -> None:
    try:
        resp = await client.post(
            f"{COMFYUI_URL}/free",
            json={"unload_models": True, "free_memory": True},
            timeout=30.0,
        )
        logger.info("ComfyUI /free -> %s", resp.status_code)
    except Exception as exc:  # noqa: BLE001 -- best effort, must not mask the real result/error
        logger.warning("ComfyUI /free failed (%s: %s)", type(exc).__name__, exc)


@mcp.tool()
async def generate_image(
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    width: int = 1024,
    height: int = 1024,
    steps: int = 8,
    seed: int = -1,
) -> Image:
    """Generate an image locally with Z-Image Turbo (via ComfyUI on aibox's GPU).

    This fully unloads whatever chat model is currently resident to give
    the whole GPU to image generation, then reloads the chat model lazily
    afterward -- so it takes over the GPU for the duration of the call
    (roughly 20-40s depending on model cold-start). Any chat message sent
    elsewhere during that window is automatically redirected to a cloud
    model instead of queueing or failing.

    Args:
        prompt: What to generate. Be descriptive -- style, subject, setting,
            lighting all help.
        negative_prompt: What to avoid in the image. Defaults to a generic
            quality/anatomy negative prompt if not given.
        width: Image width in pixels. Default 1024.
        height: Image height in pixels. Default 1024.
        steps: Sampling steps. Default 8 (Z-Image Turbo is tuned for very
            few steps -- more isn't necessarily better).
        seed: Fixed seed for reproducibility. Default -1 picks a random
            seed each call.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")

    already_busy = await _redis.set(BUSY_KEY, "1", ex=BUSY_TTL_SECONDS, nx=True)
    if not already_busy:
        raise RuntimeError(
            "Media generation is already in progress. ComfyUI only holds "
            "the GPU for one job at a time -- try again in a moment."
        )

    resolved_seed = random.randint(0, 2**31 - 1) if seed < 0 else seed

    async with httpx.AsyncClient() as client:
        try:
            await _unload_chat_model(client)

            workflow = copy.deepcopy(WORKFLOW_TEMPLATE)
            workflow["67"]["inputs"]["text"] = prompt
            workflow["71"]["inputs"]["text"] = negative_prompt
            workflow["68"]["inputs"]["width"] = width
            workflow["68"]["inputs"]["height"] = height
            workflow["70"]["inputs"]["steps"] = steps
            workflow["70"]["inputs"]["seed"] = resolved_seed

            submit_resp = await client.post(
                f"{COMFYUI_URL}/prompt", json={"prompt": workflow}, timeout=30.0
            )
            submit_resp.raise_for_status()
            prompt_id = submit_resp.json()["prompt_id"]
            logger.info("submitted prompt_id=%s seed=%s", prompt_id, resolved_seed)

            deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
            history = None
            while time.monotonic() < deadline:
                hist_resp = await client.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=15.0)
                hist_resp.raise_for_status()
                body = hist_resp.json()
                if prompt_id in body:
                    history = body[prompt_id]
                    break
                await asyncio.sleep(POLL_INTERVAL_SECONDS)

            if history is None:
                raise TimeoutError(
                    f"ComfyUI did not finish prompt {prompt_id} within {POLL_TIMEOUT_SECONDS}s"
                )

            status = history.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI reported an error for prompt {prompt_id}: {status}")

            images = history.get("outputs", {}).get("9", {}).get("images", [])
            if not images:
                raise RuntimeError(f"ComfyUI finished prompt {prompt_id} but produced no image output")
            image_ref = images[0]

            view_resp = await client.get(
                f"{COMFYUI_URL}/view",
                params={
                    "filename": image_ref["filename"],
                    "subfolder": image_ref.get("subfolder", ""),
                    "type": image_ref.get("type", "output"),
                },
                timeout=30.0,
            )
            view_resp.raise_for_status()
            image_bytes = view_resp.content
            logger.info("fetched output image %s (%d bytes)", image_ref["filename"], len(image_bytes))

            return Image(data=image_bytes, format="png")
        finally:
            await _free_comfyui_vram(client)
            await _redis.delete(BUSY_KEY)


if __name__ == "__main__":
    # streamable_http_path defaults to "/mcp" (confirmed by inspecting the
    # installed mcp package's MCPServer.run_streamable_http_async) -- that's
    # the path k8s/p3/hermes-agent.yaml's mcp_servers.media-gen.url assumes.
    # json_response=True: plain JSON responses instead of SSE streaming --
    # simpler for a single-tool server with no need for server-initiated
    # push during a call.
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000, json_response=True)
