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

DELIVERY (fixed 2026-09-08, third pass -- see git history for the first
two): confirmed via Hermes's own docs (hermes-agent.nousresearch.com/docs/
user-guide/features/deliverable-mode) that Hermes does NOT auto-attach
binary/base64 image content returned from a tool call to the outgoing
message -- its real mechanism is text-path-based. Live proof this was A
bug: a real end-to-end test generated a real image (confirmed via
media-gen-mcp's own logs -- unload, submit, poll, fetch, free all
succeeded) and Hermes's reply accurately DESCRIBED the image but no photo
reached Telegram, because the original Image-content return gave the model
something to see/describe, not a path to mention.

Second pass (return a plain string with the path) assumed the docs page's
claim that bare-path-mention alone is sufficient. THIS WAS INCOMPLETE:
Hermes's own source (github.com/NousResearch/hermes-agent PR #27813, which
added that bare-path feature) says explicitly "Existing MEDIA:<path> tag
path is untouched; bare-path delivery is the simpler primitive small
models can use without the tag syntax" -- i.e. there are TWO coexisting,
independently-documented mechanisms, not one, and per-gateway wiring gaps
between them are real and precedented (github.com/NousResearch/hermes-agent
issue #5105: the Signal gateway has the MEDIA: tag parser in code but it
was never wired into that gateway's actual send path). Rather than bet on
which one this deployment's Telegram wiring actually uses, generate_image
now emits BOTH in its returned string: the bare path, and an explicit
`MEDIA:<path>` line for the model to copy verbatim into its reply. Costs
nothing, covers either mechanism.

Either way, the file itself still has to physically exist somewhere
Hermes's own pod can read it: write the fetched bytes to a hostPath volume
shared with Hermes's own pod (SHARED_OUTPUT_DIR, mounted identically in
both Deployments -- see k8s/media-gen/media-gen-mcp-deployment.yaml and
k8s/p3/hermes-agent.yaml). That plumbing is unchanged by this pass --
only the returned text format changed.
"""
import asyncio
import copy
import logging
import os
import random
import time
import uuid

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
from mcp.server.mcpserver import MCPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s media-gen-mcp %(levelname)s: %(message)s")
logger = logging.getLogger("media-gen-mcp")

LLAMA_SWAP_URL = os.environ.get("LLAMA_SWAP_URL", "http://llama-swap.ai-models.svc.cluster.local:8080")
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://comfyui.ai-models.svc.cluster.local:8188")
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis.ai-memory.svc.cluster.local:6379/0")
# Must be the SAME path inside both this container and hermes-agent's --
# see the "DELIVERY" note above. A hostPath volume, not a PVC: this is a
# single-node cluster, so both pods always land on the same node anyway,
# and hostPath matches the pattern already used for /data/models,
# /data/comfyui elsewhere in this repo instead of introducing a new PVC.
SHARED_OUTPUT_DIR = os.environ.get("SHARED_OUTPUT_DIR", "/shared-output")

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

# Video: HunyuanVideo 1.5, fp8-scaled, 480p, cfg-distilled (Eldar's pick via
# AskUserQuestion 2026-09-08, after the earlier "LTX-2 19B distilled NVFP4"
# plan was found not to fit aibox's 16GB card at all -- confirmed against
# Lightricks' own HF repo + maintainer replies, not guessed). Same ComfyUI
# pod as image gen -- no new Deployment, just three more model files under
# the existing /data/comfyui/basedir/models/ hostPath (diffusion_models/
# text_encoders/vae) plus a second, smaller text encoder ComfyUI's own
# DualCLIPLoader also requires (byt5_small_glyphxl_fp16.safetensors --
# only found this was needed empirically, from a real 400 validation error
# submitting the workflow, not from any docs page).
#
# Like Z-Image Turbo, this graph is NOT hand-authored. Captured live via
# window.app.graphToPrompt() in a browser tab pointed at the running
# ComfyUI pod (through a temporary kubectl port-forward), starting from
# ComfyUI's own official "Hunyuan Video 1.5 Text to Video" template
# (search Browse Templates for "Hunyuan"), then editing four widget values
# to match the fp8/480p/distilled checkpoint actually on disk:
#   node 12 (UNETLoader): 720p fp16 checkpoint -> our 480p fp8 distilled one
#   node 124 (EmptyHunyuanVideo15Latent): 1280x720 -> 848x480 (ComfyUI's own
#     node default for this width/height, confirmed via GET /object_info)
#   node 132 (ModelSamplingSD3): shift 7 -> 5
#   node 131 (CFGGuider): cfg 6 -> 1
# (shift/cfg values per the template's own embedded reference table for the
# "480p_t2v_distilled" row -- distilled checkpoints use cfg=1, non-distilled
# need cfg=6; getting this wrong doesn't error, it just generates a much
# worse video, so it's worth getting right rather than leaving the loaded
# template's non-distilled defaults in place).
# Submitted for real via POST /prompt from that same browser tab (bypassing
# this server entirely, direct against ComfyUI) -- CONFIRMED end-to-end: a
# real 848x480, 49-frame (~2s) video generated in ~107s total (including a
# cold model load, this box's first-ever video generation) matching the
# template's own default prompt, fetched back via GET /view and visually
# confirmed to be a real, coherent video (not noise/garbage).
#
# NOT YET VERIFIED: delivery to Telegram. Image delivery took three passes
# to get right (see generate_image's docstring/comment above) because
# Hermes doesn't auto-attach tool output -- it scans reply TEXT for a path.
# That fix (bare path + MEDIA:<path> line) is carried over here on the
# assumption it's format-agnostic, but that assumption is UNTESTED for
# video specifically -- Hermes's Telegram gateway might handle
# video-attachment differently from photo-attachment (separate sendVideo
# vs sendPhoto-style call). First real Telegram test after deploying this
# will confirm or break that assumption -- don't treat this as done until
# a real video actually lands in Telegram, the same bar image gen had to
# clear.
VIDEO_WORKFLOW_TEMPLATE = {
    "8": {
        "inputs": {"samples": ["127", 0], "vae": ["10", 0]},
        "class_type": "VAEDecode",
        "_meta": {"title": "VAE Decode"},
    },
    "10": {
        "inputs": {"vae_name": "hunyuanvideo15_vae_fp16.safetensors"},
        "class_type": "VAELoader",
        "_meta": {"title": "Load VAE"},
    },
    "11": {
        "inputs": {
            "clip_name1": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "clip_name2": "byt5_small_glyphxl_fp16.safetensors",
            "type": "hunyuan_video_15",
            "device": "default",
        },
        "class_type": "DualCLIPLoader",
        "_meta": {"title": "Load CLIP (Dual)"},
    },
    "12": {
        "inputs": {
            "unet_name": "hunyuanvideo1.5_480p_t2v_cfg_distilled_fp8_scaled.safetensors",
            "weight_dtype": "default",
        },
        "class_type": "UNETLoader",
        "_meta": {"title": "Load Diffusion Model"},
    },
    "44": {
        "inputs": {"text": "", "clip": ["11", 0]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Positive Prompt)"},
    },
    "93": {
        "inputs": {"text": "", "clip": ["11", 0]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Negative Prompt)"},
    },
    "101": {
        "inputs": {"fps": 24, "bit_depth": "auto", "color_space": "sRGB", "images": ["8", 0]},
        "class_type": "CreateVideo",
        "_meta": {"title": "Create Video"},
    },
    "102": {
        "inputs": {
            "filename_prefix": "video/hunyuan_video_1.5",
            "format": "auto",
            "format.codec": "h264",
            "format.codec.encoding": "auto",
            "codec": "auto",
            "video": ["101", 0],
        },
        "class_type": "SaveVideo",
        "_meta": {"title": "Save Video"},
    },
    "124": {
        "inputs": {"width": 848, "height": 480, "length": 49, "batch_size": 1},
        "class_type": "EmptyHunyuanVideo15Latent",
        "_meta": {"title": "Empty HunyuanVideo 1.5 Latent"},
    },
    "127": {
        "inputs": {
            "noise": ["129", 0],
            "guider": ["131", 0],
            "sampler": ["130", 0],
            "sigmas": ["128", 0],
            "latent_image": ["124", 0],
        },
        "class_type": "SamplerCustomAdvanced",
        "_meta": {"title": "SamplerCustomAdvanced"},
    },
    "128": {
        "inputs": {"scheduler": "simple", "steps": 20, "denoise": 1, "model": ["12", 0]},
        "class_type": "BasicScheduler",
        "_meta": {"title": "BasicScheduler"},
    },
    "129": {
        "inputs": {"noise_seed": 42},
        "class_type": "RandomNoise",
        "_meta": {"title": "RandomNoise"},
    },
    "130": {
        "inputs": {"sampler_name": "euler"},
        "class_type": "KSamplerSelect",
        "_meta": {"title": "KSamplerSelect"},
    },
    "131": {
        "inputs": {"cfg": 1, "model": ["132", 0], "positive": ["44", 0], "negative": ["93", 0]},
        "class_type": "CFGGuider",
        "_meta": {"title": "CFG Guider"},
    },
    "132": {
        "inputs": {"shift": 5, "model": ["12", 0]},
        "class_type": "ModelSamplingSD3",
        "_meta": {"title": "ModelSamplingSD3"},
    },
}

# A cold model load + a real generation took ~107s for a 49-frame test clip
# on this box -- a full-length (121-frame) request or a cold-load-plus-long
# video could run well past image gen's 180s budget. Generous, not tuned.
VIDEO_POLL_TIMEOUT_SECONDS = 600
VIDEO_BUSY_TTL_SECONDS = 900  # safety-net only, same finally-block pattern as images

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
) -> str:
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

    Returns:
        A short confirmation message containing a `MEDIA:<path>` line for
        the generated image -- include that line VERBATIM, on its own, in
        your reply to the user (in addition to whatever else you say) so
        Hermes's gateway actually attaches the file instead of you just
        describing the image. This is Hermes's own documented attachment
        syntax, not something ComfyUI or this tool understands.
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

            os.makedirs(SHARED_OUTPUT_DIR, exist_ok=True)
            out_path = os.path.join(SHARED_OUTPUT_DIR, f"{uuid.uuid4().hex}.png")
            with open(out_path, "wb") as f:
                f.write(image_bytes)
            logger.info("saved image to shared path %s", out_path)

            return (
                f"Image generated successfully and saved to {out_path}\n\n"
                f"To actually attach it, include this exact line on its own "
                f"in your reply (don't just describe the image):\n"
                f"MEDIA:{out_path}"
            )
        finally:
            # asyncio.shield() so this cleanup still finishes even if the
            # whole call gets cancelled from outside (e.g. Hermes's MCP
            # session hitting its idle timeout mid-generation, confirmed as
            # a real occurrence 2026-09-09 -- without shield(), a
            # cancellation can rip through these awaits too and leave
            # media-gen:busy stuck in Redis for the rest of its TTL even
            # though this finally block did technically run).
            await asyncio.shield(_free_comfyui_vram(client))
            await asyncio.shield(_redis.delete(BUSY_KEY))


@mcp.tool()
async def generate_video(
    prompt: str,
    negative_prompt: str = "",
    length: int = 49,
    steps: int = 20,
    seed: int = -1,
) -> str:
    """Generate a short video locally with HunyuanVideo 1.5 (via ComfyUI on aibox's GPU).

    Fixed at 848x480 (480p) and cfg=1/shift=5 -- these are locked to the
    specific fp8/480p/cfg-distilled checkpoint installed on this box, not
    independently tunable (a 720p or non-distilled checkpoint isn't
    installed here; wrong cfg/shift values silently produce a worse video
    rather than erroring). Like generate_image, this fully unloads whatever
    chat model is resident to give the whole GPU to ComfyUI, then reloads
    the chat model lazily afterward. Any chat message sent elsewhere during
    that window is automatically redirected to a cloud model.

    A real 49-frame/~2s clip took ~107s end-to-end on this box (including a
    cold model load) -- expect video generation to take noticeably longer
    than generate_image's ~20-40s, and scale roughly with `length`.

    Args:
        prompt: What the video should show. Describe subject, motion,
            camera movement, and style -- motion description matters more
            here than for a still image.
        negative_prompt: What to avoid. Empty by default -- this checkpoint
            is cfg-distilled (cfg=1), so the negative prompt has little to
            no effect; kept as a parameter for parity with generate_image
            and in case that changes.
        length: Number of frames at 24fps (e.g. 49 =~ 2s, 121 =~ 5s).
            Longer videos take proportionally longer to generate. Must be
            1 + a multiple of 4 (ComfyUI's own constraint for this node).
        steps: Sampling steps. Default 20 (this workflow's own template
            default, reduced from the Hunyuan team's original 50 for
            speed -- raise it if quality matters more than turnaround).
        seed: Fixed seed for reproducibility. Default -1 picks a random
            seed each call.

    Returns:
        A short confirmation message containing a `MEDIA:<path>` line for
        the generated video -- include that line VERBATIM, on its own, in
        your reply to the user, the same as generate_image's output. NOTE:
        unlike image delivery, this has not yet been confirmed to actually
        reach Telegram as a playable video -- if it doesn't, that's the
        first thing to check (see the top-of-file comment on this).
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")

    already_busy = await _redis.set(BUSY_KEY, "1", ex=VIDEO_BUSY_TTL_SECONDS, nx=True)
    if not already_busy:
        raise RuntimeError(
            "Media generation is already in progress. ComfyUI only holds "
            "the GPU for one job at a time -- try again in a moment."
        )

    resolved_seed = random.randint(0, 2**31 - 1) if seed < 0 else seed

    async with httpx.AsyncClient() as client:
        try:
            await _unload_chat_model(client)

            workflow = copy.deepcopy(VIDEO_WORKFLOW_TEMPLATE)
            workflow["44"]["inputs"]["text"] = prompt
            workflow["93"]["inputs"]["text"] = negative_prompt
            workflow["124"]["inputs"]["length"] = length
            workflow["128"]["inputs"]["steps"] = steps
            workflow["129"]["inputs"]["noise_seed"] = resolved_seed

            submit_resp = await client.post(
                f"{COMFYUI_URL}/prompt", json={"prompt": workflow}, timeout=30.0
            )
            submit_resp.raise_for_status()
            prompt_id = submit_resp.json()["prompt_id"]
            logger.info("submitted video prompt_id=%s seed=%s length=%s", prompt_id, resolved_seed, length)

            deadline = time.monotonic() + VIDEO_POLL_TIMEOUT_SECONDS
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
                    f"ComfyUI did not finish video prompt {prompt_id} within {VIDEO_POLL_TIMEOUT_SECONDS}s"
                )

            status = history.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI reported an error for prompt {prompt_id}: {status}")

            # SaveVideo's output uses the same "images" key ComfyUI uses for
            # SaveImage (confirmed empirically against a real completed job,
            # not documented anywhere obvious) -- node "102" here, not "9".
            videos = history.get("outputs", {}).get("102", {}).get("images", [])
            if not videos:
                raise RuntimeError(f"ComfyUI finished prompt {prompt_id} but produced no video output")
            video_ref = videos[0]

            view_resp = await client.get(
                f"{COMFYUI_URL}/view",
                params={
                    "filename": video_ref["filename"],
                    "subfolder": video_ref.get("subfolder", ""),
                    "type": video_ref.get("type", "output"),
                },
                timeout=60.0,
            )
            view_resp.raise_for_status()
            video_bytes = view_resp.content
            logger.info("fetched output video %s (%d bytes)", video_ref["filename"], len(video_bytes))

            os.makedirs(SHARED_OUTPUT_DIR, exist_ok=True)
            out_path = os.path.join(SHARED_OUTPUT_DIR, f"{uuid.uuid4().hex}.mp4")
            with open(out_path, "wb") as f:
                f.write(video_bytes)
            logger.info("saved video to shared path %s", out_path)

            return (
                f"Video generated successfully and saved to {out_path}\n\n"
                f"To actually attach it, include this exact line on its own "
                f"in your reply (don't just describe the video):\n"
                f"MEDIA:{out_path}"
            )
        finally:
            # same shield() reasoning as generate_image's finally above --
            # video's lock is even more important to get right, since its
            # TTL is 3x longer (VIDEO_BUSY_TTL_SECONDS=900 vs 300), so a
            # stuck lock here means up to 15 minutes of every chat message
            # AND every future generate call getting bounced, not 5.
            await asyncio.shield(_free_comfyui_vram(client))
            await asyncio.shield(_redis.delete(BUSY_KEY))


if __name__ == "__main__":
    # streamable_http_path defaults to "/mcp" (confirmed by inspecting the
    # installed mcp package's MCPServer.run_streamable_http_async) -- that's
    # the path k8s/p3/hermes-agent.yaml's mcp_servers.media-gen.url assumes.
    # json_response=True: plain JSON responses instead of SSE streaming --
    # simpler for a single-tool server with no need for server-initiated
    # push during a call.
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000, json_response=True)
