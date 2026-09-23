"""media-augment-mcp: an HTTP MCP server exposing image editing/augmentation
(Qwen-Image-2.1: background swaps, compositing, reference-image edits) to
Hermes.

Split out as its OWN service from media-gen-mcp (Eldar's explicit call,
2026-09-23) rather than adding an edit_image tool there -- this is the
first tool in the media-gen family that needs an INPUT image, not just a
text prompt, and keeping it in its own pod means a bug in the new
input-handling path can't take down the already-working generate_image/
generate_video/generate_music tools. Talks to the SAME ComfyUI instance
and the SAME Redis busy flag as media-gen-mcp (see BUSY_KEY below) --
they still can't run concurrently, they just live in separate pods now.

INPUT IMAGE HANDOFF -- the one real design problem this service had to
solve, confirmed by reading the actual manifests rather than assumed:
Hermes stages an inbound Telegram photo at /opt/data/cache/images/ inside
ITS OWN pod, but that path is a PersistentVolumeClaim (hermes-data),
mounted ONLY into hermes-agent's pod -- there is no volume plumbing
anywhere in this repo that would let a different pod read it directly
(confirmed by grepping k8s/p3/hermes-agent.yaml's volumes: block). This
is the exact same constraint vault-mcp's ingest_file already documents
and works around (see hermes-agent.yaml's tool-use notes: "ingest_file
only accepts a file_path already inside the vault or /shared-output/ --
if the file is sitting in your own cache instead, copy it into the
vault's Attachments folder yourself first"). So this tool follows the
same established convention instead of inventing a new one: Hermes must
copy the cached photo into SHARED_OUTPUT_DIR (the existing
/data/media-gen-output hostPath, already mounted read-write into
hermes-agent's pod for the opposite direction -- see media-gen-mcp's own
SHARED_OUTPUT_DIR comment) BEFORE calling edit_image, and pass that
shared-path as image_path. Hermes already has plain file-copy tools
available (local terminal backend, see model.terminal in
hermes-agent.yaml), so this needs no new capability on its side, just a
documented calling convention -- see edit_image's docstring.

From SHARED_OUTPUT_DIR this tool then copies the file a second time, into
COMFYUI_INPUT_DIR -- ComfyUI's own LoadImage node only reads a filename
out of its own input/ folder, not an arbitrary path, so there's no way
around this second hop. COMFYUI_INPUT_DIR is a new hostPath
(/data/comfyui/basedir/input, the same directory ComfyUI's own
Deployment already mounts as part of its basedir volume -- see
k8s/media-gen/comfyui-deployment.yaml) added to THIS pod so it can write
there directly.

WORKFLOW JSON -- started as a HAND CONVERSION from the litegraph/UI-export
JSON Eldar's Windows session handed off (Image_Augmentation.json -- the
nodes/links/widgets_values array format the ComfyUI frontend saves, NOT
the flat {"<id>": {"class_type", "inputs"}} format the real /prompt API
needs), then VERIFIED 2026-09-23 against a real successful generation on
aibox (Qwen_image_2.1_edit_00001, 359s, background swap on a real family
photo, correct result) by capturing window.app.graphToPrompt() from that
run's loaded graph in ComfyUI's own web UI -- same method media-gen-mcp's
WORKFLOW_TEMPLATE was built from. The captured payload matched the hand
conversion node-for-node, input-for-input, with no differences. Node IDs
below match the source JSON's own id numbering, not renumbered.
"""
import asyncio
import copy
import logging
import os
import random
import shutil
import time
import uuid

import httpx
import redis.asyncio as redis
# mcp>=2.0 renamed FastMCP -> MCPServer, see media-gen-mcp/server.py's own
# top-of-file comment for the full story -- same package, same import path.
from mcp.server.mcpserver import Context, MCPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s media-augment-mcp %(levelname)s: %(message)s")
logger = logging.getLogger("media-augment-mcp")

LLAMA_SWAP_URL = os.environ.get("LLAMA_SWAP_URL", "http://llama-swap.ai-models.svc.cluster.local:8080")
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://comfyui.ai-models.svc.cluster.local:8188")
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis.ai-memory.svc.cluster.local:6379/0")
# For _expand_image_reference (see comment above that function). Same
# base_url/auth pattern media-gen-mcp/server.py already uses to reach
# LiteLLM for its own character-reference expansion -- Bearer auth against
# LITELLM_MASTER_KEY, sourced from the same litellm-secret Secret.
LITELLM_URL = os.environ.get("LITELLM_URL", "http://litellm.ai-models.svc.cluster.local:4000/v1")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")

# Same hostPath as media-gen-mcp/hermes-agent's SHARED_OUTPUT_DIR -- reused
# here for BOTH directions (Hermes drops the source photo here for us to
# read, we drop the edited result here for Hermes to attach) rather than
# adding a second hostPath just for inputs. One shared drop folder is
# simpler than two, and nothing here needs them kept apart.
SHARED_OUTPUT_DIR = os.environ.get("SHARED_OUTPUT_DIR", "/shared-output")

# NEW volume, not shared with any other pod today -- see the top-of-file
# comment for why this hop exists. Must be the same host directory ComfyUI
# itself reads its input/ folder from (comfyui-deployment.yaml's basedir
# hostPath covers this already; we just also mount the input/ subpath here).
COMFYUI_INPUT_DIR = os.environ.get("COMFYUI_INPUT_DIR", "/comfyui-input")

# SAME key media-gen-mcp uses -- this has to be one shared flag, not a
# per-service one, since the whole point is stopping generate_image/
# generate_video/generate_music and edit_image from ever racing each
# other (or a chat request) for the 5070 Ti. See media-gen-mcp/server.py's
# own top-of-file comment for the full concurrency-model writeup; this
# service follows the identical model, just duplicated rather than shared
# as a library (matches how generate_image/generate_video/generate_music
# already each carry their own copy of this same pattern in that file).
BUSY_KEY = "media-gen:busy"
BUSY_TTL_SECONDS = 300  # safety-net only -- the finally block is the primary clear path

# Settle time after ComfyUI's /free returns 200, before actually clearing
# the busy flag -- same real OOM-race fix as media-gen-mcp's
# VRAM_FREE_SETTLE_SECONDS (2026-09-09 incident, see that file's comment
# for the full story). Qwen-Image-2.1 uses plain safetensors checkpoints
# (int8_convrot), not GGUF, so it's assumed to behave like Z-Image Turbo's
# /free path rather than Wan2.2's GGUF stuck-VRAM one -- NOT YET CONFIRMED
# against a real run; if VRAM doesn't actually drop after /free, switch
# this tool to the harder _restart_comfyui_and_wait_ready pattern
# media-gen-mcp's generate_video/generate_music already use instead.
VRAM_FREE_SETTLE_SECONDS = 3

# A real warm-ComfyUI generation on aibox (2026-09-23 manual verification
# run) took 359s end to end -- already past the original 300s value this
# constant was first set to, so that was a confirmed-too-short bug, not
# just a cautious guess that needed padding. 600s gives real margin above
# that observed number, plus room for a cold load (first call after
# ComfyUI restarts, or after any other model evicted it) which can run
# longer still given Qwen-Image-2.1's three files sum to ~16.1GB -- much
# larger than Z-Image Turbo's single 12.3GB checkpoint.
POLL_TIMEOUT_SECONDS = 600
POLL_INTERVAL_SECONDS = 1.0
PROGRESS_REPORT_INTERVAL_SECONDS = 20  # see media-gen-mcp's own comment on this same constant

# TextEncodeQwenImage21 requires the source photo to be referenced in the
# prompt text via the exact placeholder tag <image1> (see server.py's
# top-of-file WORKFLOW JSON note) -- but that's an implementation detail
# of this one ComfyUI node, not something a caller should have to know or
# type. Same pattern media-gen-mcp's generate_video already uses for a
# different translation problem (named characters Wan2.2 doesn't
# recognize -- see that file's CHARACTER_EXPANSION_SYSTEM_PROMPT comment
# for the full precedent this follows): a one-shot system-prompted rewrite
# through the local expert chat model, bypassing LiteLLM's "auto" semantic
# router the same way and for the same reason (a single non-conversational
# rewrite, not a user-facing chat turn the router's classifier was tuned
# for). Eldar's explicit instruction (2026-09-23): "I want some model to
# insert this mandatory stuff leaving me something easy to explain,
# without writing those tags."
IMAGE_TAG_EXPANSION_MODEL = "qwen3.8-27b-Expert"

IMAGE_TAG_EXPANSION_SYSTEM_PROMPT = """You rewrite image-editing instructions for a diffusion model that requires the source photo to be referenced with the exact placeholder tag <image1> somewhere in the text -- it has no other way of knowing what <image1> shows when it reads the prompt.

Rewrite the user's instruction:
- Insert <image1> at the point(s) where the instruction refers to the photo/image/picture being edited (words like "it", "this", "the photo", "the image", or an implied subject with no referring word at all).
- Make sure <image1> appears at least once, tied to the main subject being edited.
- Keep everything else in the instruction unchanged: the requested edit, style, mood, what to keep unchanged.
- Output ONLY the rewritten instruction text. No preamble, no explanation, no surrounding quotes."""


# API-format translation of Image_Augmentation.json -- verified 2026-09-23
# against a real successful generation's own graphToPrompt() capture, see
# the top-of-file WORKFLOW JSON note.
WORKFLOW_TEMPLATE = {
    "1": {
        "class_type": "LoadImage",
        "inputs": {"image": "PLACEHOLDER.png"},  # overwritten per-call in edit_image()
    },
    "2": {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": "qwen_image_2.1_int8_convrot.safetensors",
            "weight_dtype": "default",
        },
    },
    "9": {
        "class_type": "QwenImage21Cache",
        "inputs": {"model": ["2", 0], "device": "auto", "dtype": "default"},
    },
    "3": {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": "qwen3vl_8b_int8_convrot.safetensors",
            "type": "qwen_image",
            "device": "default",
        },
    },
    "4": {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"},
    },
    "5": {
        "class_type": "TextEncodeQwenImage21",
        "inputs": {
            "clip": ["3", 0],
            "vae": ["4", 0],
            "images.image_1": ["1", 0],
            "prompt": "PLACEHOLDER",  # overwritten per-call
            "negative_prompt": "",
            "resolution": 0,  # 0 = keep reference image at native size (rounded to 32px)
        },
    },
    "6": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["9", 0],
            "positive": ["5", 0],
            "negative": ["5", 1],
            "latent_image": ["5", 2],
            "seed": 0,  # overwritten per-call
            "steps": 25,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1,
        },
    },
    "7": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["6", 0], "vae": ["4", 0]},
    },
    "8": {
        "class_type": "SaveImageAdvanced",
        "inputs": {
            "images": ["7", 0],
            "filename_prefix": "Qwen_image_2.1_edit",
            "format": "png",
            "format.bit_depth": "8-bit",
            "format.input_color_space": "sRGB",
        },
    },
}

mcp = MCPServer("media-augment")
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


async def _expand_image_reference(client: httpx.AsyncClient, prompt: str) -> str:
    """Ask the local expert chat model to insert the mandatory <image1> tag
    into `prompt` so the caller never has to know/type it (see the comment
    above IMAGE_TAG_EXPANSION_SYSTEM_PROMPT). UNLIKE media-gen-mcp's
    _expand_character_references, this one isn't allowed to just fall back
    to the original prompt on failure -- <image1> is structurally required
    by TextEncodeQwenImage21, not a quality nice-to-have, so a failed or
    non-compliant LLM rewrite instead falls back to a deterministic wrap
    that's guaranteed to contain the tag.
    """
    try:
        headers = {"Authorization": f"Bearer {LITELLM_API_KEY}"} if LITELLM_API_KEY else {}
        resp = await client.post(
            f"{LITELLM_URL}/chat/completions",
            headers=headers,
            json={
                "model": IMAGE_TAG_EXPANSION_MODEL,
                "messages": [
                    {"role": "system", "content": IMAGE_TAG_EXPANSION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 300,
            },
            # Same slow-local-27B budget media-gen-mcp's equivalent call uses.
            timeout=90.0,
        )
        resp.raise_for_status()
        expanded = resp.json()["choices"][0]["message"]["content"].strip()
        if expanded and "<image1>" in expanded:
            logger.info("image-tag expansion: %r -> %r", prompt, expanded)
            return expanded
        logger.warning(
            "image-tag expansion produced no usable <image1> tag (got %r) -- using deterministic fallback",
            expanded,
        )
    except Exception as exc:  # noqa: BLE001 -- fall through to the deterministic wrap below
        logger.warning(
            "image-tag expansion failed (%s: %s) -- using deterministic fallback", type(exc).__name__, exc
        )
    # Deterministic fallback: always contains a valid <image1> reference,
    # even if the LLM call errored or ignored the instruction.
    return f"In <image1>: {prompt}"


@mcp.tool()
async def edit_image(
    prompt: str,
    image_path: str,
    ctx: Context,
    negative_prompt: str = "",
    seed: int = -1,
) -> str:
    """Edit or augment an existing image with Qwen-Image-2.1 (via ComfyUI on aibox's GPU).

    Use this for requests like "change the background to X", "put this
    person into a different scene", or any edit to a photo that already
    exists -- NOT for generating a brand-new image from scratch (use
    generate_image for that instead).

    Before calling this tool, the source photo must already be sitting
    somewhere this tool can read it -- your own /opt/data/cache/images/
    scratch cache is NOT reachable from this pod. Copy the file into
    /shared-output/ yourself first (you have write access there, same
    place generate_image/generate_video/generate_music deliver their own
    output), then pass that /shared-output/... path as image_path.

    Describe the edit naturally -- e.g. "change the background to a dense
    jungle, keep the person exactly as they appear". You don't need to
    reference the image with any special tag; this tool handles that
    translation internally.

    This fully unloads whatever chat model is currently resident to give
    the whole GPU to image editing, then reloads the chat model lazily
    afterward -- takes over the GPU for the duration of the call. Any
    chat message sent elsewhere during that window is automatically
    redirected to a cloud model instead of queueing or failing.

    Args:
        prompt: The edit instruction, in plain language -- no special
            tags needed.
        image_path: Absolute path to the source photo, already copied
            into /shared-output/ (see above).
        negative_prompt: What to avoid in the result. Optional.
        seed: Fixed seed for reproducibility. Default -1 picks a random
            seed each call.

    Returns:
        A short confirmation message containing a `MEDIA:<path>` line for
        the edited image -- include that line VERBATIM, on its own, in
        your reply to the user (in addition to whatever else you say) so
        Hermes's gateway actually attaches the file instead of you just
        describing the image. This is Hermes's own documented attachment
        syntax, not something ComfyUI or this tool understands.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not image_path or not os.path.isfile(image_path):
        raise ValueError(
            f"image_path {image_path!r} doesn't exist -- copy the source photo into "
            f"{SHARED_OUTPUT_DIR}/ first, then pass that path"
        )

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

            # Second hop: ComfyUI's LoadImage node only reads a bare
            # filename out of its own input/ folder -- copy the source
            # photo there under a collision-proof name.
            os.makedirs(COMFYUI_INPUT_DIR, exist_ok=True)
            input_filename = f"{uuid.uuid4().hex}{os.path.splitext(image_path)[1] or '.png'}"
            shutil.copyfile(image_path, os.path.join(COMFYUI_INPUT_DIR, input_filename))
            logger.info("staged input image as %s", input_filename)

            expanded_prompt = await _expand_image_reference(client, prompt)

            workflow = copy.deepcopy(WORKFLOW_TEMPLATE)
            workflow["1"]["inputs"]["image"] = input_filename
            workflow["5"]["inputs"]["prompt"] = expanded_prompt
            workflow["5"]["inputs"]["negative_prompt"] = negative_prompt
            workflow["6"]["inputs"]["seed"] = resolved_seed

            submit_resp = await client.post(
                f"{COMFYUI_URL}/prompt", json={"prompt": workflow}, timeout=30.0
            )
            submit_resp.raise_for_status()
            prompt_id = submit_resp.json()["prompt_id"]
            logger.info("submitted prompt_id=%s seed=%s", prompt_id, resolved_seed)

            poll_start = time.monotonic()
            last_progress = poll_start
            deadline = poll_start + POLL_TIMEOUT_SECONDS
            history = None
            while time.monotonic() < deadline:
                hist_resp = await client.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=15.0)
                hist_resp.raise_for_status()
                body = hist_resp.json()
                if prompt_id in body:
                    history = body[prompt_id]
                    break
                now = time.monotonic()
                if now - last_progress >= PROGRESS_REPORT_INTERVAL_SECONDS:
                    try:
                        await ctx.report_progress(
                            progress=now - poll_start, total=POLL_TIMEOUT_SECONDS,
                            message="Qwen-Image-2.1 still rendering...",
                        )
                    except Exception as exc:  # noqa: BLE001 -- best effort, must not break generation
                        logger.warning("report_progress failed (%s: %s)", type(exc).__name__, exc)
                    last_progress = now
                await asyncio.sleep(POLL_INTERVAL_SECONDS)

            if history is None:
                raise TimeoutError(
                    f"ComfyUI did not finish prompt {prompt_id} within {POLL_TIMEOUT_SECONDS}s"
                )

            status = history.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI reported an error for prompt {prompt_id}: {status}")

            images = history.get("outputs", {}).get("8", {}).get("images", [])
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
                f"Image edited successfully and saved to {out_path}\n\n"
                f"To actually attach it, include this exact line on its own "
                f"in your reply (don't just describe the image):\n"
                f"MEDIA:{out_path}"
            )
        finally:
            # asyncio.shield() so this cleanup still finishes even if the
            # call gets cancelled from outside -- same real incident
            # media-gen-mcp's generate_image hit (2026-09-09, MCP idle
            # session timeout mid-generation), same fix.
            await asyncio.shield(_free_comfyui_vram(client))
            await asyncio.sleep(VRAM_FREE_SETTLE_SECONDS)
            await asyncio.shield(_redis.delete(BUSY_KEY))


if __name__ == "__main__":
    # streamable_http_path defaults to "/mcp" -- see media-gen-mcp's own
    # comment on this same call for why json_response is deliberately NOT
    # set (progress-notification heartbeats get silently dropped in JSON
    # mode, confirmed against a real production incident there).
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
