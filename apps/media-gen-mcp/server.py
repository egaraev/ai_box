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
# For _expand_character_references (see comment above that function). Same
# base_url/auth pattern hermes-agent.yaml and open-webui.yaml already use to
# call LiteLLM (k8s/p1/litellm.yaml) -- Bearer auth against its
# LITELLM_MASTER_KEY, sourced from the same litellm-secret Secret.
LITELLM_URL = os.environ.get("LITELLM_URL", "http://litellm.ai-models.svc.cluster.local:4000/v1")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")
# Must be the SAME path inside both this container and hermes-agent's --
# see the "DELIVERY" note above. A hostPath volume, not a PVC: this is a
# single-node cluster, so both pods always land on the same node anyway,
# and hostPath matches the pattern already used for /data/models,
# /data/comfyui elsewhere in this repo instead of introducing a new PVC.
SHARED_OUTPUT_DIR = os.environ.get("SHARED_OUTPUT_DIR", "/shared-output")

BUSY_KEY = "media-gen:busy"
BUSY_TTL_SECONDS = 300  # safety-net only -- the finally block is the primary clear path

# How long to wait after ComfyUI's /free returns 200 before actually clearing
# the busy flag -- see the comment on this constant's use in generate_video's
# finally block (real live OOM this fixed, 2026-09-09) for the full story.
# Short version: /free returning 200 doesn't mean CUDA has actually finished
# reclaiming the VRAM yet, and clearing busy immediately lets the very next
# chat request race that teardown straight into an OOM.
VRAM_FREE_SETTLE_SECONDS = 3

# Poll /history for up to this long before giving up. Z-Image Turbo took
# ~18-20s end-to-end in the verified live test; 180s leaves generous room
# for a cold model load on top of that without hanging forever on a stuck job.
POLL_TIMEOUT_SECONDS = 180
POLL_INTERVAL_SECONDS = 1.0

DEFAULT_NEGATIVE_PROMPT = "low quality, bad anatomy, extra digits, missing digits, extra limbs, missing limbs"

# Wan2.2's own official ComfyUI template default negative prompt (kept
# verbatim, not translated -- it's what the checkpoint was tuned/tested
# against). Unlike HunyuanVideo's cfg-distilled checkpoint, Wan2.2 T2V-A14B
# uses real cfg=3.5 guidance, so this actually has an effect.
DEFAULT_VIDEO_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，裸露，NSFW"
)

# Wan2.2 T2V-A14B (like most open video models) has weak-to-no recall of
# named characters/franchises/celebrities/real people -- confirmed
# empirically 2026-09-09 against the real deployed graph: the prompt
# "Spacemarine from Starcraft video game is playing soccer" produced a
# plain human in a soccer kit, no armor at all. Rewording the SAME request
# as a plain visual description instead ("hulking armored super-soldier,
# massive bulky power armor, wide shoulder pauldrons, glowing visor
# helmet...") produced exactly the intended result on the first try. So
# generate_video() now runs every prompt through the local expert chat
# model first (see _expand_character_references below) to do that
# translation automatically, rather than relying on Eldar or Hermes to
# phrase every request as a visual description by hand.
#
# Bypasses LiteLLM's own "auto" semantic router entirely (k8s/p1/litellm.yaml's
# custom_router_hook.py, arch-router-1.5b classifying casual/reasoning/etc)
# -- this is a single one-shot system-prompted rewrite with no conversation
# history, not a user-facing chat turn, and the classifier was built and
# tuned for the latter. There's no good reason to gamble a real character's
# visual accuracy on a 1.5B classifier's guess landing this in the "casual"
# bucket. Calls the 27B reasoning/expert model directly by its litellm
# model_list name instead, for reliability over speed.
VIDEO_PROMPT_EXPANSION_MODEL = "qwen3.8-27b-Expert"

CHARACTER_EXPANSION_SYSTEM_PROMPT = """You rewrite prompts for a text-to-video diffusion model that does NOT reliably recognize named characters, franchises, celebrities, or real people -- it only understands plain visual descriptions (physical build, clothing/armor, colors, distinguishing features, props/weapons).

Rewrite the user's request:
- If it names a specific character, franchise, celebrity, or real person, REPLACE that name with a detailed visual description of how they actually look (build, outfit/armor, colors, notable features, iconic props/weapons). Do not keep the name itself in the output.
- Keep everything else in the request unchanged: setting, action, camera/style notes, mood.
- If the request has no named character/franchise/celebrity/person in it already, return it essentially unchanged -- light cleanup only, do not invent new details.
- Output ONLY the rewritten prompt text. No preamble, no explanation, no surrounding quotes."""


async def _expand_character_references(client: httpx.AsyncClient, prompt: str) -> str:
    """Best-effort: ask the local expert chat model to translate any named
    character/franchise/celebrity/real-person reference in `prompt` into a
    plain visual description (see the comment above
    CHARACTER_EXPANSION_SYSTEM_PROMPT for why -- Wan2.2 doesn't reliably
    recognize names). Falls back to the original prompt unchanged on any
    failure -- this is a quality enhancement, not something that should
    ever block a video generation.
    """
    try:
        headers = {"Authorization": f"Bearer {LITELLM_API_KEY}"} if LITELLM_API_KEY else {}
        resp = await client.post(
            f"{LITELLM_URL}/chat/completions",
            headers=headers,
            json={
                "model": VIDEO_PROMPT_EXPANSION_MODEL,
                "messages": [
                    {"role": "system", "content": CHARACTER_EXPANSION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.4,
                "max_tokens": 300,
            },
            # The expert model is a slow local 27B model (Eldar's own
            # observation earlier this project: ~30-50s per response on
            # this box) -- generous, not tuned.
            timeout=90.0,
        )
        resp.raise_for_status()
        expanded = resp.json()["choices"][0]["message"]["content"].strip()
        if not expanded:
            raise ValueError("expert model returned empty content")
        logger.info("prompt expansion: %r -> %r", prompt, expanded)
        return expanded
    except Exception as exc:  # noqa: BLE001 -- best effort, must never block a real generation
        logger.warning(
            "prompt expansion failed (%s: %s) -- using original prompt unchanged", type(exc).__name__, exc
        )
        return prompt


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

# Video: Wan2.2 T2V-A14B (Alibaba), GGUF Q4_K_M quantized -- replaces the
# earlier HunyuanVideo 1.5 setup (2026-09-08 - 09) after Eldar asked for
# less hallucination / better precision / longer videos and research
# pointed at switching model families rather than tuning the cfg-distilled
# Hunyuan checkpoint (whose cfg is baked to 1, so negative prompts do
# essentially nothing there). Wan2.2 T2V-A14B is a genuine two-expert MoE
# (27B total params, 14B active per step: a "high-noise" expert for the
# first half of denoising, a "low-noise" expert for the second half) with
# real non-distilled CFG guidance (cfg=3.5), so negative prompts actually
# matter here -- see DEFAULT_VIDEO_NEGATIVE_PROMPT above.
#
# Two open questions from research had to be settled empirically before
# committing to this, both confirmed live against the real GGUF files on
# this box (2026-09-09) rather than trusted from docs/community claims --
# the project has been burned by unverified community VRAM numbers before
# (the LTX-2 mis-sizing mistake, 2026-09-08, is why LTX-2.3 was researched
# but NOT picked here despite promising-looking community posts):
#   1. Do the two experts need to be simultaneously VRAM-resident? NO --
#      confirmed via a real nvidia-smi trace (1s samples) through an
#      actual generation: the high-noise expert loads + samples, plateaus
#      ~13.3-13.5GB, gets EVICTED, then the low-noise expert loads +
#      samples, plateaus at the same ~13.2-13.3GB ceiling -- not additive.
#      ComfyUI's model manager swaps them sequentially, the same way
#      llama-swap swaps chat models.
#   2. Does Q4_K_M actually fit the 16GB card with real headroom? YES --
#      peak observed was 13495MiB against a 16303MiB card (~2.8GB spare)
#      at 480x480 / 33 frames / 20 steps.
#
# Graph captured live via window.app.graphToPrompt() in a browser tab
# against the running ComfyUI pod, starting from ComfyUI's own official
# "Wan 2.2 14B Text to Video" template (Browse Templates -> search
# "Wan2.2"), then:
#   - both "Load Diffusion Model" (UNETLoader, safetensors) nodes swapped
#     for "Unet Loader (GGUF)" (UnetLoaderGGUF) nodes pointing at the
#     Q4_K_M-quantized high/low-noise files actually on disk (the fp8
#     safetensors files the template defaults to were deliberately never
#     downloaded -- only the smaller GGUF quants, for VRAM headroom)
#   - the template's built-in "Lightning LoRA" turbo-mode scaffolding (2
#     LoraLoaderModelOnly nodes + 5 ComfySwitchNode nodes + primitive
#     int/float nodes picking between "Original" and "Lightning" timing/
#     cfg values) stripped out entirely -- not needed, and the Lightning
#     LoRA files (2.29GB) were never downloaded. What's left below is
#     hardcoded to the template's own "Original" (non-turbo) branch
#     values: 20 steps total, split at step 10 (halfway) between experts,
#     cfg=3.5, shift=5 -- the template's own tuned defaults for this
#     checkpoint, not guessed.
# Submitted for real via the ComfyUI UI (Run button) against this exact
# graph -- CONFIRMED end-to-end: a real 480x480, 33-frame (~2s @ 16fps)
# video generated successfully, visually confirmed coherent (not noise),
# with the VRAM behavior described in points 1/2 above.
#
# NOT YET VERIFIED: delivery to Telegram. Image delivery took three passes
# to get right (see generate_image's docstring/comment above) because
# Hermes doesn't auto-attach tool output -- it scans reply TEXT for a path.
# That fix (bare path + MEDIA:<path> line) is carried over here on the
# assumption it's format-agnostic, but that assumption is still UNTESTED
# for video specifically and was already untested under the old Hunyuan
# version this replaces -- first real Telegram test after deploying this
# will confirm or break it.
VIDEO_WORKFLOW_TEMPLATE = {
    "1": {
        "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"},
        "class_type": "CLIPLoader",
        "_meta": {"title": "Load CLIP"},
    },
    "2": {
        "inputs": {"vae_name": "wan_2.1_vae.safetensors"},
        "class_type": "VAELoader",
        "_meta": {"title": "Load VAE"},
    },
    "3": {
        "inputs": {"unet_name": "wan2.2_t2v_high_noise_14B_Q4_K_M.gguf"},
        "class_type": "UnetLoaderGGUF",
        "_meta": {"title": "Unet Loader (GGUF) - High Noise"},
    },
    "4": {
        "inputs": {"unet_name": "wan2.2_t2v_low_noise_14B_Q4_K_M.gguf"},
        "class_type": "UnetLoaderGGUF",
        "_meta": {"title": "Unet Loader (GGUF) - Low Noise"},
    },
    "5": {
        "inputs": {"shift": 5, "model": ["3", 0]},
        "class_type": "ModelSamplingSD3",
        "_meta": {"title": "ModelSamplingSD3 (High Noise)"},
    },
    "6": {
        "inputs": {"shift": 5, "model": ["4", 0]},
        "class_type": "ModelSamplingSD3",
        "_meta": {"title": "ModelSamplingSD3 (Low Noise)"},
    },
    "7": {
        "inputs": {"text": "", "clip": ["1", 0]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Positive Prompt)"},
    },
    "8": {
        "inputs": {"text": DEFAULT_VIDEO_NEGATIVE_PROMPT, "clip": ["1", 0]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Negative Prompt)"},
    },
    "9": {
        "inputs": {"width": 832, "height": 480, "length": 49, "batch_size": 1},
        "class_type": "EmptyHunyuanLatentVideo",
        "_meta": {"title": "Empty Latent Video"},
    },
    "10": {
        "inputs": {
            "add_noise": "enable",
            "noise_seed": 42,
            "steps": 20,
            "cfg": 3.5,
            "sampler_name": "euler",
            "scheduler": "simple",
            "start_at_step": 0,
            "end_at_step": 10,
            "return_with_leftover_noise": "enable",
            "model": ["5", 0],
            "positive": ["7", 0],
            "negative": ["8", 0],
            "latent_image": ["9", 0],
        },
        "class_type": "KSamplerAdvanced",
        "_meta": {"title": "KSampler (Advanced) - High Noise Pass"},
    },
    "11": {
        "inputs": {
            "add_noise": "disable",
            "noise_seed": 0,
            "steps": 20,
            "cfg": 3.5,
            "sampler_name": "euler",
            "scheduler": "simple",
            "start_at_step": 10,
            "end_at_step": 20,
            "return_with_leftover_noise": "disable",
            "model": ["6", 0],
            "positive": ["7", 0],
            "negative": ["8", 0],
            "latent_image": ["10", 0],
        },
        "class_type": "KSamplerAdvanced",
        "_meta": {"title": "KSampler (Advanced) - Low Noise Pass"},
    },
    "12": {
        "inputs": {"samples": ["11", 0], "vae": ["2", 0]},
        "class_type": "VAEDecode",
        "_meta": {"title": "VAE Decode"},
    },
    "13": {
        "inputs": {"fps": 16, "bit_depth": "auto", "color_space": "sRGB", "images": ["12", 0]},
        "class_type": "CreateVideo",
        "_meta": {"title": "Create Video"},
    },
    "14": {
        "inputs": {
            "filename_prefix": "video/wan2.2_t2v",
            "format": "auto",
            "format.codec": "auto",
            "codec": "auto",
            "video": ["13", 0],
        },
        "class_type": "SaveVideo",
        "_meta": {"title": "Save Video"},
    },
}

# Two cold GGUF loads (~9GB each, high-noise then low-noise, sequentially --
# see the VRAM trace note above VIDEO_WORKFLOW_TEMPLATE) plus sampling took
# on the order of a few minutes for a small 480x480/33-frame/20-step test
# clip on this box -- not precisely timed, but clearly longer than the old
# single-checkpoint Hunyuan path since two separate models now load per
# call instead of one. A longer request (more frames/steps) could run well
# past image gen's 180s budget. Generous, not tuned.
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
            # VRAM_FREE_SETTLE_SECONDS (added 2026-09-09, real live failure):
            # see the comment on this same line in generate_video below for
            # the full story -- same race, same fix, applies here too since
            # this finally block has the identical free-then-clear-busy
            # shape.
            await asyncio.sleep(VRAM_FREE_SETTLE_SECONDS)
            await asyncio.shield(_redis.delete(BUSY_KEY))


@mcp.tool()
async def generate_video(
    prompt: str,
    negative_prompt: str = "",
    length: int = 49,
    steps: int = 20,
    seed: int = -1,
) -> str:
    """Generate a short video locally with Wan2.2 T2V-A14B (via ComfyUI on aibox's GPU).

    Fixed at 832x480 (480p) and cfg=3.5/shift=5 -- these are the checkpoint's
    own tuned defaults (from ComfyUI's official Wan2.2 template), not
    independently tunable. Runs at 16fps (Wan2.2's native rate -- NOT the
    24fps the old HunyuanVideo checkpoint this replaces used, so the same
    `length` now covers slightly more real-world seconds). Two GGUF-
    quantized "expert" models (high-noise, low-noise -- a genuine
    Mixture-of-Experts architecture) load sequentially during a single
    call, never simultaneously (confirmed via a live VRAM trace -- see the
    comment above VIDEO_WORKFLOW_TEMPLATE), so this still fits the 16GB
    card with headroom despite being two ~9GB GGUF files. Like
    generate_image, this fully unloads whatever chat model is resident to
    give the whole GPU to ComfyUI, then reloads the chat model lazily
    afterward. Any chat message sent elsewhere during that window is
    automatically redirected to a cloud model.

    Unlike the old Hunyuan checkpoint (cfg-distilled, cfg baked to 1), this
    one uses real cfg=3.5 guidance, so negative_prompt actually has an
    effect here -- it's no longer just a no-op parameter kept for parity.

    Named characters (e.g. "Tony Stark", "a Spacemarine from Starcraft") are
    handled automatically -- before submitting to ComfyUI, this call runs
    `prompt` through the local expert chat model, which rewrites any named
    character/franchise/celebrity/real-person reference into a plain visual
    description (build, outfit/armor, colors, distinguishing features).
    Wan2.2 doesn't reliably recognize names by themselves (confirmed
    empirically -- see the comment above CHARACTER_EXPANSION_SYSTEM_PROMPT),
    so it's fine, and expected, to reference a character directly in
    `prompt` -- no need to describe their appearance by hand. This adds
    real latency (the expert model itself is slow, up to ~30-50s) on top of
    the video generation time; if that expansion call fails for any reason
    it silently falls back to the original prompt rather than blocking the
    generation, so a failure here degrades quality, not reliability.

    Args:
        prompt: What the video should show. Describe subject, motion,
            camera movement, and style -- motion description matters more
            here than for a still image. Named characters are fine to
            reference directly (see above) -- they get auto-translated to
            a visual description before generation.
        negative_prompt: What to avoid. Empty by default falls back to
            DEFAULT_VIDEO_NEGATIVE_PROMPT (the checkpoint's own tuned
            default) rather than actually submitting an empty string.
        length: Number of frames at 16fps (e.g. 49 =~ 3s). Longer videos
            take proportionally longer to generate. Must be 1 + a multiple
            of 4 (ComfyUI's own constraint for this latent node).
        steps: Total sampling steps, split evenly across the high-noise and
            low-noise expert passes (e.g. 20 -> 10 steps each). Default 20
            (this workflow's own template default).
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
    split_step = max(1, steps // 2)

    async with httpx.AsyncClient() as client:
        try:
            # Must happen BEFORE _unload_chat_model -- this call needs a
            # chat model loaded (it'll trigger llama-swap to lazily load
            # the expert model if it isn't resident already). Doing it
            # after the unload would just force an extra, wasted reload.
            expanded_prompt = await _expand_character_references(client, prompt)

            await _unload_chat_model(client)

            workflow = copy.deepcopy(VIDEO_WORKFLOW_TEMPLATE)
            workflow["7"]["inputs"]["text"] = expanded_prompt
            workflow["8"]["inputs"]["text"] = negative_prompt or DEFAULT_VIDEO_NEGATIVE_PROMPT
            workflow["9"]["inputs"]["length"] = length
            workflow["10"]["inputs"]["steps"] = steps
            workflow["10"]["inputs"]["end_at_step"] = split_step
            workflow["10"]["inputs"]["noise_seed"] = resolved_seed
            workflow["11"]["inputs"]["steps"] = steps
            workflow["11"]["inputs"]["start_at_step"] = split_step

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
            # not documented anywhere obvious) -- node "14" here, not "9".
            videos = history.get("outputs", {}).get("14", {}).get("images", [])
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
            # VRAM_FREE_SETTLE_SECONDS (added 2026-09-09, real live failure):
            # a real "Spacemarine... soccer" generation succeeded end to
            # end (video saved, ComfyUI's /free returned 200) but the very
            # next chat message Hermes sent (to compose its reply) got
            # routed to the now-"not busy" local qwen3.8-27b-Expert and
            # immediately OOM'd trying to allocate ~12.7GB -- confirmed via
            # llama-swap's own upstream log (cudaMalloc failed: out of
            # memory), repeated across 3-4 retries. Root cause: ComfyUI's
            # /free endpoint returning 200 does not mean CUDA has actually
            # finished reclaiming the ~10-13GB it was holding (cudaFree /
            # GGML allocator teardown for a buffer that size isn't
            # instant) -- clearing the busy flag immediately let the very
            # next request race that teardown. media-gen-mcp's own
            # Deployment requests no nvidia.com/gpu (see
            # k8s/media-gen/media-gen-mcp-deployment.yaml), so it has no
            # nvidia-smi/pynvml visibility to poll for "actually free" --
            # a short fixed settle delay before releasing the lock is the
            # pragmatic fix instead of adding GPU access to this pod just
            # for that check. Not empirically tuned to a minimum -- 3s is a
            # conservative first guess; shorten it later if it proves to
            # always be more than enough, lengthen it if the race is still
            # observed.
            await asyncio.sleep(VRAM_FREE_SETTLE_SECONDS)
            await asyncio.shield(_redis.delete(BUSY_KEY))


if __name__ == "__main__":
    # streamable_http_path defaults to "/mcp" (confirmed by inspecting the
    # installed mcp package's MCPServer.run_streamable_http_async) -- that's
    # the path k8s/p3/hermes-agent.yaml's mcp_servers.media-gen.url assumes.
    # json_response=True: plain JSON responses instead of SSE streaming --
    # simpler for a single-tool server with no need for server-initiated
    # push during a call.
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000, json_response=True)
