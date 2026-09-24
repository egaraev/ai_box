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
from mcp.server.mcpserver import Context, MCPServer

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

# For _restart_comfyui_and_wait_ready (see comment above that function) --
# calls the Kubernetes API directly (in-cluster service-account token, not
# kubectl) to delete the ComfyUI pod by label, since GGUF-loaded models are
# known not to reliably release VRAM through /free alone. Needs the RBAC
# ServiceAccount/Role/RoleBinding added to
# k8s/media-gen/media-gen-mcp-deployment.yaml -- without it this call just
# 403s and falls back to logging a warning (fail-open, same as every other
# best-effort helper in this file).
K8S_API_SERVER = "https://kubernetes.default.svc"
K8S_SA_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
K8S_SA_CA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
COMFYUI_NAMESPACE = "ai-models"
COMFYUI_POD_LABEL_SELECTOR = "app=comfyui"
# How long to wait for ComfyUI to come back up and start answering requests
# again after being restarted, before giving up and clearing the busy flag
# anyway (better to risk one request racing a still-starting pod than to
# wedge the busy flag -- and hence the router's cloud-fallback -- forever).
COMFYUI_RESTART_READY_TIMEOUT_SECONDS = 120
COMFYUI_RESTART_READY_POLL_INTERVAL_SECONDS = 2.0
# ADDED 2026-09-24, real production failure (circumstantial, not confirmed
# via a raw CUDA trace -- see the caveat below): a real generate_video call
# delivered its video fine, but the very next chat completion (Hermes's own
# follow-up reply, routed to qwen3.8-27b-Expert) crashed 3 times in a row
# right after this function's restart-and-wait-ready returned -- confirmed
# via llama-swap's own log, `[WARN] matrix: starting qwen3.8-27b-Expert
# failed: upstream command exited prematurely`, timestamps lining up exactly
# with the busy flag clearing and the next chat request landing. Hermes's
# own retry logic recovered on attempt 2, so this was invisible to Eldar,
# but a real crash happened. Same class of race this codebase has hit and
# fixed multiple times already (see VRAM_FREE_SETTLE_SECONDS above): the
# NEW ComfyUI pod answering "/" only proves ITS web server started, not
# that the OLD pod's CUDA context has actually been torn down and its VRAM
# reclaimed by the driver yet -- pod deletion is a kubelet/containerd/
# driver-level teardown that can lag behind the new pod's own readiness,
# which has nothing to do with the old pod's GPU state at all.
#
# UPDATE 2026-09-24, same day, real measured follow-up incident: the 5s
# value above turned out to be way too short. A live test showed the exact
# same crash happening even WITH the 5s settle in place --
# "ComfyUI back up and answering after restart" logged at 11:06:16.051,
# the tool-use router's next match (which only fires once the busy flag
# clears, i.e. only after this function's sleep+return) logged at
# 11:06:21.068 -- exactly +5.02s, confirming the 5s sleep really was
# running -- and qwen3.8-27b-Expert STILL failed to start, burning through
# litellm's 2 internal retries until 11:06:38.883 (+22.8s). Hermes's own
# retry (which re-hit the same router match at 11:06:41.310, +25.3s)
# finally succeeded. So the real VRAM/CUDA teardown after a full ComfyUI
# pod restart apparently takes on the order of 20-25s, not 5 -- bumped to
# 30s for margin over that measured window.
#
# CAVEAT: still not confirmed via a live-streamed upstream stderr showing
# the actual CUDA allocation failure (that only streams in real time via
# llama-swap's /logs/stream/upstream -- see aibox-config-restart-verify
# skill -- and wasn't being tailed during either incident so far). Both
# incidents match the same timing signature (crash clusters right after a
# ComfyUI pod restart, recovers once enough time has passed), which is
# strong evidence for the VRAM-teardown-race theory, but if crashes persist
# even at 30s, the next step should be capturing that live trace rather
# than keeping this number.
COMFYUI_POD_TEARDOWN_SETTLE_SECONDS = 30
# Must be the SAME path inside both this container and hermes-agent's --
# see the "DELIVERY" note above. A hostPath volume, not a PVC: this is a
# single-node cluster, so both pods always land on the same node anyway,
# and hostPath matches the pattern already used for /data/models,
# /data/comfyui elsewhere in this repo instead of introducing a new PVC.
SHARED_OUTPUT_DIR = os.environ.get("SHARED_OUTPUT_DIR", "/shared-output")

BUSY_KEY = "media-gen:busy"
BUSY_TTL_SECONDS = 300  # safety-net only -- the finally block is the primary clear path

# How long to wait after ComfyUI's /free returns 200 before actually clearing
# the busy flag -- see the comment on this constant's use in generate_image's
# finally block (real live OOM this fixed, 2026-09-09) for the full story.
# Short version: /free returning 200 doesn't mean CUDA has actually finished
# reclaiming the VRAM yet, and clearing busy immediately lets the very next
# chat request race that teardown straight into an OOM.
#
# NOTE: generate_video no longer uses this -- Wan2.2's GGUF-loaded experts
# turned out not to reliably release VRAM through /free AT ALL (not just
# slowly), so generate_video was moved to a hard ComfyUI pod restart
# instead (_restart_comfyui_and_wait_ready below). This constant/delay is
# still correct and still needed for generate_image's plain checkpoint
# loader (Z-Image Turbo, not GGUF), which hasn't shown the stuck-VRAM
# symptom -- only the settle-timing race this was originally written for.
VRAM_FREE_SETTLE_SECONDS = 3

# Poll /history for up to this long before giving up. Z-Image Turbo took
# ~18-20s end-to-end in the verified live test; 180s leaves generous room
# for a cold model load on top of that without hanging forever on a stuck job.
POLL_TIMEOUT_SECONDS = 180
POLL_INTERVAL_SECONDS = 1.0

# How often (seconds) to send an MCP progress notification during a long
# poll loop (video generation, or ComfyUI-restart readiness-waiting).
# ADDED 2026-09-09, real live bug: a real "Iron Man playing hockey" video
# generated successfully server-side (confirmed via our own logs -- prompt
# submitted, polled, fetched, saved, ComfyUI restarted, all clean) but
# Hermes reported "MCPError: Connection closed" and never got the result,
# then retried the whole thing from scratch over raw curl. Root cause
# (confirmed via github.com/NousResearch/hermes-agent PR #21323, "bump
# sse_read_timeout on SSE transport"): Hermes's MCP client has a stream
# IDLE timeout -- how long it'll wait between events on the connection,
# NOT how long the overall tool call is allowed to take -- and that's a
# completely separate mechanism from mcp_servers.<name>.timeout (which we
# already correctly set to 900 in hermes-agent.yaml; it governs a
# different, higher-level "give up and report failure" layer). That PR's
# own description says the equivalent idle-read-timeout on the Streamable
# HTTP path we actually use (not SSE) is ~300s by default -- and the real
# failure observed here dropped the connection at ~291s of total silence
# on the wire, an almost exact match. The fix isn't a bigger timeout
# value (there's no config knob for this one in config.yaml at all, near
# as this project could find) -- it's making sure the connection is never
# actually silent for that long: MCP's standard progress-notification
# mechanism (ctx.report_progress(), real protocol traffic on the same
# connection) both informs the caller AND, as a side effect, resets
# whatever "time since last event" clock the client is tracking. 20s is a
# comfortable safety margin under even a much stricter idle threshold than
# the ~300s this project could actually confirm, not just the one that
# bit us.
PROGRESS_REPORT_INTERVAL_SECONDS = 20

DEFAULT_NEGATIVE_PROMPT = "low quality, bad anatomy, extra digits, missing digits, extra limbs, missing limbs"

# LTX-2.3's own official ComfyUI template default negative prompt (kept
# verbatim -- captured directly from the real submitted /prompt payload
# for a verified-successful generation, see VIDEO_WORKFLOW_TEMPLATE's own
# comment below). Replaces Wan2.2's Chinese-language negative prompt
# (removed 2026-09-24 when LTX-2.3 replaced Wan2.2 entirely) -- this one is
# English because it's what LTX-2.3's own template ships with, not a
# translation choice.
DEFAULT_VIDEO_NEGATIVE_PROMPT = "pc game, console game, video game, cartoon, childish, ugly"

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

# Video: LTX-2.3 (Lightricks) -- replaces the earlier Wan2.2 T2V-A14B setup
# (2026-09-24, Eldar's explicit request, right after this project's
# CUDA 13.0 / torch-nightly-cu130 upgrade closed enough of a generation-
# speed gap to make LTX-2.3 practical here). LTX-2.3 was actually looked at
# once before and shelved (2026-09-08, "the LTX-2 mis-sizing mistake" --
# see the old Wan2.2 comment this replaces) over unverified community VRAM
# claims -- this pass is different: it's grounded in a real generation
# that was submitted, polled, and downloaded live against this exact box
# (2026-09-24, same day as the CUDA upgrade work), not a repeat of that
# mistake.
#
# Two real advantages over Wan2.2, both confirmed by the live test, not
# assumed: LTX-2.3 generates synced AUDIO along with the video in the same
# pass (Wan2.2 was silent-video-only), and it renders at a much higher
# native resolution (1920x1080 vs Wan2.2's 832x480) via a two-pass
# pipeline instead of Wan2.2's single KSamplerAdvanced split:
#   1. low-res draft pass -- EmptyLTXVLatentVideo at HALF the target
#      resolution -> SamplerCustomAdvanced (9-step euler_ancestral_cfg_pp,
#      cfg=1, RANDOMIZED seed -- the only seed this tool exposes, node
#      324:279 below).
#   2. LTXVLatentUpsampler does a 2x spatial upscale of that draft's
#      latent back up to the target resolution.
#   3. refinement pass -- a second SamplerCustomAdvanced (4-step
#      euler_cfg_pp, a fast/distilled sigma schedule "0.85, 0.7250,
#      0.4219, 0.0", cfg=1, FIXED seed=42, node 324:278 below) sharpens
#      the upscaled latent. This seed and schedule are the template's own
#      tuned values for its distilled LoRA (ltx-2.3-22b-distilled-
#      lora-384.safetensors, loaded at strength 0.5) -- not something this
#      tool exposes, the same way Wan2.2's cfg=3.5/shift=5 weren't.
#   4. VAEDecodeTiled (video) + LTXVAudioVAEDecode (audio) -> CreateVideo
#      muxes both into one file -> SaveVideo.
# Text conditioning goes through Gemma 3 12B (LTXAVTextEncoderLoader) --
# a real LLM, not a CLIP-style text encoder like Wan2.2's UMT5 -- so this
# may end up recognizing named characters/franchises far better than
# Wan2.2 did (see CHARACTER_EXPANSION_SYSTEM_PROMPT's own comment above).
# UNCONFIRMED either way -- generate_video() below still runs every prompt
# through that same expansion step as a precaution, since it's a
# best-effort no-op on failure, not because LTX-2.3's own name recognition
# has actually been tested.
#
# Graph below is the EXACT payload ComfyUI's own frontend submitted for a
# real, verified-successful generation on this box -- captured live via
# `fetch('/history/<prompt_id>')` against the running pod's real API
# (not graphToPrompt() this time, though the result is equivalent: the
# history endpoint hands back the literal graph that was actually
# submitted) after running ComfyUI's own official LTX-2.3 Text-to-Video
# template end to end from the UI and confirming a real, coherent video
# with audio came out. Node IDs use the "324:NNN" naming ComfyUI's own UI
# assigned when it flattened the template's subgraph (id 324, containing
# nodes 278-323) into the submitted /prompt payload -- kept verbatim
# rather than renumbered, same "don't touch what's proven to work"
# approach WORKFLOW_TEMPLATE and MUSIC_WORKFLOW_TEMPLATE already use.
# Node 325 (SaveVideo) sits outside the subgraph in the original capture.
#
# "Switch to Text to Video?" (324:304) is hardcoded True -- the template
# also supports img2vid (LTXVImgToVideoInplace, bypassed when this is
# True) but generate_video() is text-to-video only, same scope as the old
# Wan2.2 tool. ComfyUI's own graph validator still requires node 324:322
# (LoadImage) to name a real file even while fully bypassed, so it's
# pinned to a placeholder confirmed to already exist in aibox's ComfyUI
# input folder ("1c0f4c651854421d919f7d6507e72746.jpg" -- the same
# placeholder that got the standalone workflow JSON working earlier this
# same debugging session) -- it's never actually read.
#
# Confirmed output mapping via that same live /history fetch: node "325"
# (SaveVideo) reports its output under the "images" key, same convention
# SaveImage and Wan2.2's own SaveVideo both used.
#
# NOT YET VERIFIED from THIS tool specifically (same caveat every new
# model here gets until it's actually exercised end to end through Hermes):
# delivery to Telegram, and whether ComfyUI actually releases VRAM cleanly
# on its own afterward. LTX-2.3 loads via plain CheckpointLoaderSimple
# (not GGUF like Wan2.2), so it MAY behave like Z-Image Turbo -- release
# cleanly via /free -- rather than needing Wan2.2's hard pod-restart
# workaround, but that hasn't been proven with a live VRAM trace yet, so
# generate_video()'s cleanup below conservatively keeps using the
# already-proven-safe _restart_comfyui_and_wait_ready until someone
# actually runs that trace.
VIDEO_WORKFLOW_TEMPLATE = {
    "325": {
        "inputs": {
            "filename_prefix": "video/ltx2.3_t2v",
            "format": "auto",
            "format.codec": "auto",
            "codec": "auto",
            "video": ["324:312", 0],
        },
        "class_type": "SaveVideo",
        "_meta": {"title": "Save Video"},
    },
    "324:278": {"inputs": {"noise_seed": 42}, "class_type": "RandomNoise", "_meta": {"title": "RandomNoise"}},
    "324:279": {"inputs": {"noise_seed": 982558454235282}, "class_type": "RandomNoise", "_meta": {"title": "RandomNoise"}},
    "324:280": {
        "inputs": {"video_latent": ["324:290", 0], "audio_latent": ["324:309", 1]},
        "class_type": "LTXVConcatAVLatent",
        "_meta": {"title": "Concat AV Latent"},
    },
    "324:281": {
        "inputs": {"ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors"},
        "class_type": "LTXVAudioVAELoader",
        "_meta": {"title": "Load LTXV Audio VAE"},
    },
    "324:282": {"inputs": {"sampler_name": "euler_cfg_pp"}, "class_type": "KSamplerSelect", "_meta": {"title": "KSamplerSelect"}},
    "324:283": {"inputs": {"sigmas": "0.85, 0.7250, 0.4219, 0.0"}, "class_type": "ManualSigmas", "_meta": {"title": "ManualSigmas"}},
    "324:284": {
        "inputs": {"cfg": 1, "model": ["324:287", 0], "positive": ["324:286", 0], "negative": ["324:286", 1]},
        "class_type": "CFGGuider",
        "_meta": {"title": "CFG Guider"},
    },
    "324:285": {
        "inputs": {
            "noise": ["324:279", 0],
            "guider": ["324:316", 0],
            "sampler": ["324:293", 0],
            "sigmas": ["324:308", 0],
            "latent_image": ["324:321", 0],
        },
        "class_type": "SamplerCustomAdvanced",
        "_meta": {"title": "SamplerCustomAdvanced"},
    },
    "324:286": {
        "inputs": {"positive": ["324:306", 0], "negative": ["324:306", 1], "latent": ["324:309", 0]},
        "class_type": "LTXVCropGuides",
        "_meta": {"title": "LTXVCropGuides"},
    },
    "324:287": {
        "inputs": {
            "lora_name": "ltx-2.3-22b-distilled-lora-384.safetensors",
            "strength_model": 0.5,
            "model": ["324:318", 0],
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": "Load LoRA"},
    },
    "324:288": {
        "inputs": {"longer_edge": 1536, "images": ["324:292", 0]},
        "class_type": "ResizeImagesByLongerEdge",
        "_meta": {"title": "Resize Images by Longer Edge (DEPRECATED)"},
    },
    "324:289": {
        "inputs": {"samples": ["324:309", 0], "upscale_model": ["324:313", 0], "vae": ["324:318", 2]},
        "class_type": "LTXVLatentUpsampler",
        "_meta": {"title": "LTXVLatentUpsampler"},
    },
    "324:290": {
        "inputs": {
            "strength": 1,
            "bypass": ["324:304", 0],
            "vae": ["324:318", 2],
            "image": ["324:291", 0],
            "latent": ["324:289", 0],
        },
        "class_type": "LTXVImgToVideoInplace",
        "_meta": {"title": "LTXVImgToVideoInplace"},
    },
    "324:291": {
        "inputs": {"img_compression": 18, "image": ["324:288", 0]},
        "class_type": "LTXVPreprocess",
        "_meta": {"title": "LTXV Preprocess"},
    },
    "324:292": {
        "inputs": {
            "resize_type": "scale dimensions",
            "resize_type.width": ["324:314", 0],
            "resize_type.height": ["324:301", 0],
            "resize_type.crop": "center",
            "scale_method": "lanczos",
            "input": ["324:322", 0],
        },
        "class_type": "ResizeImageMaskNode",
        "_meta": {"title": "Resize Image/Mask"},
    },
    "324:293": {"inputs": {"sampler_name": "euler_ancestral_cfg_pp"}, "class_type": "KSamplerSelect", "_meta": {"title": "KSamplerSelect"}},
    "324:294": {
        "inputs": {"expression": "a/2", "values.a": ["324:314", 0]},
        "class_type": "ComfyMathExpression",
        "_meta": {"title": "Math Expression"},
    },
    "324:296": {
        "inputs": {"expression": "a/2", "values.a": ["324:301", 0]},
        "class_type": "ComfyMathExpression",
        "_meta": {"title": "Math Expression"},
    },
    "324:297": {
        "inputs": {"width": ["324:294", 1], "height": ["324:296", 1], "length": ["324:323", 1], "batch_size": 1},
        "class_type": "EmptyLTXVLatentVideo",
        "_meta": {"title": "Empty LTXV Latent Video"},
    },
    "324:298": {
        "inputs": {
            "strength": 0.7,
            "bypass": ["324:304", 0],
            "vae": ["324:318", 2],
            "image": ["324:291", 0],
            "latent": ["324:297", 0],
        },
        "class_type": "LTXVImgToVideoInplace",
        "_meta": {"title": "LTXVImgToVideoInplace"},
    },
    "324:299": {
        "inputs": {"samples": ["324:311", 1], "audio_vae": ["324:281", 0]},
        "class_type": "LTXVAudioVAEDecode",
        "_meta": {"title": "LTXV Audio VAE Decode"},
    },
    "324:300": {
        "inputs": {"expression": "a", "values.a": ["324:302", 0]},
        "class_type": "ComfyMathExpression",
        "_meta": {"title": "Math Expression (fps)"},
    },
    "324:301": {"inputs": {"value": 1080}, "class_type": "PrimitiveInt", "_meta": {"title": "Height"}},
    "324:302": {"inputs": {"value": 25}, "class_type": "PrimitiveInt", "_meta": {"title": "Frame Rate"}},
    "324:303": {"inputs": {"value": 5}, "class_type": "PrimitiveInt", "_meta": {"title": "Duration"}},
    "324:304": {"inputs": {"value": True}, "class_type": "PrimitiveBoolean", "_meta": {"title": "Switch to Text to Video?"}},
    "324:305": {
        "inputs": {"text": ["324:320", 0], "clip": ["324:319", 0]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Prompt)"},
    },
    "324:306": {
        "inputs": {"frame_rate": ["324:300", 0], "positive": ["324:305", 0], "negative": ["324:315", 0]},
        "class_type": "LTXVConditioning",
        "_meta": {"title": "LTXVConditioning"},
    },
    "324:307": {
        "inputs": {"frames_number": ["324:323", 1], "frame_rate": ["324:300", 1], "batch_size": 1, "audio_vae": ["324:281", 0]},
        "class_type": "LTXVEmptyLatentAudio",
        "_meta": {"title": "Empty LTXV Latent Audio"},
    },
    "324:308": {
        "inputs": {"sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"},
        "class_type": "ManualSigmas",
        "_meta": {"title": "ManualSigmas"},
    },
    "324:309": {"inputs": {"av_latent": ["324:285", 0]}, "class_type": "LTXVSeparateAVLatent", "_meta": {"title": "Separate AV Latent"}},
    "324:310": {
        "inputs": {
            "noise": ["324:278", 0],
            "guider": ["324:284", 0],
            "sampler": ["324:282", 0],
            "sigmas": ["324:283", 0],
            "latent_image": ["324:280", 0],
        },
        "class_type": "SamplerCustomAdvanced",
        "_meta": {"title": "SamplerCustomAdvanced"},
    },
    "324:311": {"inputs": {"av_latent": ["324:310", 0]}, "class_type": "LTXVSeparateAVLatent", "_meta": {"title": "Separate AV Latent"}},
    "324:312": {
        "inputs": {
            "fps": ["324:300", 0],
            "bit_depth": "auto",
            "color_space": "sRGB",
            "codec": "none",
            "images": ["324:317", 0],
            "audio": ["324:299", 0],
        },
        "class_type": "CreateVideo",
        "_meta": {"title": "Create Video"},
    },
    "324:313": {
        "inputs": {"model_name": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"},
        "class_type": "LatentUpscaleModelLoader",
        "_meta": {"title": "Load Latent Upscale Model"},
    },
    "324:314": {"inputs": {"value": 1920}, "class_type": "PrimitiveInt", "_meta": {"title": "Width"}},
    "324:315": {
        "inputs": {"text": "pc game, console game, video game, cartoon, childish, ugly", "clip": ["324:319", 0]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Prompt)"},
    },
    "324:316": {
        "inputs": {"cfg": 1, "model": ["324:287", 0], "positive": ["324:306", 0], "negative": ["324:306", 1]},
        "class_type": "CFGGuider",
        "_meta": {"title": "CFG Guider"},
    },
    "324:317": {
        "inputs": {
            "tile_size": 768,
            "overlap": 64,
            "temporal_size": 4096,
            "temporal_overlap": 4,
            "samples": ["324:311", 0],
            "vae": ["324:318", 2],
        },
        "class_type": "VAEDecodeTiled",
        "_meta": {"title": "VAE Decode (Tiled)"},
    },
    "324:318": {"inputs": {"ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors"}, "class_type": "CheckpointLoaderSimple", "_meta": {"title": "Load Checkpoint"}},
    "324:319": {
        "inputs": {
            "text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors",
            "ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors",
            "device": "default",
        },
        "class_type": "LTXAVTextEncoderLoader",
        "_meta": {"title": "Load LTXV Audio Text Encoder"},
    },
    "324:320": {"inputs": {"value": ""}, "class_type": "PrimitiveStringMultiline", "_meta": {"title": "Prompt"}},
    "324:321": {
        "inputs": {"video_latent": ["324:298", 0], "audio_latent": ["324:307", 0]},
        "class_type": "LTXVConcatAVLatent",
        "_meta": {"title": "Concat AV Latent"},
    },
    "324:322": {
        "inputs": {"image": "1c0f4c651854421d919f7d6507e72746.jpg"},
        "class_type": "LoadImage",
        "_meta": {"title": "Load Image"},
    },
    "324:323": {
        "inputs": {"expression": "a * b + 1", "values.a": ["324:303", 0], "values.b": ["324:302", 0]},
        "class_type": "ComfyMathExpression",
        "_meta": {"title": "Math Expression (length)"},
    },
}

# Real measured end-to-end time for this exact graph on this box (default
# 1920x1080/5s/25fps settings, live-tested 2026-09-24): ~160-180s -- well
# under budget here, but a longer/bigger request could run well past that,
# so this stays generous rather than tuned tight to the measured number.
VIDEO_POLL_TIMEOUT_SECONDS = 600
VIDEO_BUSY_TTL_SECONDS = 900  # safety-net only, same finally-block pattern as images

# Music: ACE-Step 1.5 (ace-step/ACE-Step-1.5), native ComfyUI support (no
# custom node -- confirmed via docs.comfy.org/tutorials/audio/ace-step/
# ace-step-v1-5: "now natively supported in ComfyUI", current as of a recent
# nightly build). Split-files checkpoint set (non-XL 2B DiT + 1.7B-LM text
# encoder combo -- the clean, no-CPU-offload option for a 16-20GB card per
# ACE-Step-1.5's own GPU_COMPATIBILITY.md Tier 6a; the XL 4B DiT variant
# needs offload at 16GB and is deliberately not used here), not the
# single-file all-in-one checkpoint -- chosen because the AIO template's
# TextEncodeAceStepAudio1.5 node serializes several extra trailing widget
# values (cover/remix-type controls, undocumented) that the split template
# doesn't have, so the split version is the simpler, fully-documented graph
# to hand-build an API-format payload from.
#
# CAPTURED LIVE 2026-09-10 via the browser's Network tab against a real,
# successful generation on aibox (the console's window.app.graphToPrompt()
# didn't work on this ComfyUI build -- the actual submitted /prompt request
# body was pulled from DevTools instead, same end result: the literal graph
# that already produced a real 2-minute track, not a guess). This replaced
# an earlier hand-built version (constructed from ComfyUI's shipped template
# JSON, sight-unseen) that had real mismatches this capture caught before
# deployment: the save node is SaveAudioAdvanced (format/format.quality
# fields), not SaveAudioMP3 (quality field) as first guessed; and
# TextEncodeAceStepAudio1.5 needs cfg_scale/generate_audio_codes/temperature/
# top_p/top_k/min_p, and its language/key fields are actually named
# "language"/"keyscale", not "vocal_language"/"key".
# The UI's PrimitiveFloat "Song Duration" node (fanning one widget to both
# EmptyAceStep1.5LatentAudio.seconds and TextEncodeAceStepAudio1.5.duration)
# is collapsed here into setting the same value on both fields directly --
# equivalent for a single API-submitted /prompt call, no primitive needed.
# Model files needed (download once, place under ComfyUI's basedir --
# see the ACE-Step 1.5 install steps given alongside this change):
#   models/diffusion_models/acestep_v1.5_turbo.safetensors
#   models/text_encoders/qwen_0.6b_ace15.safetensors
#   models/text_encoders/qwen_1.7b_ace15.safetensors (NOT qwen_4b_ace15.safetensors --
#     the official template defaults to the 4B variant, but ACE-Step-1.5's own
#     GPU_COMPATIBILITY.md recommends 1.7B for the 16-20GB tier this card is in)
#   models/vae/ace_1.5_vae.safetensors
# STILL UNVERIFIED: the /history output key SaveAudioAdvanced's result lands
# under. generate_image/generate_video both confirmed "images" empirically
# for their Save* nodes; SaveAudioAdvanced is a newer, audio-specific node
# and may use "audio" instead -- generate_music below checks both rather
# than gambling on one, so a wrong guess degrades to a clear "no output"
# error instead of silently looking in the wrong place.
MUSIC_WORKFLOW_TEMPLATE = {
    "104": {
        "inputs": {"unet_name": "acestep_v1.5_turbo.safetensors", "weight_dtype": "default"},
        "class_type": "UNETLoader",
        "_meta": {"title": "Load Diffusion Model"},
    },
    "105": {
        "inputs": {
            "clip_name1": "qwen_0.6b_ace15.safetensors",
            "clip_name2": "qwen_1.7b_ace15.safetensors",
            "type": "ace",
            "device": "default",
        },
        "class_type": "DualCLIPLoader",
        "_meta": {"title": "Load CLIP (Dual)"},
    },
    "106": {
        "inputs": {"vae_name": "ace_1.5_vae.safetensors"},
        "class_type": "VAELoader",
        "_meta": {"title": "Load VAE"},
    },
    "78": {
        "inputs": {"shift": 3, "model": ["104", 0]},
        "class_type": "ModelSamplingAuraFlow",
        "_meta": {"title": "ModelSamplingAuraFlow"},
    },
    "94": {
        "inputs": {
            "clip": ["105", 0],
            "tags": "",
            "lyrics": "",
            "seed": 42,
            "duration": 120,
            "timesignature": "4",
            "language": "en",
            "keyscale": "C major",
            "generate_audio_codes": True,
            "cfg_scale": 2,
            "bpm": 120,
            "temperature": 0.85,
            "top_p": 0.9,
            "top_k": 0,
            "min_p": 0,
        },
        "class_type": "TextEncodeAceStepAudio1.5",
        "_meta": {"title": "TextEncodeAceStepAudio1.5"},
    },
    "47": {
        "inputs": {"conditioning": ["94", 0]},
        "class_type": "ConditioningZeroOut",
        "_meta": {"title": "Conditioning Zero Out"},
    },
    "98": {
        "inputs": {"seconds": 120, "batch_size": 1},
        "class_type": "EmptyAceStep1.5LatentAudio",
        "_meta": {"title": "Empty Ace Step 1.5 Latent Audio"},
    },
    "3": {
        "inputs": {
            "seed": 42,
            "steps": 8,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1,
            "model": ["78", 0],
            "positive": ["94", 0],
            "negative": ["47", 0],
            "latent_image": ["98", 0],
        },
        "class_type": "KSampler",
        "_meta": {"title": "KSampler"},
    },
    "18": {
        "inputs": {"samples": ["3", 0], "vae": ["106", 0]},
        "class_type": "VAEDecodeAudio",
        "_meta": {"title": "VAE Decode Audio"},
    },
    "111": {
        "inputs": {
            "filename_prefix": "audio/media-gen-mcp",
            "format": "mp3",
            "format.quality": "128k",
            "audio": ["18", 0],
        },
        "class_type": "SaveAudioAdvanced",
        "_meta": {"title": "Save Audio (Advanced)"},
    },
}

# ACE-Step 1.5 turbo is an 8-step distilled config (same class of fast/cheap
# as Z-Image Turbo's image generation, not a slow multi-minute render like
# Wan2.2 video) -- generous timeout anyway to cover a cold checkpoint load
# the first time it's used, same reasoning as the other two poll timeouts.
MUSIC_POLL_TIMEOUT_SECONDS = 300
# 400 -> 600 2026-09-10, alongside switching the finally block to
# _restart_comfyui_and_wait_ready: that adds up to COMFYUI_RESTART_READY_
# TIMEOUT_SECONDS (120s) of pod-restart wait on top of generation itself, so
# 400 no longer left enough margin (300 poll + 120 restart leaves only 20s
# spare). Safety-net only, same finally-block pattern as video.
MUSIC_BUSY_TTL_SECONDS = 600

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


async def _restart_comfyui_and_wait_ready(client: httpx.AsyncClient, ctx: Context) -> None:
    """Used by generate_video's finally block instead of _free_comfyui_vram.

    Real live incident, 2026-09-09: after a successful Wan2.2 video
    generation, ComfyUI's /free returned 200 but the python3 process stayed
    pinned at 10756MiB for 9+ minutes afterward (confirmed via a live
    nvidia-smi trace showing the exact same figure, unchanged, tick after
    tick -- not a slow drain, a genuinely stuck allocation), which then
    OOM'd the very next local chat-model load. This is not specific to our
    code or the earlier settle-delay race (VRAM_FREE_SETTLE_SECONDS) --
    it's a known, currently-unresolved limitation of GGUF-loaded models in
    ComfyUI specifically, corroborated by third parties hitting the same
    thing: github.com/1038lab/ComfyUI-QwenVL issue #104 ("not possible to
    free the VRAM ... only a hard restart of Comfyui helps"), and
    github.com/SeanScripts/ComfyUI-Unload-Model's own README: "there have
    been some recent changes in the GGUF loader nodes that could cause the
    unload command to not actually unload the GGUF models." Since Wan2.2's
    two experts are both loaded via city96/ComfyUI-GGUF's UnetLoaderGGUF
    node, generate_video hits exactly this case -- generate_image doesn't
    (Z-Image Turbo uses a plain UNETLoader, not GGUF), so it keeps using
    _free_comfyui_vram + VRAM_FREE_SETTLE_SECONDS unchanged.

    /free is still attempted first below (cheap, sometimes actually works --
    it did in an earlier test the same day), but this function unconditionally
    restarts the ComfyUI pod afterward regardless of what /free reports,
    since that's the only mechanism anyone (including us, live) has
    confirmed actually reclaims GGUF VRAM.

    Needs RBAC (k8s/media-gen/media-gen-mcp-deployment.yaml): a dedicated
    ServiceAccount for this Deployment, bound via a RoleBinding to a Role in
    the ai-models namespace granting get/list/delete/deletecollection on
    pods. Delete is by label selector (K8s "DeleteCollection", hence the
    deletecollection verb) rather than by name, since the pod's name changes
    on every restart.

    Caller (generate_video's finally block) clears the busy flag only after
    this returns, not just after issuing the delete -- so the busy window,
    and the router's cloud-fallback for chat requests, correctly covers the
    whole restart + cold-start window, not just the moment the delete call
    was made.
    """
    try:
        await client.post(
            f"{COMFYUI_URL}/free",
            json={"unload_models": True, "free_memory": True},
            timeout=30.0,
        )
    except Exception as exc:  # noqa: BLE001 -- best effort, the restart below is the real fix
        logger.warning("ComfyUI /free failed (%s: %s) -- restarting pod anyway", type(exc).__name__, exc)

    try:
        with open(K8S_SA_TOKEN_PATH) as f:
            token = f.read().strip()
        async with httpx.AsyncClient(verify=K8S_SA_CA_PATH, timeout=30.0) as k8s_client:
            resp = await k8s_client.delete(
                f"{K8S_API_SERVER}/api/v1/namespaces/{COMFYUI_NAMESPACE}/pods",
                params={"labelSelector": COMFYUI_POD_LABEL_SELECTOR},
                headers={"Authorization": f"Bearer {token}"},
            )
            logger.info("k8s delete comfyui pod(s) -> %s", resp.status_code)
    except Exception as exc:  # noqa: BLE001 -- best effort; if this fails VRAM stays stuck, same failure mode as before this fix existed
        logger.warning("comfyui pod restart failed (%s: %s) -- VRAM may stay stuck until a manual restart", type(exc).__name__, exc)
        return

    # Wait for the new pod to actually answer before letting anything else
    # (another generate call, or a chat request once busy clears) hit
    # ComfyUI again. Polls "/" (the frontend index page) rather than a
    # specific API endpoint like /system_stats, since "/" is the one thing
    # already confirmed to exist on this ComfyUI version without having to
    # guess -- and custom nodes (including ComfyUI-GGUF) import at startup
    # before ComfyUI starts serving HTTP at all, so a 200 here also means
    # they're already loaded.
    start = time.monotonic()
    last_progress = start
    deadline = start + COMFYUI_RESTART_READY_TIMEOUT_SECONDS
    comfyui_answered = False
    while time.monotonic() < deadline:
        await asyncio.sleep(COMFYUI_RESTART_READY_POLL_INTERVAL_SECONDS)
        try:
            resp = await client.get(f"{COMFYUI_URL}/", timeout=5.0)
            if resp.status_code == 200:
                logger.info("ComfyUI back up and answering after restart")
                comfyui_answered = True
                break
        except Exception:  # noqa: BLE001 -- expected while the pod is still (re)starting
            pass
        now = time.monotonic()
        if now - last_progress >= PROGRESS_REPORT_INTERVAL_SECONDS:
            # See PROGRESS_REPORT_INTERVAL_SECONDS's own comment -- this
            # keeps Hermes's connection from going idle long enough to get
            # dropped while we wait for ComfyUI to come back up.
            try:
                await ctx.report_progress(
                    progress=now - start,
                    total=COMFYUI_RESTART_READY_TIMEOUT_SECONDS,
                    message="Restarting ComfyUI to fully clear VRAM...",
                )
            except Exception as exc:  # noqa: BLE001 -- best effort, must not break the restart wait
                logger.warning("report_progress failed (%s: %s)", type(exc).__name__, exc)
            last_progress = now
    if not comfyui_answered:
        logger.warning(
            "ComfyUI did not answer within %ss of restart -- clearing busy flag anyway",
            COMFYUI_RESTART_READY_TIMEOUT_SECONDS,
        )

    # ADDED 2026-09-24, same incident as COMFYUI_POD_TEARDOWN_SETTLE_SECONDS's
    # own comment above -- "/" answering 200 only proves the NEW pod's web
    # server is up, not that the OLD pod's CUDA context has actually been
    # torn down and its VRAM reclaimed by the driver yet. Sleep here
    # unconditionally (whether ComfyUI answered or we gave up waiting) so
    # whatever chat-model load happens right after the busy flag clears gets
    # a bit more real-world margin, mirroring how VRAM_FREE_SETTLE_SECONDS is
    # applied unconditionally elsewhere regardless of what /free reported.
    await asyncio.sleep(COMFYUI_POD_TEARDOWN_SETTLE_SECONDS)


@mcp.tool()
async def generate_image(
    prompt: str,
    ctx: Context,
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
                # See PROGRESS_REPORT_INTERVAL_SECONDS's own comment
                # (generate_video hit this for real) -- images are normally
                # well under the connection-idle risk window, but a cold
                # model load can run long, so this costs nothing to have here.
                now = time.monotonic()
                if now - last_progress >= PROGRESS_REPORT_INTERVAL_SECONDS:
                    try:
                        await ctx.report_progress(
                            progress=now - poll_start, total=POLL_TIMEOUT_SECONDS,
                            message="Z-Image Turbo still rendering...",
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
            # see the comment on this constant's definition above for the
            # full story -- a real OOM this fixed, caused by /free
            # returning 200 before CUDA had actually finished reclaiming
            # the VRAM. generate_video no longer has this same shape (moved
            # to _restart_comfyui_and_wait_ready, see its docstring) since
            # GGUF-loaded models turned out not to release VRAM through
            # /free at all, not just slowly -- this constant/delay is still
            # correct here though, since Z-Image Turbo's plain checkpoint
            # loader only ever showed the slow-settle version of the
            # problem, not the stuck-forever one.
            await asyncio.sleep(VRAM_FREE_SETTLE_SECONDS)
            await asyncio.shield(_redis.delete(BUSY_KEY))


@mcp.tool()
async def generate_video(
    prompt: str,
    ctx: Context,
    negative_prompt: str = "",
    width: int = 1920,
    height: int = 1080,
    duration_seconds: int = 5,
    fps: int = 25,
    seed: int = -1,
) -> str:
    """Generate a short video WITH audio locally with LTX-2.3 (via ComfyUI on aibox's GPU).

    Replaces the earlier Wan2.2 T2V-A14B pipeline (2026-09-24). Two real
    upgrades over Wan2.2, both confirmed by a real live test on this box,
    not assumed: LTX-2.3 generates synced AUDIO along with the video in
    the same call (Wan2.2 was silent-video-only), and it renders at a much
    higher default resolution (1920x1080 vs Wan2.2's 832x480). Internally
    this runs a two-pass pipeline -- a fast low-res draft, then a 2x
    latent upscale plus a short distilled refinement pass -- not one flat
    KSampler the way Wan2.2 was; see the comment above
    VIDEO_WORKFLOW_TEMPLATE for the full breakdown. Real measured
    end-to-end time on this box for this exact graph at the defaults
    below: ~160-180s. Like generate_image, this fully unloads whatever
    chat model is resident to give the whole GPU to ComfyUI, then reloads
    the chat model lazily afterward -- any chat message sent elsewhere
    during that window is automatically redirected to a cloud model.

    Named characters (e.g. "Tony Stark", "a Spacemarine from Starcraft")
    are handled automatically -- before submitting to ComfyUI, this call
    runs `prompt` through the local expert chat model, which rewrites any
    named character/franchise/celebrity/real-person reference into a plain
    visual description (build, outfit/armor, colors, distinguishing
    features), the same expansion step the old Wan2.2 version of this tool
    used (see the comment above CHARACTER_EXPANSION_SYSTEM_PROMPT). LTX-2.3
    uses Gemma 3 12B as its text encoder -- a real LLM, not a CLIP-style
    encoder like Wan2.2's UMT5 -- so it may turn out to need this expansion
    step less than Wan2.2 did, but that hasn't actually been tested yet,
    so it's kept on as a precaution. This adds real latency (the expert
    model itself is slow, up to ~30-50s) on top of the video generation
    time; if that expansion call fails for any reason it silently falls
    back to the original prompt rather than blocking the generation, so a
    failure here degrades quality, not reliability.

    Args:
        prompt: What the video should show. Describe subject, motion,
            camera movement, and style -- motion description matters more
            here than for a still image. Named characters are fine to
            reference directly (see above) -- they get auto-translated to
            a visual description before generation.
        negative_prompt: What to avoid. Empty by default falls back to
            DEFAULT_VIDEO_NEGATIVE_PROMPT (LTX-2.3's own template default).
        width: Output width in pixels. Default 1920. The low-res draft
            pass renders at half this and gets 2x upscaled back up, so
            keep this an even number.
        height: Output height in pixels. Default 1080 -- same even-number
            note as width.
        duration_seconds: Video length in seconds. Default 5. Longer
            videos take proportionally longer to generate.
        fps: Output frame rate. Default 25 (this template's own default).
        seed: Fixed seed for reproducibility. Default -1 picks a random
            seed each call. Only affects the low-res draft pass -- the
            refinement pass uses a fixed seed by design (part of the
            template's own tuned distilled-LoRA schedule, not something
            this tool exposes, the same way Wan2.2's cfg/shift weren't).

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
            # Must happen BEFORE _unload_chat_model -- this call needs a
            # chat model loaded (it'll trigger llama-swap to lazily load
            # the expert model if it isn't resident already). Doing it
            # after the unload would just force an extra, wasted reload.
            expanded_prompt = await _expand_character_references(client, prompt)

            await _unload_chat_model(client)

            workflow = copy.deepcopy(VIDEO_WORKFLOW_TEMPLATE)
            workflow["324:320"]["inputs"]["value"] = expanded_prompt
            workflow["324:315"]["inputs"]["text"] = negative_prompt or DEFAULT_VIDEO_NEGATIVE_PROMPT
            workflow["324:314"]["inputs"]["value"] = width
            workflow["324:301"]["inputs"]["value"] = height
            workflow["324:303"]["inputs"]["value"] = duration_seconds
            workflow["324:302"]["inputs"]["value"] = fps
            workflow["324:279"]["inputs"]["noise_seed"] = resolved_seed

            submit_resp = await client.post(
                f"{COMFYUI_URL}/prompt", json={"prompt": workflow}, timeout=30.0
            )
            submit_resp.raise_for_status()
            prompt_id = submit_resp.json()["prompt_id"]
            logger.info(
                "submitted video prompt_id=%s seed=%s %sx%s %ss@%sfps",
                prompt_id, resolved_seed, width, height, duration_seconds, fps,
            )

            poll_start = time.monotonic()
            last_progress = poll_start
            deadline = poll_start + VIDEO_POLL_TIMEOUT_SECONDS
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
                    # See PROGRESS_REPORT_INTERVAL_SECONDS's own comment --
                    # a real generation ran this loop silently for ~5
                    # minutes and Hermes's connection got dropped partway
                    # through (confirmed live, 2026-09-09). This keeps the
                    # connection from ever going that long without traffic.
                    try:
                        await ctx.report_progress(
                            progress=now - poll_start,
                            total=VIDEO_POLL_TIMEOUT_SECONDS,
                            message="LTX-2.3 still rendering...",
                        )
                    except Exception as exc:  # noqa: BLE001 -- best effort, must not break generation
                        logger.warning("report_progress failed (%s: %s)", type(exc).__name__, exc)
                    last_progress = now
                await asyncio.sleep(POLL_INTERVAL_SECONDS)

            if history is None:
                raise TimeoutError(
                    f"ComfyUI did not finish video prompt {prompt_id} within {VIDEO_POLL_TIMEOUT_SECONDS}s"
                )

            status = history.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI reported an error for prompt {prompt_id}: {status}")

            # SaveVideo's output uses the same "images" key ComfyUI uses for
            # SaveImage and Wan2.2's own SaveVideo (confirmed empirically
            # against a real completed job, not documented anywhere
            # obvious) -- node "325" here, not "14" (Wan2.2's node id).
            videos = history.get("outputs", {}).get("325", {}).get("images", [])
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
            #
            # STILL using the hard-restart cleanup (_restart_comfyui_and_
            # wait_ready) inherited from Wan2.2, not the gentler
            # _free_comfyui_vram + settle-delay pattern generate_image
            # uses -- LTX-2.3 loads via plain CheckpointLoaderSimple (not
            # GGUF like Wan2.2), so it MAY release VRAM cleanly through
            # /free the way Z-Image Turbo does, but that hasn't been
            # confirmed with a live VRAM trace yet. Given how expensive a
            # wrong guess is here (a crash-looping main chat model -- see
            # generate_music's own finally-block comment for the one time
            # this project guessed wrong and paid for it), this keeps the
            # already-proven-safe hard restart until someone actually runs
            # that trace and confirms /free alone is enough.
            await asyncio.shield(_restart_comfyui_and_wait_ready(client, ctx))
            await asyncio.shield(_redis.delete(BUSY_KEY))


@mcp.tool()
async def generate_music(
    ctx: Context,
    lyrics: str = "",
    tags: str = "pop, upbeat, energetic",
    duration: float = 120.0,
    bpm: int = 120,
    vocal_language: str = "en",
    key: str = "C major",
    seed: int = -1,
) -> str:
    """Generate a full song with vocals locally with ACE-Step 1.5 (via ComfyUI on aibox's GPU).

    Like generate_image/generate_video, this fully unloads whatever chat
    model is currently resident to give the whole GPU to generation, then
    reloads the chat model lazily afterward. Any chat message sent
    elsewhere during that window is automatically redirected to a cloud
    model. ACE-Step 1.5 turbo is an 8-step distilled model (same class of
    fast as Z-Image Turbo's image generation) -- confirmed live 2026-09-10:
    a real 2-minute track rendered successfully via this exact graph in the
    ComfyUI UI (see MUSIC_WORKFLOW_TEMPLATE's own comment for how it was
    captured). Untested from THIS tool specifically, though -- the graph is
    confirmed, the MCP wiring around it isn't yet.

    Args:
        lyrics: The song's lyrics, with section markers -- e.g.
            "[Verse]\\nFirst line...\\n[Chorus]\\nHook line..." ACE-Step 1.5's
            own convention is [Verse]/[Chorus]/[Bridge]/[Outro] markers, one
            section per line group. An instrumental-only track can pass an
            empty string or "[Instrumental]".
        tags: Comma-separated genre/mood/instrumentation tags -- e.g.
            "synthwave, energetic, female vocals, 80s". Free text, not a
            fixed vocabulary.
        duration: Song length in seconds. Default 120 (2 min), matching the
            official template's own default. Longer songs take
            proportionally longer to render.
        bpm: Target tempo. Default 120.
        vocal_language: Language code for the vocals (e.g. "en", "ja", "ko",
            "zh", "es"). Default "en".
        key: Musical key, e.g. "C major", "E minor". Default "C major".
        seed: Fixed seed for reproducibility. Default -1 picks a random seed
            each call.

    Returns:
        A short confirmation message containing a `MEDIA:<path>` line for
        the generated song -- include that line VERBATIM, on its own, in
        your reply to the user, the same as generate_image/generate_video's
        output. NOTE: unlike image delivery, this has not yet been confirmed
        to actually reach Telegram as a playable audio file -- if it
        doesn't, that's the first thing to check (see the top-of-file
        comment on generate_image's own delivery history for the pattern
        this class of bug takes).
    """
    if not tags or not tags.strip():
        raise ValueError("tags must not be empty -- ACE-Step 1.5 needs at least genre/mood tags to generate from")

    already_busy = await _redis.set(BUSY_KEY, "1", ex=MUSIC_BUSY_TTL_SECONDS, nx=True)
    if not already_busy:
        raise RuntimeError(
            "Media generation is already in progress. ComfyUI only holds "
            "the GPU for one job at a time -- try again in a moment."
        )

    resolved_seed = random.randint(0, 2**31 - 1) if seed < 0 else seed

    async with httpx.AsyncClient() as client:
        try:
            await _unload_chat_model(client)

            workflow = copy.deepcopy(MUSIC_WORKFLOW_TEMPLATE)
            workflow["94"]["inputs"]["tags"] = tags
            workflow["94"]["inputs"]["lyrics"] = lyrics or "[Instrumental]"
            workflow["94"]["inputs"]["seed"] = resolved_seed
            workflow["94"]["inputs"]["bpm"] = bpm
            workflow["94"]["inputs"]["duration"] = duration
            workflow["94"]["inputs"]["language"] = vocal_language
            workflow["94"]["inputs"]["keyscale"] = key
            workflow["98"]["inputs"]["seconds"] = duration
            workflow["3"]["inputs"]["seed"] = resolved_seed

            submit_resp = await client.post(
                f"{COMFYUI_URL}/prompt", json={"prompt": workflow}, timeout=30.0
            )
            submit_resp.raise_for_status()
            prompt_id = submit_resp.json()["prompt_id"]
            logger.info("submitted music prompt_id=%s seed=%s duration=%s", prompt_id, resolved_seed, duration)

            poll_start = time.monotonic()
            last_progress = poll_start
            deadline = poll_start + MUSIC_POLL_TIMEOUT_SECONDS
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
                            progress=now - poll_start, total=MUSIC_POLL_TIMEOUT_SECONDS,
                            message="ACE-Step 1.5 still rendering...",
                        )
                    except Exception as exc:  # noqa: BLE001 -- best effort, must not break generation
                        logger.warning("report_progress failed (%s: %s)", type(exc).__name__, exc)
                    last_progress = now
                await asyncio.sleep(POLL_INTERVAL_SECONDS)

            if history is None:
                raise TimeoutError(
                    f"ComfyUI did not finish music prompt {prompt_id} within {MUSIC_POLL_TIMEOUT_SECONDS}s"
                )

            status = history.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI reported an error for prompt {prompt_id}: {status}")

            # STILL UNCONFIRMED (see MUSIC_WORKFLOW_TEMPLATE's own comment):
            # SaveAudioAdvanced is a newer, audio-specific Save* node, unlike
            # SaveImage/SaveVideo which both confirmed "images" empirically
            # -- checking both keys rather than gambling on one, so a wrong
            # guess here degrades to a clear "produced no audio output"
            # error (fixable by checking a real /history response) instead
            # of silently looking in the wrong place. Node "111" here.
            node_output = history.get("outputs", {}).get("111", {})
            audios = node_output.get("audio") or node_output.get("images") or []
            if not audios:
                raise RuntimeError(f"ComfyUI finished prompt {prompt_id} but produced no audio output")
            audio_ref = audios[0]

            view_resp = await client.get(
                f"{COMFYUI_URL}/view",
                params={
                    "filename": audio_ref["filename"],
                    "subfolder": audio_ref.get("subfolder", ""),
                    "type": audio_ref.get("type", "output"),
                },
                timeout=60.0,
            )
            view_resp.raise_for_status()
            audio_bytes = view_resp.content
            logger.info("fetched output audio %s (%d bytes)", audio_ref["filename"], len(audio_bytes))

            os.makedirs(SHARED_OUTPUT_DIR, exist_ok=True)
            out_path = os.path.join(SHARED_OUTPUT_DIR, f"{uuid.uuid4().hex}.mp3")
            with open(out_path, "wb") as f:
                f.write(audio_bytes)
            logger.info("saved audio to shared path %s", out_path)

            return (
                f"Song generated successfully and saved to {out_path}\n\n"
                f"To actually attach it, include this exact line on its own "
                f"in your reply (don't just describe the song):\n"
                f"MEDIA:{out_path}"
            )
        finally:
            # CHANGED 2026-09-10, before generate_music's own cleanup path
            # was ever actually exercised: originally used generate_image's
            # gentler pattern (_free_comfyui_vram + a settle delay) on the
            # assumption that ACE-Step 1.5's plain-safetensors checkpoint
            # (not GGUF-loaded like Wan2.2) would release cleanly the same
            # way Z-Image Turbo does. A real incident the same day showed
            # exactly the failure this assumption risked -- a manual ComfyUI
            # UI test (bypassing this tool, and so this cleanup code,
            # entirely) left ACE-Step's checkpoint resident long enough that
            # qwen3.8-27b-Expert's next load hit a genuine CUDA OOM
            # (cudaMalloc failed on a 12.7GB buffer -- confirmed via the real
            # upstream stderr, not just llama-swap's generic wrapper error;
            # see /areas/ai-box-k3s.md for the full trace). That incident
            # doesn't itself prove generate_music's own cleanup is
            # insufficient (the UI test never ran it at all), but given how
            # expensive a wrong guess is here -- a crash-looping main chat
            # model -- this switches to generate_video's harder, already-
            # proven-necessary pattern (full ComfyUI pod restart, not just
            # /free) as a precaution rather than waiting for a second real
            # incident to confirm it the hard way.
            await asyncio.shield(_restart_comfyui_and_wait_ready(client, ctx))
            await asyncio.shield(_redis.delete(BUSY_KEY))


if __name__ == "__main__":
    # streamable_http_path defaults to "/mcp" (confirmed by inspecting the
    # installed mcp package's MCPServer.run_streamable_http_async) -- that's
    # the path k8s/p3/hermes-agent.yaml's mcp_servers.media-gen.url assumes.
    #
    # json_response=True REMOVED 2026-09-10, real production bug: it was
    # set on the (wrong) assumption that this server has "no need for
    # server-initiated push during a call" -- but PROGRESS_REPORT_INTERVAL_
    # SECONDS's own ctx.report_progress() heartbeats (added 2026-09-09,
    # see that constant's comment) ARE exactly that kind of push, meant to
    # keep Hermes's MCP connection from going idle during a multi-minute
    # generate_video/generate_image call. Confirmed via reading the
    # installed mcp package's source live on the pod (v2.2.0,
    # mcp/server/streamable_http.py's message_router): in JSON-response
    # mode, ANY message tied to the current request's related_request_id
    # -- which includes every progress notification -- is unconditionally
    # dropped before being queued, logged only at DEBUG ("dropped message
    # related to request ... in JSON mode"), never raised as an error. So
    # every report_progress() call since 2026-09-09 has silently no-op'd:
    # no exception, no warning, but nothing ever actually reached Hermes on
    # the wire either -- the connection was genuinely silent for the whole
    # render duration regardless, which is exactly what caused today's
    # (2026-09-10) repeat of the original "Iron Man playing hockey"-class
    # incident. Removing json_response=True falls back to the transport's
    # default SSE-based streaming response, which has a real wire form for
    # interim messages -- this is what actually lets the existing
    # heartbeat code do what it was written to do. See /areas/ai-box-k3s.md
    # for the full investigation if this needs revisiting.
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
