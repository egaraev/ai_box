# telegram_sync.py -- Telegram channels -> Obsidian vault sync (Telethon user API).
#
# Ported from the old Windows-lab pipeline (ai_ml_cv_projects/ai-lab/step10_n8n/telegram_sync.py),
# same treatment as gmail_sync.py got: paths repointed to aibox, LiteLLM calls
# repointed to aibox's own gateway and switched from "auto" to a direct fast
# model for the same reason (see gmail_sync.py's header comment).
#
# Reads posts from the configured Telegram channels, runs a relevance filter
# (keyword shortcuts + LLM) so only real AI/tech content gets saved, tags via
# LiteLLM against the same KNOWN_TAGS vocabulary gmail_sync.py uses, and
# saves into /data/obsidian_vault/AI. Tracks the last-seen message ID per
# channel so re-running only processes new posts.
#
# Image posts now get OCR'd too (2026-09-14, ported from gmail_sync.py's
# same-day addition): a photo or image document on a channel post is saved
# into vault_root/Attachments (the same folder gmail_sync.py's own
# attachments and vault-mcp's ingest_file use) and transcribed via the same
# locked vision route/prompt -- verbatim text, not a scene caption. The
# transcription becomes the note body when the post has no caption (a bare
# image post), or gets appended below the caption otherwise. An image-only
# post skips the pre-OCR relevance filter (there's no caption to judge yet)
# and is saved unconditionally -- same tradeoff gmail_sync.py accepts by not
# filtering image attachments at all.
#
# Setup (one-time, in the scripts/.venv already used by vault_sync.py):
#   pip install telethon
#
#   Get an api_id/api_hash from https://my.telegram.org/apps, then:
#     export TELEGRAM_API_ID=12345678
#     export TELEGRAM_API_HASH=abc123def456...
#   First run only (interactive login -- saves a reusable session file):
#     python3 scripts/telegram_sync.py --auth
#   (If a .telegram_session file already exists from the Windows lab, copy
#   it into this scripts/ folder instead and skip --auth entirely. Running
#   inside the N8N pod: --auth needs a real terminal, so do this one-time
#   step via `kubectl exec -it ... -- python3 /opt/scripts/telegram_sync.py
#   --auth` once TELEGRAM_SYNC_STATE_DIR/TELEGRAM_API_ID/TELEGRAM_API_HASH
#   are set in n8n-deployment.yaml -- the resulting session file lands on
#   the persistent n8n-data volume, see STATE_DIR below.)
#
# Usage:
#   export LITELLM_MASTER_KEY=$(kubectl get secret -n ai-models litellm-secret -o jsonpath='{.data.master_key}' | base64 -d)
#   python3 scripts/telegram_sync.py --dry-run     # preview only
#   python3 scripts/telegram_sync.py               # live sync
#   python3 scripts/telegram_sync.py --no-filter   # skip the LLM relevance filter
#   python3 scripts/telegram_sync.py --debug       # show filter decisions

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import timezone
from pathlib import Path

# ── config ───────────────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent

# Where the state file + session file live. Defaults to right next to this
# script (HERE), same as always, for the host scripts/.venv case -- but
# inside the N8N pod this script's own directory (/opt/scripts) is baked
# into the image, not on any mounted volume, so anything written there is
# gone the next time the pod restarts (n8n-deployment.yaml uses Recreate,
# and an image rebuild wipes it too). TELEGRAM_SYNC_STATE_DIR overrides this
# to a subfolder of the n8n-data hostPath mount instead, which IS persistent
# -- set only in n8n-deployment.yaml's env block, unset (falls back to HERE)
# everywhere else, so nothing changes for the host-venv case.
STATE_DIR = Path(os.environ.get("TELEGRAM_SYNC_STATE_DIR", str(HERE)))
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE   = STATE_DIR / ".telegram_sync_state.json"
SESSION_FILE = str(STATE_DIR / ".telegram_session")

DEFAULT_VAULT_AI = Path("/data/obsidian_vault/AI")

# Same 2 channels as the old lab -- edit this list to add more later
CHANNELS = [
    "prompt_design",
    "ai_rostov",
]

API_ID   = int(os.getenv("TELEGRAM_API_ID", "0"))
API_HASH = os.getenv("TELEGRAM_API_HASH", "")

FETCH_LIMIT = 50

# Same NodePort/env-var convention as scripts/vault_sync.py and gmail_sync.py
# -- overridable so this also works run in-cluster (N8N Execute Command),
# where localhost doesn't reach the node's NodePorts.
LITELLM_URL = os.environ.get("LITELLM_URL", "http://localhost:30400")
LITELLM_MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY")
TAG_MODEL = "hermes3-llama-3.1-8b"
# Same locked vision route vault-mcp's ingest_file / gmail_sync.py's own
# image OCR already use (custom_router_hook.py's _has_image check) --
# verbatim transcription of a photographed note/document/screenshot, not a
# scene caption.
VISION_MODEL = "qwen3.8-27b-Expert-Vision"

# Same gid/convention as gmail_sync.py's chgrp_writable -- kept as its own
# small copy here rather than a shared import, same reasoning as every
# other script in this repo (see gmail_sync.py's own comment on why). The
# N8N pod writes into the vault as root, so without this any note or
# attachment this script saves would land root:root and Eldar/Hermes
# (gid 10000) couldn't edit or delete it later.
HERMES_GID = 10000

def chgrp_writable(path, dir_mode=0o775, file_mode=0o664):
    mode = dir_mode if path.is_dir() else file_mode
    os.chown(path, -1, HERMES_GID)   # -1 leaves the owner (root) unchanged, only the group changes
    path.chmod(mode)

# ── known tags vocabulary (same list gmail_sync.py uses -- both feed the same AI/ folder) ──

KNOWN_TAGS = [
    "agents", "prompting", "rag", "fine-tuning", "local-llm", "model-release",
    "mcp", "tools", "vibe-coding", "security", "embeddings", "benchmarks",
    "automation", "claude", "openai", "coding-assistant", "multimodal",
    "context-window", "vision", "voice", "open-source",
]

# ── relevance filter ─────────────────────────────────────────────────────────

# Keyword hits here reject a post outright, before any LLM call
SKIP_KEYWORDS = [
    # Politics / war
    "война", "военн", "армия", "украин", "кремл", "нато", "санкц",
    "политическ", "голосов", "депутат", "президент",
    # Sports
    "футбол", "хоккей", "баскетбол", "спорт",
    # Financial noise
    "акции", "триллион", "капитализаци",
    # Audio-only posts
    "аудиоверси",
    # Non-AI tools / marketing noise
    "figma", "imaz", "ios-приложен", "app store",
    "жюри", "личный кабинет", "лендинг", "конверси",
    "это канал", "стримчанск", "инстушк", "конкурс",
]

# Keyword hits here accept a post outright, skipping the LLM call
SAVE_PRIORITY = [
    "claude", "gpt", "gemini", "llama", "mistral", "qwen", "hermes",
    "deepseek", "glm", "openclaw",
    "anthropic", "openai", "ollama",
    "codex", "cursor", "windsurf", "langchain", "langgraph", "crewai", "autogen",
    "mcp", "rag", "llm", "cloudflare", "n8n",
    "клод", "клауд", "гпт", "опенаи", "антропик", "гемини",
]


# Drops a leading "audio version" header line some posts have; returns ""
# if the post has nothing left after stripping it
def _strip_audio_header(text):
    lines = text.strip().splitlines()
    if lines and ("аудиоверси" in lines[0].lower() or lines[0].strip().startswith("🎧")):
        rest = "\n".join(lines[1:]).strip()
        return rest if len(rest) > 20 else ""
    return text


# Decides whether a post is worth saving: audio-only/bare-URL -> reject,
# SKIP_KEYWORDS hit -> reject, SAVE_PRIORITY hit -> accept, otherwise ask
# the LLM. Defaults to SAVE if the LLM call fails (fail open).
def is_relevant(text, debug=False):
    content = _strip_audio_header(text.strip())
    if not content:
        if debug:
            print("  [filter] SKIP -- audio-only post")
        return False

    lower = content.lower()

    if re.fullmatch(r"https?://\S+", content):
        if debug:
            print("  [filter] SKIP -- bare URL")
        return False

    for kw in SKIP_KEYWORDS:
        if kw in lower:
            if debug:
                print(f"  [filter] SKIP -- keyword {kw!r}")
            return False

    for kw in SAVE_PRIORITY:
        if kw in lower:
            if debug:
                print(f"  [filter] SAVE -- priority {kw!r}")
            return True

    prompt = (
        "You are a strict filter for a PERSONAL TECHNICAL AI knowledge base.\n"
        "The post may be in Russian, Ukrainian, or English.\n\n"
        "Reply SAVE only when the post contains CONCRETE TECHNICAL AI content:\n"
        "  • new AI model or tool release with technical details\n"
        "  • how to use an AI tool, model, or technique (practical guide)\n"
        "  • prompting methods, agent architecture, RAG, embeddings, fine-tuning\n"
        "  • code, integrations, benchmarks, or step-by-step instructions\n\n"
        "Reply SKIP for EVERYTHING ELSE, including:\n"
        "  • personal stories, reflections, or motivational content\n"
        "  • company valuations, funding rounds, stock prices\n"
        "  • community announcements, meetups, ads, contest results\n"
        "  • opinion or prediction posts without actionable technical content\n"
        "  • gaming, sports, politics, or non-AI topics\n\n"
        "Default: SKIP when uncertain.\n\n"
        f"Post:\n{content[:800]}\n\nReply with exactly one word: SAVE or SKIP"
    )
    import requests
    try:
        # sends {"model": "hermes3-llama-3.1-8b", "messages": [{"role": "user", "content": prompt}], ...}
        # and gets back {"choices": [{"message": {"content": "SAVE"}}]}
        r = requests.post(
            f"{LITELLM_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {LITELLM_MASTER_KEY}"},
            json={
                "model": TAG_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 5,
            },
            timeout=30,
        )
        answer = r.json()["choices"][0]["message"]["content"].strip().upper()
        verdict = answer.startswith("SAVE")
        if debug:
            print(f"  [filter] LLM → {answer}")
        return verdict
    except Exception as e:
        print(f"  [filter] LLM unavailable ({e}) -- saving by default")
        return True


# ── LLM helpers ──────────────────────────────────────────────────────────────

def _llm_generate(prompt, num_predict=80):
    import requests
    try:
        r = requests.post(
            f"{LITELLM_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {LITELLM_MASTER_KEY}"},
            json={
                "model": TAG_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": num_predict,
            },
            timeout=30,
        )
        raw = r.json()["choices"][0]["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    except Exception:
        return ""


# sends this to LiteLLM's vision-capable model (same route/prompt
# gmail_sync.py's own ocr_image uses for photographed notes -- verbatim
# transcription, not a scene caption):
#   {"model": "qwen3.8-27b-Expert-Vision", "messages": [{"role": "user",
#    "content": [{"type": "text", "text": "..."}, {"type": "image_url",
#    "image_url": {"url": "data:image/jpeg;base64,..."}}]}], "max_tokens": 2000}
# returns the usual OpenAI chat shape: {"choices": [{"message": {"content": "..."}}]}
# Returns "" (never raises) on failure -- a bad OCR call shouldn't drop the
# whole post, the image itself still gets saved either way. Deliberately
# duplicated from gmail_sync.py's own copy rather than shared -- same
# reasoning as vault-mcp/vault_sync.py's own duplicated chunking logic (see
# /areas/ai-box-k3s.md).
def ocr_image(image_bytes, mime_type):
    import base64
    import requests
    b64 = base64.b64encode(image_bytes).decode("ascii")
    prompt = (
        "Transcribe ALL visible text in this image, verbatim and in full, exactly as written "
        "(keep the original language -- don't translate it). Preserve line breaks and structure "
        "(headings, lists, tables) as plain text where it's clear. If there is little or no text, "
        "instead describe what the image shows in one or two sentences."
    )
    payload = {
        "model": VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
            ],
        }],
        "max_tokens": 2000,
    }
    headers = {"Authorization": f"Bearer {LITELLM_MASTER_KEY}"}

    # Every model on aibox loads lazily (see aibox-config-restart-verify) --
    # a 502/503 here usually just means the vision model was cold and
    # llama-swap hadn't finished swapping it in yet, not that the route is
    # actually down (confirmed live in gmail_sync.py's own version of this).
    # One retry after a short wait covers this without giving up on real data.
    import time
    for attempt in range(2):
        try:
            r = requests.post(f"{LITELLM_URL}/v1/chat/completions", headers=headers, json=payload, timeout=90)
        except Exception as exc:   # connection refused/timeout -- LiteLLM itself unreachable, retrying won't help
            print(f"    ⚠ image OCR failed: {exc}")
            return ""

        if r.status_code in (502, 503) and attempt == 0:
            print(f"    ⚠ image OCR got HTTP {r.status_code} (model likely still cold-loading) -- retrying once in 10s")
            time.sleep(10)
            continue

        if r.status_code != 200:
            print(f"    ⚠ image OCR failed: HTTP {r.status_code} -- {r.text[:500]}")
            return ""

        content = r.json()["choices"][0]["message"]["content"].strip()
        if not content:
            print(f"    ⚠ image OCR returned an empty response (model: {VISION_MODEL})")
        return content

    return ""


# Extracts and translates the first meaningful line of a post to English.
def translate_title(text):
    first_line = ""
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            first_line = line[:120]
            break

    if not first_line:
        return "Untitled"

    if not re.search(r"[а-яёА-ЯЁіїєґ]", first_line):
        return _clean_title(first_line)

    prompt = (
        "Translate this Telegram post title to English.\n"
        "Return ONLY the English title -- no explanation, no quotes, no punctuation at the end.\n"
        "Keep it concise (max 10 words).\n\n"
        f"Title: {first_line}"
    )
    result = _llm_generate(prompt, num_predict=40)
    return _clean_title(result) if result else _clean_title(first_line)


def _clean_title(title):
    title = re.sub(r"\*+", "", title)
    title = re.sub(r"^\s*[-•▸→]+\s*", "", title)
    title = re.sub(r"[\U0001F300-\U0001FFFF]", "", title)
    title = title.strip().strip(".,;:!?")
    return title[:80] or "Untitled"


# Picks 2-5 KNOWN_TAGS for this post via LLM
def extract_tags(text):
    prompt = (
        "Pick 2-5 topic tags for this AI post from the list below.\n"
        "Return ONLY a comma-separated list of tags -- no explanation, no quotes.\n\n"
        f"Available tags: {', '.join(KNOWN_TAGS)}\n\n"
        f"Post:\n{text[:800]}\n\nTags:"
    )
    raw  = _llm_generate(prompt, num_predict=60)
    tags = [t.strip().lower().replace(" ", "-") for t in raw.split(",") if t.strip()]
    return [t for t in tags if t in KNOWN_TAGS][:5]


# ── state ────────────────────────────────────────────────────────────────────

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ── note formatting ──────────────────────────────────────────────────────────

def title_to_filename(title, channel, date_str):
    safe = re.sub(r"[^\w\s-]", "", title).strip()
    safe = re.sub(r"[\s_]+", "-", safe)
    safe = re.sub(r"-{2,}", "-", safe)
    safe = safe[:50] or "post"
    return f"TG-{safe}-{date_str}"


# Turns arbitrary text into a filesystem-safe slug for OCR'd attachment
# filenames. Capped by BYTES, not characters -- these channels are
# Russian-language, and Python's \w treats Cyrillic as ordinary word
# characters that pass straight through uncut, each 2-3 bytes in UTF-8, not
# 1 (real case, gmail_sync.py's own vault, 2026-09-14: a 60-CHARACTER cap on
# a Cyrillic title produced a ~110-BYTE filename). Deliberately duplicated
# from gmail_sync.py's own copy rather than shared, same reasoning as
# ocr_image above.
def slugify(text, max_bytes=50):
    safe = re.sub(r"[^\w\s-]", "", text).strip()
    safe = re.sub(r"[\s_]+", "-", safe)
    safe = re.sub(r"-{2,}", "-", safe)

    encoded = safe.encode("utf-8")
    if len(encoded) > max_bytes:
        encoded = encoded[:max_bytes]
        while encoded:   # back off until this is a valid UTF-8 boundary
            try:
                safe = encoded.decode("utf-8")
                break
            except UnicodeDecodeError:
                encoded = encoded[:-1]
        else:
            safe = ""

    return safe.strip("-")


# Saves an image into vault_root/Attachments -- the SAME top-level folder
# gmail_sync.py's own format_attachments and vault-mcp's ingest_file use, so
# all three ingestion paths share one Attachments dir and never collide on
# filenames. Named after its own transcribed content (e.g.
# "new-model-benchmark-chart.jpg"), not the meaningless "tg-48213.jpg" this
# script would otherwise give it -- falls back to that original name only
# when OCR found nothing to name it after. Returns a markdown block (an
# Obsidian embed link followed by the transcribed text) ready to append to
# the note body; an image whose OCR found nothing still gets the embed link
# alone. Deliberately duplicated from gmail_sync.py's own copy rather than
# shared, same reasoning as ocr_image above.
def format_attachments(vault_ai, images):
    attachments_dir = vault_ai.parent / "Attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)
    chgrp_writable(attachments_dir)

    blocks = []
    for filename, mime_type, data in images:
        text = ocr_image(data, mime_type)

        ext = Path(filename).suffix or ".jpg"
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        base = slugify(first_line) or Path(filename).stem
        dest = attachments_dir / f"{base}{ext}"
        if dest.exists():
            import uuid
            dest = attachments_dir / f"{base}-{uuid.uuid4().hex[:8]}{ext}"
        dest.write_bytes(data)
        chgrp_writable(dest)

        print(f"    ✓ attachment OCR'd: Attachments/{dest.name}" if text
              else f"    ⚠ attachment saved, OCR found no text: Attachments/{dest.name}")

        link = f"![[Attachments/{dest.name}]]"
        blocks.append(f"{link}\n\n{text}" if text else link)

    return "\n\n".join(blocks)


# No `audience` field -- same reasoning as gmail_sync.py's format_note.
def format_note(title, text, channel, post_date, message_id, tags):
    friendly_date   = post_date.strftime("%B %d, %Y %H:%M")
    frontmatter_dt  = post_date.strftime("%Y-%m-%d")
    channel_url     = f"https://t.me/{channel}/{message_id}"
    tags_str        = ", ".join(tags)

    return f"""---
date: {frontmatter_dt}
source: telegram
channel: {channel}
tags: [{tags_str}]
---

# {title}

**Source:** [@{channel}]({channel_url}), {friendly_date}

---

{text.strip()}
"""


# ── sync ─────────────────────────────────────────────────────────────────────

# Downloads a message's image, if it has one -- either a compressed
# Telegram photo (msg.photo) or a file sent as a document whose mime type is
# image/* (msg.document). Returns (filename, mime_type, raw_bytes) or None.
# Only ever one image per message (a Telegram channel post doesn't have
# Gmail-style multi-attachment MIME trees to walk) -- format_attachments
# above still takes a list, for the same call shape as gmail_sync.py's own.
async def get_image_media(client, msg):
    if msg.photo:
        data = await client.download_media(msg, file=bytes)
        return (f"tg-{msg.id}.jpg", "image/jpeg", data) if data else None

    if msg.document and (msg.document.mime_type or "").startswith("image/"):
        filename = f"tg-{msg.id}"
        for attr in msg.document.attributes:
            name = getattr(attr, "file_name", None)
            if name:
                filename = name
                break
        data = await client.download_media(msg, file=bytes)
        return (filename, msg.document.mime_type, data) if data else None

    return None


# Fetches new messages from every channel in CHANNELS, filters/translates/
# tags each one, and writes a vault note for anything worth keeping.
# Returns the number of notes created.
async def sync_async(vault_ai, dry_run=False, no_filter=False, debug=False):
    from telethon import TelegramClient
    from telethon.tl.types import Message

    state   = load_state()
    created = 0

    async with TelegramClient(SESSION_FILE, API_ID, API_HASH) as client:
        for channel in CHANNELS:
            last_id = state.get(f"last_id_{channel}", 0)
            print(f"[tg] @{channel} -- fetching since message_id {last_id}")

            try:
                entity = await client.get_entity(channel)
            except Exception as e:
                print(f"  ✗ could not access @{channel}: {e}")
                continue

            messages = await client.get_messages(
                entity,
                limit=FETCH_LIMIT,
                min_id=last_id,
            )
            messages = sorted(messages, key=lambda m: m.id)

            new_last_id = last_id
            for msg in messages:
                if not isinstance(msg, Message):
                    continue

                text  = msg.text or msg.message or ""
                image = await get_image_media(client, msg)

                if not text.strip() and not image:
                    continue

                note_text = _strip_audio_header(text.strip()) if text.strip() else ""
                if text.strip() and not note_text and not image:
                    print(f"  ✗ skipped (audio-only): {text[:60].strip()}")
                    new_last_id = max(new_last_id, msg.id)
                    continue

                post_date = msg.date.astimezone(timezone.utc)
                date_str  = post_date.strftime("%Y-%m-%d-%H%M")

                # An image-only post (no caption, or an audio-only caption)
                # has nothing for the relevance filter to judge until OCR
                # runs -- skip the pre-filter for those and let it through;
                # a real caption still gets filtered exactly as before.
                if note_text and not no_filter and not is_relevant(note_text, debug=debug):
                    print(f"  ✗ skipped (not relevant): {text[:60].strip()}")
                    new_last_id = max(new_last_id, msg.id)
                    continue

                # OCR costs a real LLM call per image -- skipped in dry-run,
                # same reasoning as gmail_sync.py's own format_attachments call.
                if image and not dry_run:
                    attachments_md = format_attachments(vault_ai, [image])
                    note_text = f"{note_text}\n\n{attachments_md}" if note_text else attachments_md
                elif image:
                    print(f"    (dry-run) would OCR 1 image attachment: {image[0]}")

                if dry_run:
                    raw_first = (note_text.strip().splitlines()[0][:60].strip()
                                 if note_text.strip() else "(image, not OCR'd in dry-run)")
                    title = _clean_title(raw_first)
                else:
                    title = translate_title(note_text) if note_text else "Untitled"

                tag_source = note_text or title
                tags = [] if dry_run else extract_tags(tag_source)

                filename  = title_to_filename(title, channel, date_str) + ".md"
                note_path = vault_ai / filename
                content   = format_note(title, note_text, channel, post_date,
                                        msg.id, tags)

                if dry_run:
                    print(f"  (dry-run) would create: {filename}")
                    print(f"    {text[:80].strip()}")
                else:
                    if note_path.exists():
                        print(f"  ⚠ already exists (skipping): {filename}")
                    else:
                        vault_ai.mkdir(parents=True, exist_ok=True)
                        chgrp_writable(vault_ai)
                        note_path.write_text(content, encoding="utf-8")
                        chgrp_writable(note_path)   # otherwise this (root, in the N8N pod) write leaves the note root:root -- Eldar/Hermes can't edit or delete it later
                        print(f"  ✓ created: {filename}  tags={tags}")
                        created += 1

                new_last_id = max(new_last_id, msg.id)

            if not dry_run:
                state[f"last_id_{channel}"] = new_last_id
            print(f"[tg] @{channel} done -- last_id now {new_last_id}")

    if not dry_run:
        save_state(state)

    print(f"[tg] sync complete -- {created} new notes created")
    return created


# Runs Telethon's interactive login flow once, saving a reusable session file
async def auth_async():
    from telethon import TelegramClient
    async with TelegramClient(SESSION_FILE, API_ID, API_HASH) as client:
        me = await client.get_me()
        print(f"[tg] Authenticated as: {me.first_name} (@{me.username})")
        print(f"[tg] Session saved to: {SESSION_FILE}.session")


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    if not LITELLM_MASTER_KEY:
        raise SystemExit(
            "Set LITELLM_MASTER_KEY first, e.g.:\n"
            "  export LITELLM_MASTER_KEY=$(kubectl get secret -n ai-models litellm-secret "
            "-o jsonpath='{.data.master_key}' | base64 -d)"
        )

    p = argparse.ArgumentParser(description="Telegram channels -> Obsidian sync")
    p.add_argument("--vault",     default=str(DEFAULT_VAULT_AI),
                   help=f"Vault AI folder (default: {DEFAULT_VAULT_AI})")
    p.add_argument("--dry-run",   action="store_true",
                   help="Print what would be created without writing files")
    p.add_argument("--auth",      action="store_true",
                   help="Run interactive authentication (required on first use)")
    p.add_argument("--no-filter", action="store_true",
                   help="Skip LLM relevance filter -- save all posts")
    p.add_argument("--debug",     action="store_true",
                   help="Show filter decisions (keyword / LLM verdict)")
    args = p.parse_args()

    if not API_ID or not API_HASH:
        print("Error: TELEGRAM_API_ID and TELEGRAM_API_HASH must be set")
        sys.exit(1)

    if args.auth:
        asyncio.run(auth_async())
    else:
        asyncio.run(sync_async(
            Path(args.vault),
            dry_run=args.dry_run,
            no_filter=args.no_filter,
            debug=args.debug,
        ))


# parses CLI flags and runs either --auth or a live/dry-run Telegram-to-vault sync
if __name__ == "__main__":
    main()
