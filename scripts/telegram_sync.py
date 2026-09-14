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
# Setup (one-time, in the scripts/.venv already used by vault_sync.py):
#   pip install telethon
#
#   Get an api_id/api_hash from https://my.telegram.org/apps, then:
#     export TELEGRAM_API_ID=12345678
#     export TELEGRAM_API_HASH=abc123def456...
#   First run only (interactive login -- saves a reusable session file):
#     python3 scripts/telegram_sync.py --auth
#   (If a .telegram_session file already exists from the Windows lab, copy
#   it into this scripts/ folder instead and skip --auth entirely.)
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

HERE         = Path(__file__).resolve().parent
STATE_FILE   = HERE / ".telegram_sync_state.json"
SESSION_FILE = str(HERE / ".telegram_session")

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

                text = msg.text or msg.message or ""
                if not text.strip():
                    continue

                note_text = _strip_audio_header(text.strip())
                if not note_text:
                    print(f"  ✗ skipped (audio-only): {text[:60].strip()}")
                    new_last_id = max(new_last_id, msg.id)
                    continue

                post_date = msg.date.astimezone(timezone.utc)
                date_str  = post_date.strftime("%Y-%m-%d-%H%M")

                if not no_filter and not is_relevant(note_text, debug=debug):
                    print(f"  ✗ skipped (not relevant): {text[:60].strip()}")
                    new_last_id = max(new_last_id, msg.id)
                    continue

                if dry_run:
                    raw_first = note_text.strip().splitlines()[0][:60].strip()
                    title = _clean_title(raw_first)
                else:
                    title = translate_title(note_text)

                tags = [] if dry_run else extract_tags(note_text)

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
                        note_path.write_text(content, encoding="utf-8")
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
