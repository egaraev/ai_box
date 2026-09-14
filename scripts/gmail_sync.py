# gmail_sync.py -- Gmail -> Obsidian vault sync, routed by topic.
#
# Ported from the old Windows-lab pipeline (ai_ml_cv_projects/ai-lab/step10_n8n/gmail_sync.py),
# same treatment as vault_sync.py got: paths repointed to aibox, LiteLLM calls
# repointed to aibox's own gateway. One deliberate behavior change: tag
# extraction/title cleanup now call hermes3-llama-3.1-8b directly instead of
# the old lab's "auto" model -- these are short, internal/system calls (not
# live user chat), so they should bypass arch-router's classification step
# the same way media-gen-mcp's captioning and vault-mcp's vision captioning
# already do for their own internal LLM calls (see /areas/ai-box-k3s.md).
#
# Generalized to multiple topics (2026-09-14, was AI-only before): Eldar
# sends himself an email with the subject starting with a topic word --
# "AI ...", "Health ...", "Legal ...", "Family ...", "Household ...", or
# "Reference ..." -- and that word decides which vault folder the note lands
# in. Matches vault-mcp's own ALLOWED_FOLDERS taxonomy (apps/vault-mcp/
# server.py) so a note routed here lands exactly where Hermes would route the
# same content itself -- one consistent folder set across both ingestion
# paths. Tags are freeform (2-5 short LLM-picked tags, no fixed vocabulary)
# instead of the old AI-only KNOWN_TAGS list -- a fixed vocabulary doesn't
# scale to topics that didn't exist yet when it was written, and Hermes's own
# ingest_note already tags freeform the same way.
#
# Tracks synced message IDs in a local state file so re-running only
# processes new mail. Standalone for now (same as vault_sync.py) -- runs via
# the N8N Code node, see aibox-ingestion-pipeline.json.
#
# Setup (one-time, in the scripts/.venv already used by vault_sync.py):
#   pip install google-auth google-auth-oauthlib google-api-python-client
#
#   Needs Gmail OAuth credentials at ~/.gmail-mcp/credentials.json and
#   ~/.gmail-mcp/gcp-oauth.keys.json -- reuse the same ones the Windows lab
#   used if the gmail-mcp plugin isn't already set up on aibox (Google's
#   OAuth refresh token isn't tied to one machine, so just copying both
#   files over works, no need to redo the consent flow), or set gmail-mcp up
#   fresh here. Running inside the N8N pod: GMAIL_MCP_DIR (set in
#   n8n-deployment.yaml) points this at a subfolder of the persistent
#   n8n-data volume instead of $HOME/.gmail-mcp (which is /root, ephemeral,
#   see STATE_DIR below) -- copy the two files there.
#
# Usage:
#   export LITELLM_MASTER_KEY=$(kubectl get secret -n ai-models litellm-secret -o jsonpath='{.data.master_key}' | base64 -d)
#   python3 scripts/gmail_sync.py --dry-run                # preview only
#   python3 scripts/gmail_sync.py                          # live sync
#   python3 scripts/gmail_sync.py --preseed                # mark existing vault notes as already-synced, run once

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from email.header import decode_header
from pathlib import Path

# ── paths / config ──────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent

# Where the state file lives. Defaults to right next to this script (HERE),
# same as always, for the host scripts/.venv case -- but inside the N8N pod
# this script's own directory (/opt/scripts) is baked into the image, not on
# any mounted volume, so anything written there is gone the next time the
# pod restarts. GMAIL_SYNC_STATE_DIR overrides this to a subfolder of the
# n8n-data hostPath mount instead (persistent) -- set only in
# n8n-deployment.yaml's env block, unset (falls back to HERE) everywhere
# else. Same reasoning as telegram_sync.py's STATE_DIR.
STATE_DIR = Path(os.environ.get("GMAIL_SYNC_STATE_DIR", str(HERE)))
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / ".gmail_sync_state.json"

# GMAIL_MCP_DIR: defaults to ~/.gmail-mcp (matches the gmail-mcp plugin
# convention) -- overridable the same way for the pod, since $HOME there is
# /root (confirmed live), not wherever the actual credential files live.
GMAIL_MCP_DIR   = Path(os.environ.get("GMAIL_MCP_DIR", str(Path.home() / ".gmail-mcp")))
TOKEN_FILE      = GMAIL_MCP_DIR / "credentials.json"
OAUTH_KEYS_FILE = GMAIL_MCP_DIR / "gcp-oauth.keys.json"

# Base of the vault -- each topic below gets its own subfolder under this,
# not one fixed destination like the old AI-only version had.
DEFAULT_VAULT_ROOT = Path("/data/obsidian_vault")

# Recognized subject-prefix topics -> destination vault folder. Matches
# vault-mcp's ALLOWED_FOLDERS (apps/vault-mcp/server.py) minus Inbox (nothing
# here should silently land in Inbox -- an email whose subject doesn't start
# with one of these words just isn't a match, same as the old AI-only
# behavior for anything not starting with "AI"). Order matters only in that
# no word here is a prefix of another, so a plain startswith check is enough.
TOPICS = {
    "AI": "AI",
    "HEALTH": "Health",
    "LEGAL": "Legal",
    "FAMILY": "Family",
    "HOUSEHOLD": "Household",
    "REFERENCE": "Reference",
}

DEFAULT_QUERY = "from:me subject:(" + " OR ".join(TOPICS) + ")"
DEFAULT_SINCE = ""
MAX_RESULTS   = 500

# Same NodePort/env-var convention as scripts/vault_sync.py -- overridable
# so this also works run in-cluster (N8N Execute Command), where localhost
# doesn't reach the node's NodePorts.
LITELLM_URL = os.environ.get("LITELLM_URL", "http://localhost:30400")
LITELLM_MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY")
# Fast local model, called directly -- see header comment on why this isn't "auto"
TAG_MODEL = "hermes3-llama-3.1-8b"

# ── topic detection ──────────────────────────────────────────────────────────

# Which topic word (if any) this subject starts with -- decides the
# destination folder. Returns (topic_word, folder) or (None, None) if the
# subject doesn't start with any recognized topic.
def detect_topic(subject):
    upper = subject.strip().upper()
    for word, folder in TOPICS.items():
        if upper.startswith(word):
            return word, folder
    return None, None

# ── auth ─────────────────────────────────────────────────────────────────────

# Builds a Gmail API service using the existing gmail-mcp credentials
def get_gmail_service():
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError:
        print("Missing deps. Run:")
        print("  pip install google-auth google-auth-oauthlib google-api-python-client")
        sys.exit(1)

    if not TOKEN_FILE.exists():
        print(f"Token file not found: {TOKEN_FILE}")
        sys.exit(1)
    if not OAUTH_KEYS_FILE.exists():
        print(f"OAuth keys not found: {OAUTH_KEYS_FILE}")
        sys.exit(1)

    token_data = json.loads(TOKEN_FILE.read_text())
    keys_data  = json.loads(OAUTH_KEYS_FILE.read_text())
    installed  = keys_data.get("installed", keys_data)

    creds = Credentials(
        token         = token_data.get("access_token"),
        refresh_token = token_data.get("refresh_token"),
        token_uri     = installed.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id     = installed["client_id"],
        client_secret = installed["client_secret"],
        scopes        = token_data.get("scope", "").split(),
    )

    if not creds.valid:
        creds.refresh(Request())
        token_data["access_token"] = creds.token
        TOKEN_FILE.write_text(json.dumps(token_data, indent=2))

    return build("gmail", "v1", credentials=creds)


# ── LLM helpers ──────────────────────────────────────────────────────────────

# sends this to LiteLLM's OpenAI-compatible chat endpoint:
#   {"model": "hermes3-llama-3.1-8b", "messages": [{"role": "user", "content": "..."}], ...}
# returns e.g. {"choices": [{"message": {"content": "..."}}]}
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


# Asks the LLM for 2-5 short topic tags -- freeform, no fixed vocabulary
# (unlike the old AI-only KNOWN_TAGS list), so this works for any topic
# without hand-curating a tag list per topic. folder tells the LLM what kind
# of content this is (Health, Legal, ...) so the tags actually fit -- same
# style Hermes's own ingest_note tool already uses for freeform tagging.
def extract_tags(text, folder):
    prompt = (
        f"Pick 2-5 short topic tags for this {folder} note.\n"
        "Return ONLY a comma-separated list of lowercase, hyphenated tags -- "
        "no explanation, no quotes.\n\n"
        f"Content:\n{text[:800]}\n\nTags:"
    )
    raw = _llm_generate(prompt, num_predict=60)
    return [t.strip().lower().replace(" ", "-") for t in raw.split(",") if t.strip()][:5]


# Derives a clean human-readable title from the email.
# - Strips the leading topic word + "-"/":"/etc (redundant -- the folder
#   already says what this is)
# - Falls back to the first meaningful line of the body when the subject is
#   just the bare topic word on its own
def clean_title(subject, body, topic_word):
    title = subject.strip()

    if re.fullmatch(rf"{topic_word}\s*[-–—:]?\s*", title, re.IGNORECASE):
        first_line = body.strip().splitlines()[0][:80].strip() if body else ""
        title = first_line or title

    title = re.sub(rf"^{topic_word}\s*[-–—:]\s*", "", title, flags=re.IGNORECASE).strip()
    title = re.sub(rf"^{topic_word}\s+", "", title, flags=re.IGNORECASE).strip()

    return title or subject.strip()


# ── state ────────────────────────────────────────────────────────────────────

def load_state():
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            return set(data.get("synced_ids", []))
        except Exception:
            return set()
    return set()


def save_state(synced_ids):
    STATE_FILE.write_text(
        json.dumps({"synced_ids": sorted(synced_ids)}, indent=2),
        encoding="utf-8",
    )


# ── email parsing ────────────────────────────────────────────────────────────

def decode_mime_header(value):
    parts = decode_header(value or "")
    out = []
    for chunk, enc in parts:
        if isinstance(chunk, bytes):
            out.append(chunk.decode(enc or "utf-8", errors="replace"))
        else:
            out.append(chunk)
    return "".join(out).strip()


# Extracts the plain-text body from a Gmail API message payload
# (recurses into multipart messages until it finds a text/plain part)
def get_body(msg):
    payload = msg.get("payload", {})

    def _extract(part):
        mime = part.get("mimeType", "")
        if mime == "text/plain":
            data = part.get("body", {}).get("data", "")
            if data:
                import base64
                return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
        if mime.startswith("multipart/"):
            for sub in part.get("parts", []):
                text = _extract(sub)
                if text:
                    return text
        return ""

    return _extract(payload).strip()


def title_to_filename(title, date_str=""):
    safe = re.sub(r"[^\w\s-]", "", title).strip()
    safe = re.sub(r"[\s_]+", "-", safe)
    safe = re.sub(r"-{2,}", "-", safe)
    safe = safe[:60]

    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(date_str)
        suffix = dt.strftime("%Y-%m-%d-%H%M")
    except Exception:
        suffix = ""

    return f"{safe}-{suffix}" if suffix else safe


# Formats an email as an Obsidian markdown note with frontmatter.
# No `audience` field -- same as the rest of the vault, it stays unset
# (not defaulted to "everyone") until family Hermes profiles exist.
def format_note(title, subject, sender, date_str, body, tags):
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(date_str)
        friendly_date   = dt.strftime("%B %d, %Y")
        frontmatter_date = dt.strftime("%Y-%m-%d")
    except Exception:
        friendly_date    = date_str
        frontmatter_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    sender_name = re.sub(r"<[^>]+>", "", sender).strip().strip('"') or sender
    body        = re.sub(r"\n{3,}", "\n\n", body).strip()
    tags_str    = ", ".join(tags)

    return f"""---
date: {frontmatter_date}
subject: {subject}
source: gmail
tags: [{tags_str}]
---

# {title}

**Source:** Email from {sender_name}, {friendly_date}

---

{body}
"""


# ── sync ─────────────────────────────────────────────────────────────────────

# Fetches matching emails, creates a vault note (in the folder its subject's
# topic word maps to) for any new topic-prefixed message, and records synced
# IDs. Returns the number of notes created.
def sync(vault_root, query, since="", dry_run=False):
    service    = get_gmail_service()
    synced_ids = load_state()

    full_query = query
    if since:
        full_query += f" after:{since.replace('-', '/')}"
    print(f"[gmail] Searching: {full_query!r}")

    messages   = []
    page_token = None
    while len(messages) < MAX_RESULTS:
        params = {"userId": "me", "q": full_query,
                  "maxResults": min(100, MAX_RESULTS - len(messages))}
        if page_token:
            params["pageToken"] = page_token
        result = service.users().messages().list(**params).execute()
        messages.extend(result.get("messages", []))
        page_token = result.get("nextPageToken")
        if not page_token:
            break

    print(f"[gmail] {len(messages)} messages found, {len(synced_ids)} already synced")

    new_ids = set()
    created = 0

    for item in messages:
        msg_id = item["id"]
        if msg_id in synced_ids:
            continue

        msg = service.users().messages().get(
            userId="me", id=msg_id, format="full"
        ).execute()

        headers = {h["name"]: h["value"]
                   for h in msg.get("payload", {}).get("headers", [])}
        subject = decode_mime_header(headers.get("Subject", "(no subject)"))

        topic, folder = detect_topic(subject)
        if not topic:
            new_ids.add(msg_id)
            continue

        sender = decode_mime_header(headers.get("From", "Unknown"))
        date   = headers.get("Date", "")
        body   = get_body(msg)

        if not body:
            print(f"  ⚠ no body -- skipping: {subject[:60]}")
            new_ids.add(msg_id)
            continue

        title      = clean_title(subject, body, topic)
        filename   = title_to_filename(title, date) + ".md"
        target_dir = vault_root / folder
        note_path  = target_dir / filename

        tags = [] if dry_run else extract_tags(title + "\n\n" + body[:800], folder)

        content = format_note(title, subject, sender, date, body, tags)

        if dry_run:
            print(f"  (dry-run) would create: {folder}/{filename}")
            print(f"    tags: {tags}")
        else:
            if note_path.exists():
                print(f"  ⚠ already exists (skipping): {folder}/{filename}")
            else:
                target_dir.mkdir(parents=True, exist_ok=True)
                note_path.write_text(content, encoding="utf-8")
                print(f"  ✓ created: {folder}/{filename}  tags={tags}")
                created += 1

        new_ids.add(msg_id)

    if not dry_run and new_ids:
        save_state(synced_ids | new_ids)

    print(f"[gmail] done -- {created} new notes created")
    return created


# Marks Gmail messages whose file already exists in the vault as synced --
# use once after setup so the first real sync doesn't re-create existing notes
def preseed(vault_root, query):
    service    = get_gmail_service()
    synced_ids = load_state()
    new_ids    = set()

    print("[preseed] Scanning Gmail for messages already in vault...")
    messages   = []
    page_token = None
    while len(messages) < MAX_RESULTS:
        params = {"userId": "me", "q": query,
                  "maxResults": min(100, MAX_RESULTS - len(messages))}
        if page_token:
            params["pageToken"] = page_token
        result = service.users().messages().list(**params).execute()
        messages.extend(result.get("messages", []))
        page_token = result.get("nextPageToken")
        if not page_token:
            break

    for item in messages:
        msg_id = item["id"]
        if msg_id in synced_ids:
            continue
        msg = service.users().messages().get(
            userId="me", id=msg_id, format="metadata",
            metadataHeaders=["Subject", "Date"],
        ).execute()
        headers = {h["name"]: h["value"]
                   for h in msg.get("payload", {}).get("headers", [])}
        subject = decode_mime_header(headers.get("Subject", ""))
        date    = headers.get("Date", "")

        topic, folder = detect_topic(subject)
        if not topic:
            new_ids.add(msg_id)
            continue

        title    = clean_title(subject, "", topic)
        filename = title_to_filename(title, date) + ".md"
        if (vault_root / folder / filename).exists():
            print(f"  ✓ marking as synced: {folder}/{filename}")
            new_ids.add(msg_id)
            continue

        old_safe = re.sub(r"[^\w\s-]", "", subject).strip()
        old_safe = re.sub(r"[\s_]+", "-", old_safe)
        old_safe = re.sub(r"-{2,}", "-", old_safe)
        for candidate in (vault_root / folder).glob(old_safe[:40] + "*.md"):
            print(f"  ✓ marking as synced (old-style): {folder}/{candidate.name}")
            new_ids.add(msg_id)
            break

    if new_ids:
        save_state(synced_ids | new_ids)
        print(f"[preseed] marked {len(new_ids)} messages as synced")
    else:
        print("[preseed] nothing new to mark")


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    if not LITELLM_MASTER_KEY:
        raise SystemExit(
            "Set LITELLM_MASTER_KEY first, e.g.:\n"
            "  export LITELLM_MASTER_KEY=$(kubectl get secret -n ai-models litellm-secret "
            "-o jsonpath='{.data.master_key}' | base64 -d)"
        )

    p = argparse.ArgumentParser(description="Gmail -> Obsidian sync, routed by topic")
    p.add_argument("--vault",   default=str(DEFAULT_VAULT_ROOT),
                   help=f"Path to vault root (default: {DEFAULT_VAULT_ROOT})")
    p.add_argument("--query",   default=DEFAULT_QUERY,
                   help=f"Gmail search query (default: {DEFAULT_QUERY!r})")
    p.add_argument("--since",   default=DEFAULT_SINCE, metavar="YYYY-MM-DD",
                   help="Only sync emails on or after this date")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be created without writing files")
    p.add_argument("--preseed", action="store_true",
                   help="Mark existing vault files as synced (run once after setup)")
    args = p.parse_args()

    vault_root = Path(args.vault)
    if args.preseed:
        preseed(vault_root, query=args.query)
    else:
        sync(vault_root, query=args.query, since=args.since, dry_run=args.dry_run)


# parses CLI flags and runs either --preseed or a live/dry-run Gmail-to-vault sync
if __name__ == "__main__":
    main()
