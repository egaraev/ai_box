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
# Image attachments now get OCR'd too (2026-09-14): any image on a
# topic-matched email is saved into vault_root/Attachments (same dir
# vault-mcp's ingest_file uses) and transcribed via the same locked vision
# route/prompt ingest_file uses for photographed notes -- verbatim text, not
# a scene caption. The transcription becomes the note body when the email
# itself has none (a bare-subject photo), or gets appended below it
# otherwise, and an Obsidian embed link (![[Attachments/...]]) always goes
# in either way so the original image stays one click away.
#
# MOVIE topic (added 2026-09-30): a subject/body/image movie-wishlist
# pipeline, separate from the note-writing topics above -- a "Movie: ..."
# (or bare "Movie" + a poster/screenshot attachment) email writes straight
# into Eldar's movies watch-history.yaml instead of creating a vault note.
# Title comes from the subject (clean_title's existing prefix-stripping
# already does this for free), the body (a short LLM extraction call) if
# the subject is bare, or the image (a new vision call, recognize_movie_
# title, prompted for a title rather than ocr_image's verbatim transcript)
# if there's no usable text at all.
#
# Deliberately NOT a blind append: Eldar was explicit he doesn't want this
# silently writing duplicates, and a plain normalized-string dedup check
# would miss real duplicates anyway (e.g. "Skyfall" vs. his existing "007:
# Координаты «Скайфолл»" entry -- same film, different language). So the
# extracted title plus the full current watch-history (same text shape
# movies-mcp's own _watch_history_text builds, mirrored locally below) goes
# to one more LLM call -- routed to LiteLLM's "auto" model, not the small
# fast TAG_MODEL below, since this needs real cross-lingual/fuzzy-match
# judgment, not a cheap tag pick -- which decides NEW vs. already-seen/
# already-listed. Only NEW gets written, always into
# suggested_not_yet_watched (never wishlist -- that stays reserved for
# something Eldar tells the movies agent directly, see
# apps/movies-mcp/server.py's own STATUS_LISTS comment). Any parse failure
# or LLM-call failure is treated as "don't write" -- fails safe, on
# purpose: a missed add just means re-sending the email, a wrongly-added
# title is the worse mistake here.
#
# Reads and writes apps/movies-mcp/server.py's own watch-history.yaml
# directly (MOVIE_WATCH_HISTORY_PATH below) -- both pods mount the same
# /data/obsidian_vault hostPath, and movies-mcp only exposes ask_movies/
# check_anticipated as MCP tools over streamable-http, not a plain REST
# endpoint this script could call with `requests`. Small local copy of
# server.py's own normalize/list/write logic, same self-contained-script
# convention as everywhere else in this repo (see vault_merge_duplicates.
# py's own comment on why nothing here is a shared import).
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
import yaml   # already a scripts/.venv dep (vault_sync.py uses it too) -- movie watch-history read/write below

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
# vault-mcp's ALLOWED_FOLDERS (apps/vault-mcp/vault_core.py) minus Inbox
# (nothing here should silently land in Inbox -- an email whose subject
# doesn't start with one of these words just isn't a match, same as the old
# AI-only behavior for anything not starting with "AI"). Order matters only
# in that no word here is a prefix of another, so a plain startswith check
# is enough.
#
# Expanded 2026-09-17 alongside vault-mcp's own ALLOWED_FOLDERS -- see that
# file's comment for the Work-vs-Legal and Finance-vs-Transport boundary
# rules, which apply here too.
TOPICS = {
    "AI": "AI",
    "HEALTH": "Health",
    "LEGAL": "Legal",
    "FAMILY": "Family",
    "HOUSEHOLD": "Household",
    "REFERENCE": "Reference",
    "WORK": "Work",
    "FINANCE": "Finance",
    "TRANSPORT": "Transport",
    "TRAVELS": "Travels",
    "EDUCATION": "Education",
    # Not a note-writing topic like the others -- see the MOVIE-topic header
    # comment above. "Movies" here only feeds DEFAULT_QUERY's Gmail search;
    # sync() special-cases this topic before it ever reaches the folder/
    # note-writing code path below.
    "MOVIE": "Movies",
}

DEFAULT_QUERY = "from:me subject:(" + " OR ".join(TOPICS) + ")"
DEFAULT_SINCE = ""
MAX_RESULTS   = 500

# Same gid/convention as vault-mcp's own _chgrp_writable (apps/vault-mcp/
# server.py) -- if that one ever changes, change this too. The N8N pod
# writes into the vault as root same as vault-mcp's pod does, and root can
# chown to any group regardless of who currently owns the file, so this is
# safe to call unconditionally on anything this script just wrote. Without
# it, notes/attachments land root:root and Eldar (and Hermes/vault-mcp,
# gid 10000) can't edit or delete them later.
HERMES_GID = 10000

def chgrp_writable(path, dir_mode=0o775, file_mode=0o664):
    mode = dir_mode if path.is_dir() else file_mode
    os.chown(path, -1, HERMES_GID)   # -1 leaves the owner (root) unchanged, only the group changes
    path.chmod(mode)

# Same NodePort/env-var convention as scripts/vault_sync.py -- overridable
# so this also works run in-cluster (N8N Execute Command), where localhost
# doesn't reach the node's NodePorts.
LITELLM_URL = os.environ.get("LITELLM_URL", "http://localhost:30400")
LITELLM_MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY")
# Fast local model, called directly -- see header comment on why this isn't "auto"
TAG_MODEL = "hermes3-llama-3.1-8b"
# Same locked vision route vault-mcp's ingest_file / Hermes's own
# image-in-chat path already use (custom_router_hook.py's _has_image check) --
# verbatim transcription of a photographed note/document, not a scene caption.
VISION_MODEL = "qwen3.8-27b-Expert-Vision"
# Added 2026-09-30 for the MOVIE-topic dedup decision (see header comment) --
# deliberately "auto" (LiteLLM's own router/cloud-fallback), NOT TAG_MODEL:
# Eldar's own call -- hermes3-llama-3.1-8b is too weak for the cross-lingual/
# fuzzy-match judgment this needs ("Skyfall" == his existing "007:
# Координаты «Скайфолл»" entry), unlike the short mechanical tag/title picks
# TAG_MODEL is used for elsewhere in this file.
REASONING_MODEL = "auto"

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
# model defaults to TAG_MODEL (every existing caller's fast/mechanical use
# case) -- pass model=REASONING_MODEL for a call that needs real judgment
# instead (added 2026-09-30 for the MOVIE-topic dedup decision below).
#
# timeout/label added 2026-09-30, same day: every existing caller here sends
# a short prompt (a subject line, an 800-char excerpt) so the original flat
# 30s/no-retry was fine, but decide_movie_status's prompt includes the WHOLE
# watch history (confirmed live: ~26k chars / ~6.4k tokens with Eldar's real
# history) -- that's a lot for a local model to chew through, especially
# cold, and 30s silently wasn't enough on the first live test (empty "" came
# back, which decide_movie_status could only report as "unparseable reply",
# not the real cause). Longer default timeout plus the same one-retry-on-
# cold-model pattern _vision_request already uses below, and -- the actual
# fix for the silent-failure problem -- print WHY on any failure instead of
# swallowing it, exactly like _vision_request already does.
def _llm_generate(prompt, num_predict=80, model=None, timeout=30, label="LLM call"):
    import requests
    import time
    payload = {
        "model": model or TAG_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": num_predict,
    }
    headers = {"Authorization": f"Bearer {LITELLM_MASTER_KEY}"}
    for attempt in range(2):
        try:
            r = requests.post(f"{LITELLM_URL}/v1/chat/completions", headers=headers, json=payload, timeout=timeout)
        except Exception as exc:   # connection refused/timeout -- print it, don't just return ""
            print(f"    ⚠ {label} failed: {exc}")
            return ""

        if r.status_code in (502, 503) and attempt == 0:
            print(f"    ⚠ {label} got HTTP {r.status_code} (model likely still cold-loading) -- retrying once in 10s")
            time.sleep(10)
            continue

        if r.status_code != 200:
            print(f"    ⚠ {label} failed: HTTP {r.status_code} -- {r.text[:500]}")
            return ""

        raw = r.json()["choices"][0]["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    return ""


# sends this to LiteLLM's vision-capable model (same route/prompt vault-mcp's
# ingest_file uses for photographed notes -- verbatim transcription, not a
# scene caption):
#   {"model": "qwen3.8-27b-Expert-Vision", "messages": [{"role": "user",
#    "content": [{"type": "text", "text": "..."}, {"type": "image_url",
#    "image_url": {"url": "data:image/jpeg;base64,..."}}]}], "max_tokens": 2000}
# returns the usual OpenAI chat shape: {"choices": [{"message": {"content": "..."}}]}
# Returns "" (never raises) on failure -- a bad vision call shouldn't drop the
# whole email, the attachment itself still gets saved either way.
#
# Shared by ocr_image (verbatim transcription) and recognize_movie_title
# (added 2026-09-30, asks for just a title instead) -- same model/retry/error
# handling either way, only the prompt and label differ, so this holds the
# common HTTP part and each caller below supplies its own prompt.
def _vision_request(prompt, image_bytes, mime_type, max_tokens=2000, label="vision call"):
    import base64
    import requests
    b64 = base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "model": VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
            ],
        }],
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {LITELLM_MASTER_KEY}"}

    # Every model on aibox loads lazily (see aibox-config-restart-verify) --
    # a 502/503 from LiteLLM here usually just means the vision model was
    # cold and llama-swap hadn't finished swapping it in yet, not that the
    # route is actually down. Confirmed 2026-09-14: a request landing right
    # as the model was cold-starting 502'd, while later images in the same
    # run (model now warm) OCR'd fine. One retry after a short wait covers
    # this without giving up on real data.
    import time
    for attempt in range(2):
        try:
            r = requests.post(f"{LITELLM_URL}/v1/chat/completions", headers=headers, json=payload, timeout=90)
        except Exception as exc:   # connection refused/timeout -- LiteLLM itself unreachable, retrying won't help
            print(f"    ⚠ {label} failed: {exc}")
            return ""

        if r.status_code in (502, 503) and attempt == 0:
            print(f"    ⚠ {label} got HTTP {r.status_code} (model likely still cold-loading) -- retrying once in 10s")
            time.sleep(10)
            continue

        # print the server's own response body on a non-2xx instead of just
        # raise_for_status()'s terse "500 Server Error" -- that body is the
        # only way to tell "wrong model name" from "vision route down" from
        # "image too large" apart, and the previous version threw it away
        if r.status_code != 200:
            print(f"    ⚠ {label} failed: HTTP {r.status_code} -- {r.text[:500]}")
            return ""

        # same <think>...</think> stripping _llm_generate already does for
        # the text route below -- added 2026-09-30 alongside raising
        # recognize_movie_title's own token budget: never actually needed
        # before (every existing caller here got by fine at the old, tight
        # budgets that apparently never left room for a visible think block
        # in the response), but now that recognize_movie_title has enough
        # budget to let the model finish thinking AND answer, a raw
        # unstripped block could otherwise leak into the recognized title,
        # or into an OCR'd note's saved text.
        raw = r.json()["choices"][0]["message"]["content"]
        content = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if not content:
            print(f"    ⚠ {label} returned an empty response (model: {VISION_MODEL})")
        return content

    return ""


def ocr_image(image_bytes, mime_type):
    prompt = (
        "Transcribe ALL visible text in this image, verbatim and in full, exactly as written "
        "(keep the original language -- don't translate it). Preserve line breaks and structure "
        "(headings, lists, tables) as plain text where it's clear. If there is little or no text, "
        "instead describe what the image shows in one or two sentences."
    )
    return _vision_request(prompt, image_bytes, mime_type, max_tokens=2000, label="image OCR")


# Added 2026-09-30 for the MOVIE topic -- unlike ocr_image (verbatim
# transcription), this asks the vision model to identify a movie/show title
# directly from a poster, streaming-site card, or screenshot.
def recognize_movie_title(image_bytes, mime_type):
    prompt = (
        "This image is a movie or TV show poster, streaming-site card, or screenshot. "
        "Identify the movie or show title it shows. Reply with ONLY the title itself, "
        "nothing else -- no year, no quotes, no explanation. "
        "If you cannot confidently identify a specific title, reply with exactly: UNKNOWN"
    )
    # max_tokens bumped from 40 to 2000, 2026-09-30: confirmed live this was
    # the actual failure -- HTTP 200, empty content, no error anywhere,
    # exactly the same signature as decide_movie_status's own bug (see its
    # own comment). VISION_MODEL is the same Qwen3.8-27B-Expert family as
    # REASONING_MODEL's target, so the same enable_thinking budget trap
    # applies here: 40 tokens is enough to open a <think> block and nowhere
    # near enough to close it and still write a title. _vision_request
    # doesn't strip <think> tags the way _llm_generate does (never needed
    # to, until now) -- ocr_image's own 2000-token budget is proven safe
    # here already, so reuse it rather than guess a smaller number that
    # might still be short.
    return _vision_request(prompt, image_bytes, mime_type, max_tokens=2000, label="movie title recognition")


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


# Asks the LLM for a short, human-readable title from real note content --
# used only by clean_title's bare-topic-word fallback below, for the same
# reason extract_tags asks the LLM for tags instead of using fixed words:
# the content's own first line often isn't something a person would call
# this note. Returns "" (never raises) on failure -- clean_title has its own
# first-line fallback for that case.
def generate_title(text, topic_word):
    prompt = (
        f"Write a short, specific title (3-8 words, plain text, no quotes, no "
        f"markdown) for this {topic_word} note, based on its content:\n\n"
        f"{text[:800]}\n\nTitle:"
    )
    raw = _llm_generate(prompt, num_predict=20)
    return raw.strip().strip('"').strip("'")


# Added 2026-09-30 for the MOVIE topic -- used when a "Movie" email's subject
# is bare (no title after it) but the body has text, before falling back to
# the image. Same short-extraction shape as generate_title above but asking
# for a title FROM the text rather than a title FOR it.
# Returns a LIST of titles, not a single one -- confirmed live 2026-09-30:
# Eldar's own real usage is to list several titles in one email body (e.g.
# names of movies image recognition couldn't identify from a separate
# batch), same "more than one per email" shape the image branch already
# handles in process_movie_email. An earlier single-title version of this
# ("what single movie title is this about") is exactly why that real email
# came back empty -- not a call failure, just a prompt asking a genuinely
# different question than what the body contained. Empty list means none
# found, same as the old "UNKNOWN" case.
def extract_movie_titles_from_text(text):
    prompt = (
        "List every distinct movie or TV show title mentioned in this email, one per "
        "line, nothing else -- no year, no quotes, no numbering, no bullets, no "
        "explanation. If none are clearly mentioned, reply with exactly: NONE\n\n"
        f"Email text:\n{text[:800]}\n\nTitles:"
    )
    raw = _llm_generate(prompt, num_predict=200, label="movie title extraction (body)")
    if not raw or raw.strip().upper() == "NONE":
        return []
    titles = []
    for line in raw.splitlines():
        line = line.strip().strip('"').strip("'")
        line = re.sub(r"^[-*\d.)\s]+", "", line).strip()   # strip stray "1." / "-" / "*" prefixes
        if line and line.upper() != "NONE":
            titles.append(line)
    return titles


# Derives a clean human-readable title from the email.
# - Strips the leading topic word + "-"/":"/etc (redundant -- the folder
#   already says what this is)
# - Falls back to an LLM-generated title from the body when the subject is
#   just the bare topic word on its own (dry_run skips the real LLM call and
#   falls back further, to the first meaningful line, same as before)
def clean_title(subject, body, topic_word, dry_run=False):
    title = subject.strip()

    if re.fullmatch(rf"{topic_word}\s*[-–—:]?\s*", title, re.IGNORECASE):
        # Body is the only real content here (a bare-subject photo, e.g.).
        # Strip bare Obsidian embed links (![[Attachments/...]]) first --
        # that's markdown syntax, not content, and shouldn't feed the title
        # prompt or become the first-line fallback either.
        real_text = "\n".join(
            line.strip() for line in body.strip().splitlines()
            if line.strip() and not re.fullmatch(r"!?\[\[.*\]\]", line.strip())
        )

        title = ""
        if real_text and not dry_run:
            title = generate_title(real_text, topic_word)

        if not title and real_text:
            # LLM down/empty, or dry-run -- first real line beats nothing.
            # Confirmed in practice this can be a bad title (a rotated
            # prescription photo transcribed its top-corner tracking code
            # "4N0AE107F5" before anything readable, and that became the
            # note's whole title/filename) -- exactly what generate_title
            # above exists to avoid when it's actually available.
            title = real_text.splitlines()[0][:80]

        title = title or subject.strip()

    title = re.sub(rf"^{topic_word}\s*[-–—:]\s*", "", title, flags=re.IGNORECASE).strip()
    title = re.sub(rf"^{topic_word}\s+", "", title, flags=re.IGNORECASE).strip()

    return title or subject.strip()


# ── movie wishlist pipeline (MOVIE topic, added 2026-09-30) ──────────────────
# See the MOVIE-topic header comment at the top of this file for the full
# design. This block is a self-contained mirror of apps/movies-mcp/server.py's
# own watch-history schema/logic -- same convention as everywhere else in this
# repo (see vault_merge_duplicates.py's own comment on why nothing here is a
# shared import across the two separate deployables).

MOVIE_WATCH_HISTORY_PATH = DEFAULT_VAULT_ROOT / "_system" / "specialist-agents" / "movies" / "watch-history.yaml"
# mirrors server.py's own STATUS_LISTS/PROGRESS_LISTS -- see that file's own
# comment for why wishlist is in PROGRESS_LISTS and anticipated isn't
MOVIE_STATUS_LISTS = ("completed", "on_hold", "suggested_not_yet_watched", "wishlist", "anticipated")
MOVIE_PROGRESS_LISTS = ("completed", "on_hold", "suggested_not_yet_watched", "wishlist")

# mirrors server.py's own SEASON_SUFFIX_RE -- strips a trailing "season 3" /
# "s3" / "part 2" so e.g. "Silo season 3" and "Silo" match as the same title
MOVIE_SEASON_SUFFIX_RE = re.compile(r"\s+(season\s*\d+|s\d+|part\s*\d+)$", re.IGNORECASE)


def normalize_movie_title(title):
    return MOVIE_SEASON_SUFFIX_RE.sub("", title.strip()).strip()


def load_movie_watch_history():
    try:
        with open(MOVIE_WATCH_HISTORY_PATH) as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        data = {}
    for key in ("interests",) + MOVIE_STATUS_LISTS:
        data.setdefault(key, [])
    return data


# same atomic-write-then-rename pattern as server.py's own _save_watch_history,
# plus chgrp_writable -- this script runs as root inside the n8n pod, so
# without it the file would go root-owned and Eldar/Hermes/vault-mcp would
# lose write access to it
def save_movie_watch_history(data):
    tmp_path = f"{MOVIE_WATCH_HISTORY_PATH}.tmp"
    with open(tmp_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    os.replace(tmp_path, MOVIE_WATCH_HISTORY_PATH)
    chgrp_writable(MOVIE_WATCH_HISTORY_PATH)


# mirrors server.py's own _watch_history_text -- same short plain-text shape,
# fed to decide_movie_status below so the reasoning model sees the exact same
# view of the watch history the movies agent itself would
def movie_watch_history_text(data):
    lines = []
    if data["completed"]:
        parts = []
        for item in data["completed"]:
            title = item.get("title", "?")
            if item.get("liked") is True:
                parts.append(f"{title} (liked)")
            elif item.get("liked") is False:
                parts.append(f"{title} (didn't like)")
            else:
                parts.append(title)
        lines.append("COMPLETED: " + "; ".join(parts))
    if data["on_hold"]:
        parts = []
        for item in data["on_hold"]:
            bit = item.get("title", "?")
            extra = [item[k] for k in ("stopped_at", "reason") if item.get(k)]
            if extra:
                bit += f" ({', '.join(extra)})"
            parts.append(bit)
        lines.append("ON HOLD: " + "; ".join(parts))
    if data["suggested_not_yet_watched"]:
        lines.append("SUGGESTED: " + "; ".join(item.get("title", "?") for item in data["suggested_not_yet_watched"]))
    if data["wishlist"]:
        lines.append("WISHLIST: " + "; ".join(item.get("title", "?") for item in data["wishlist"]))
    if data["anticipated"]:
        lines.append("ANTICIPATED: " + "; ".join(item.get("title", "?") for item in data["anticipated"]))
    return "\n".join(lines) if lines else "(empty -- nothing recorded yet)"


def remove_from_movie_progress_lists(data, title_norm):
    for key in MOVIE_PROGRESS_LISTS:
        data[key] = [item for item in data[key]
                     if normalize_movie_title(item.get("title", "")).lower() != title_norm.lower()]


# the actual dedup-reasoning call -- see the MOVIE-topic header comment for
# why this goes to REASONING_MODEL ("auto") instead of the fast TAG_MODEL.
# fails safe: any parse failure or call failure below comes back as
# ALREADY_LISTED, never NEW, so a broken reasoning call just means a missed
# add (re-sendable) rather than a wrongly-added title.
def decide_movie_status(title, watch_history_text):
    prompt = (
        "You are checking Eldar's personal movie/show watch-history database for one title.\n\n"
        f"CURRENT WATCH HISTORY:\n{watch_history_text}\n\n"
        f"CANDIDATE TITLE (just arrived by email -- possibly in a different language, or a "
        f"stylized/translated form of something already listed): {title}\n\n"
        "Decide whether this candidate is something Eldar has ALREADY dealt with in some form "
        "(watched, on hold, already suggested, already wishlisted, or already anticipated -- "
        "including under a differently-worded or translated title, e.g. \"Skyfall\" and "
        "\"007: Координаты «Скайфолл»\" are the SAME film), or whether it's genuinely NEW.\n\n"
        "Reply in EXACTLY this format, one field per line, nothing else:\n"
        "DECISION: NEW, ALREADY_SEEN, or ALREADY_LISTED\n"
        "MATCHED_TITLE: <the existing entry it matches, exactly as listed above -- or none>\n"
        "REASON: <one short sentence>"
    )
    # num_predict bumped from 150 to 2000, 2026-09-30: confirmed live this
    # was the actual failure the whole time, not a network/parse problem --
    # "auto" (see litellm.yaml's own comment on ROUTE_TO_MODEL) routes
    # through a Qwen3 chat_template_kwargs.enable_thinking toggle, and
    # _llm_generate's <think>...</think> stripping (pre-existing, not new)
    # is proof this stack can return a whole reasoning block inline in
    # content. 150 tokens is enough for the model to open a <think> block,
    # reason a little, and close it -- and then it's already out of budget
    # before writing a single word of the actual DECISION: answer, so what's
    # left after stripping the think block is empty. No error anywhere:
    # HTTP 200, valid JSON, "content" is just genuinely empty. 2000 mirrors
    # ocr_image's own budget elsewhere in this file -- enough room for a
    # real think block AND the three-line answer after it.
    raw = _llm_generate(prompt, num_predict=2000, model=REASONING_MODEL, timeout=90, label="movie dedup reasoning call")

    result = {"decision": None, "matched_title": None, "reason": None}
    for line in raw.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().upper()
        value = value.strip()
        if key == "DECISION" and value.upper() in ("NEW", "ALREADY_SEEN", "ALREADY_LISTED"):
            result["decision"] = value.upper()
        elif key == "MATCHED_TITLE":
            result["matched_title"] = None if value.lower() == "none" else value
        elif key == "REASON":
            result["reason"] = value

    if result["decision"] is None:
        result["decision"] = "ALREADY_LISTED"
        result["reason"] = result["reason"] or "reasoning call failed or returned an unparseable reply -- not writing"
        # _llm_generate already prints WHY on a network/HTTP failure (see its
        # own label there) -- an empty raw here means that already happened
        # and there's nothing more to add. But a non-empty raw that still
        # didn't parse is a different, silent failure mode (confirmed live
        # 2026-09-30: no HTTP/exception error at all, so the call succeeded,
        # the model just didn't follow the DECISION:/MATCHED_TITLE:/REASON:
        # format) -- print what it actually said, the only way to tell the
        # two apart and see what needs fixing (prompt, model, parsing).
        if raw:
            print(f"    ⚠ movie dedup reasoning call returned unparseable content: {raw[:300]!r}")
    return result


# candidate title cascades: subject (free via the topic-prefix strip) -> body
# (LLM extraction) -> image (vision recognition) -- first one that produces
# something usable wins. Returns True if a title was actually added.
# handles ONE already-extracted candidate title end to end: normalize,
# dedup-reason against the current watch history, write if NEW. Split out
# of process_movie_email 2026-09-30 so the image branch below can call it
# once per attachment instead of stopping at the first hit (see that
# branch's own comment). Returns True if it actually wrote something.
def _handle_movie_candidate(candidate, source, dry_run):
    title_norm = normalize_movie_title(candidate)
    print(f"  🎬 candidate title (from {source}): {title_norm!r}")

    if dry_run:
        print("    (dry-run) would ask the reasoning model whether this is new")
        return False

    watch_history = load_movie_watch_history()
    decision = decide_movie_status(title_norm, movie_watch_history_text(watch_history))

    if decision["decision"] != "NEW":
        matched = f" ({decision['matched_title']})" if decision.get("matched_title") else ""
        print(f"    already have it -- {decision['decision'].lower()}{matched}: {decision['reason']}")
        return False

    remove_from_movie_progress_lists(watch_history, title_norm)
    watch_history["suggested_not_yet_watched"].append({
        "title": title_norm,
        "note": f"added via Gmail movie email, {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
    })
    save_movie_watch_history(watch_history)
    print(f"    ✓ added to suggested_not_yet_watched: {title_norm}  ({decision['reason']})")
    return True


# candidate title cascades: subject (free via the topic-prefix strip) -> body
# (LLM extraction) -> image (vision recognition). Returns the number of
# titles actually written (0 if none) -- NOT a bool, see the image branch
# below for why: sync() adds this straight onto movies_added.
def process_movie_email(subject, body, images, dry_run):
    bare_subject = re.fullmatch(r"MOVIE\s*[-–—:]?\s*", subject.strip(), re.IGNORECASE)

    candidate = ""
    source = ""
    if not bare_subject:
        candidate = re.sub(r"^MOVIE\s*[-–—:]\s*", "", subject.strip(), flags=re.IGNORECASE).strip()
        candidate = re.sub(r"^MOVIE\s+", "", candidate, flags=re.IGNORECASE).strip()
        source = "subject"

    # subject gave a candidate -- that's exactly ONE title for this email.
    if candidate:
        if candidate.upper() == "UNKNOWN":
            print(f"  ⚠ could not determine a movie/show title -- skipping: {subject[:60]}")
            return 0
        return 1 if _handle_movie_candidate(candidate, source, dry_run) else 0

    if body.strip():
        if dry_run:
            print("    (dry-run) would extract movie title(s) from the email body")
            return 0
        # body can list SEVERAL titles (confirmed live 2026-09-30, see
        # extract_movie_titles_from_text's own comment) -- same "every one
        # gets its own dedup+write, not just the first" treatment as the
        # image branch below, including within-email dedup. If nothing
        # comes back, fall through to images rather than giving up --
        # unchanged from the original single-title behavior.
        body_titles = extract_movie_titles_from_text(body)
        if body_titles:
            seen_this_email = set()
            added = 0
            for body_title in body_titles:
                title_norm = normalize_movie_title(body_title)
                if title_norm.lower() in seen_this_email:
                    continue
                seen_this_email.add(title_norm.lower())
                if _handle_movie_candidate(body_title, "body", dry_run):
                    added += 1
            return added
        print("  ⚠ could not identify any movie/show titles in the email body")

    if not images:
        print(f"  ⚠ could not determine a movie/show title -- skipping: {subject[:60]}")
        return 0

    if dry_run:
        print(f"    (dry-run) would recognize a movie title from {len(images)} image attachment(s)")
        return 0

    # No text candidate, so every image is a SEPARATE poster/screenshot,
    # not alternate photos of the same one -- confirmed live 2026-09-30
    # ("what if i send few pictures?"). The original version looped only
    # to find one usable title and broke on the first hit, silently
    # dropping every image after it -- a real email with 3 different movie
    # posters in one bare-subject email would have added only 1. Now every
    # image gets recognized and run through the full dedup+write pipeline
    # on its own, so N posters can add up to N titles. seen_this_email
    # still dedupes within the SAME email (two photos of the same poster,
    # or the model recognizing the same title from two attachments) --
    # decide_movie_status's own cross-email dedup handles everything
    # already in the watch history, this only guards against this one
    # email writing the same title twice.
    seen_this_email = set()
    added = 0
    for filename, mime_type, data in images:
        recognized = recognize_movie_title(data, mime_type)
        if not recognized or recognized.upper() == "UNKNOWN":
            print(f"  ⚠ could not identify a movie/show title in image: {filename}")
            continue
        title_norm = normalize_movie_title(recognized)
        if title_norm.lower() in seen_this_email:
            continue
        seen_this_email.add(title_norm.lower())
        if _handle_movie_candidate(recognized, f"image ({filename})", dry_run):
            added += 1
    return added


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
# strips tags out of an HTML email body down to plain text -- good enough for
# a note body, not trying to be a real HTML renderer
def html_to_text(html_body):
    import html as html_module
    text = re.sub(r"(?is)<(script|style).*?</\1>", "", html_body)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html_module.unescape(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def get_body(msg):
    payload = msg.get("payload", {})

    # decodes one part's base64url body data, for whichever mime type the
    # caller already confirmed it wants
    def _decode(part):
        data = part.get("body", {}).get("data", "")
        if not data:
            return ""
        import base64
        return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")

    # most emails are multipart/alternative with a text/plain part alongside
    # the text/html one -- prefer plain when it's there, fall back to html
    # (stripped to text) since Gmail's default rich-text compose sends
    # html-only, no plain-text part at all
    def _find(part, mime_wanted):
        if part.get("mimeType", "") == mime_wanted:
            return _decode(part)
        if part.get("mimeType", "").startswith("multipart/"):
            for sub in part.get("parts", []):
                found = _find(sub, mime_wanted)
                if found:
                    return found
        return ""

    plain = _find(payload, "text/plain")
    if plain:
        return plain.strip()

    html_body = _find(payload, "text/html")
    if html_body:
        return html_to_text(html_body)

    return ""


# Walks a message's MIME parts collecting image attachments (inline or
# regular). Gmail's API only inlines attachment bytes directly for very
# small parts (body.data) -- anything else needs a second call keyed by
# body.attachmentId.
# returns a list of (filename, mime_type, raw_bytes) tuples, one per image
def get_image_attachments(service, msg_id, payload):
    import base64
    images = []

    def _walk(part):
        filename = part.get("filename", "")
        mime     = part.get("mimeType", "")
        if filename and mime.startswith("image/"):
            body = part.get("body", {})
            if body.get("data"):
                data = base64.urlsafe_b64decode(body["data"] + "==")
            elif body.get("attachmentId"):
                att = service.users().messages().attachments().get(
                    userId="me", messageId=msg_id, id=body["attachmentId"]
                ).execute()
                data = base64.urlsafe_b64decode(att["data"] + "==")
            else:
                data = None
            if data:
                images.append((filename, mime, data))
        for sub in part.get("parts", []):
            _walk(sub)

    _walk(payload)
    return images


# Turns arbitrary text into a filesystem-safe slug (used for both note
# filenames -- via title_to_filename below -- and OCR'd attachment
# filenames): punctuation stripped, spaces/underscores collapsed to single
# hyphens, capped at max_bytes.
#
# Capped by BYTES, not characters: Python's \w (and this whole function)
# treats Cyrillic/CJK/etc. letters as ordinary word characters, so they
# pass straight through uncut -- and every one of those is 2-3 bytes in
# UTF-8, not 1. A 60-CHARACTER cap on a Cyrillic title (real case, Eldar's
# own vault, 2026-09-14) produced a ~110-BYTE filename that was unreadable
# in `ls` and got nowhere near actually protecting the filesystem's real
# (~255-byte) limit either -- the thing that actually needs capping is
# bytes, so that's what this caps, backing off to the nearest character
# boundary rather than splitting one in half.
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


# Saves each image into vault_root/Attachments (same top-level folder
# vault-mcp's ingest_file uses, regardless of which topic folder the note
# itself lands in -- so Health/Legal/etc. notes and Hermes's own uploads
# never collide on filenames across two different Attachments dirs) and
# OCRs it via ocr_image. Named after its OWN transcribed content (e.g.
# "dr-suska-business-card.jpg"), not the meaningless IMG_20260904_105815.jpg
# a phone camera gives it -- falls back to the original filename's base
# only when OCR found nothing to name it after. Returns a markdown block --
# an Obsidian embed link per image followed by its transcribed text --
# ready to append to the note body; images whose OCR found nothing still
# get the embed link alone.
def format_attachments(vault_root, images):
    attachments_dir = vault_root / "Attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)
    chgrp_writable(attachments_dir)

    blocks = []
    for filename, mime_type, data in images:
        text = ocr_image(data, mime_type)

        ext = Path(filename).suffix or ""
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


def title_to_filename(title, date_str=""):
    safe = slugify(title)

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
# topic word maps to) for any new topic-prefixed message -- except MOVIE,
# which updates watch-history.yaml instead of writing a note, see
# process_movie_email -- and records synced IDs. Returns the number of
# notes created (movies_added is logged separately, see the final print).
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

    # Marks a message done and, outside dry-run, writes the state file right
    # away rather than batching one save for the end of the whole loop --
    # this is the only thing standing between a run that gets killed partway
    # (n8n's task-runner timeout, a pod restart) and having to redo every
    # message from scratch next time instead of just the ones still left.
    # The file is tiny (a few hundred IDs at most) so saving every message
    # costs nothing.
    def mark_synced(msg_id):
        synced_ids.add(msg_id)
        if not dry_run:
            save_state(synced_ids)

    created      = 0
    movies_added = 0

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
            mark_synced(msg_id)
            continue

        sender = decode_mime_header(headers.get("From", "Unknown"))
        date   = headers.get("Date", "")
        body   = get_body(msg)
        images = get_image_attachments(service, msg_id, msg.get("payload", {}))

        # MOVIE topic doesn't write a vault note at all -- see header
        # comment -- so it branches off here, before the generic "no body,
        # no image" skip below and before any of the note/filename logic
        # further down, and is done with this message either way once
        # process_movie_email returns. Deliberately ahead of that generic
        # skip (not after it, as this was first written) -- unlike every
        # note-writing topic, a bare "Movie: <title>" subject with no body
        # and no attachment is the NORMAL case here, the title comes free
        # from the subject alone (see process_movie_email), so that check
        # would otherwise kill the most common movie email before it ever
        # reached this branch. Confirmed happening in practice 2026-09-30.
        if topic == "MOVIE":
            # process_movie_email returns a COUNT now, not a bool -- one
            # email can add more than one title (multiple image
            # attachments, see that function's own comment).
            movies_added += process_movie_email(subject, body, images, dry_run)
            mark_synced(msg_id)
            continue

        if not body and not images:
            print(f"  ⚠ no body, no image attachments -- skipping: {subject[:60]}")
            mark_synced(msg_id)
            continue

        # clean_title only reads body content for its fallback -- a subject
        # that's the bare topic word on its own, nothing else -- so for
        # every other subject (the vast majority) this pre-OCR title is
        # already the FINAL title, and checking existence with it now, before
        # OCR runs, is what stops a rescan of an already-synced message from
        # wastefully re-OCR'ing its image and writing a second,
        # differently-named copy into Attachments/ that nothing ends up
        # using. Confirmed happening in practice on 2026-09-14: three
        # duplicate attachment files got created this way on a state-file-
        # loss rescan before this check existed.
        title      = clean_title(subject, body, topic, dry_run=dry_run)
        filename   = title_to_filename(title, date) + ".md"
        target_dir = vault_root / folder
        note_path  = target_dir / filename

        # FIXED 2026-09-28: this used to be `if not dry_run and
        # note_path.exists()`, which meant a --dry-run preview NEVER checked
        # whether the note already existed at all -- it always fell through
        # to "(dry-run) would create", even for a message that had already
        # been synced and written for real. Found live: a --dry-run run
        # confidently claimed it "would create" two notes that had already
        # existed in the vault (and in Qdrant) for days, which looked
        # exactly like a real missing-sync bug and cost a long investigation
        # before the dry-run report itself turned out to be the thing lying.
        # Real (non-dry-run) mode was never affected -- its own skip+
        # mark_synced+continue behavior is unchanged below, just no longer
        # gated behind `not dry_run` for the existence check itself.
        if note_path.exists():
            if dry_run:
                print(f"  (dry-run) already exists, would skip: {folder}/{filename}")
            else:
                print(f"  ⚠ already exists (skipping): {folder}/{filename}")
                mark_synced(msg_id)
            continue

        # OCR'd attachment text becomes the body when the email itself has
        # none (a photo with a bare subject line, e.g.) -- appended after it
        # otherwise. Skipped entirely in dry-run: OCR costs a real LLM call
        # per image, not worth spending on a preview that writes nothing.
        if images and not dry_run:
            attachments_md = format_attachments(vault_root, images)
            body = f"{body}\n\n{attachments_md}" if body else attachments_md
            # The title above was computed from body text alone -- for the
            # bare-topic-word-subject case (Eldar's own Health test email:
            # subject was just "Health", nothing else) that meant an empty
            # fallback, since the body's first line didn't exist until OCR
            # just ran. Recompute and re-check now that it does, rather
            # than write a note titled after nothing.
            title      = clean_title(subject, body, topic, dry_run=dry_run)
            filename   = title_to_filename(title, date) + ".md"
            note_path  = target_dir / filename
            if note_path.exists():
                print(f"  ⚠ already exists (skipping): {folder}/{filename}")
                mark_synced(msg_id)
                continue
        elif images:
            names = ", ".join(name for name, _, _ in images)
            print(f"    (dry-run) would OCR {len(images)} image attachment(s): {names}")

        tag_source = body or title
        tags = [] if dry_run else extract_tags(title + "\n\n" + tag_source[:800], folder)

        content = format_note(title, subject, sender, date, body, tags)

        if dry_run:
            print(f"  (dry-run) would create: {folder}/{filename}")
            print(f"    tags: {tags}")
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            chgrp_writable(target_dir)
            note_path.write_text(content, encoding="utf-8")
            chgrp_writable(note_path)   # otherwise this (root, in the N8N pod) write leaves the note root:root -- Eldar/Hermes can't edit or delete it later
            print(f"  ✓ created: {folder}/{filename}  tags={tags}")
            created += 1

        mark_synced(msg_id)

    print(f"[gmail] done -- {created} new notes created, {movies_added} movie(s) added to suggested_not_yet_watched")
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
