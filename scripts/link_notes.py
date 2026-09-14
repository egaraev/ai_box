# link_notes.py -- adds [[wiki-links]] between related Obsidian notes using
# the same "obsidian_vault" Qdrant collection vault_sync.py already fills.
#
# Ported from the old Windows-lab pipeline (ai-lab/step10_n8n/link_notes.py),
# with two changes: embeddings go through LiteLLM's nomic-embed-text-v1.5
# instead of a local HuggingFace model (same swap vault_sync.py already made),
# and the Qdrant payload is read flat (note_title, source, ...) instead of
# nested under "metadata" -- that nesting was langchain's QdrantVectorStore
# doing its own thing; the current pipeline writes flat payloads directly
# (see vault_sync.py's ingest_note).
#
# For each .md note in the scanned folder (AI by default), searches Qdrant
# for semantically similar notes and appends/refreshes a "## Related Notes"
# section with [[filename]] links -- run repeatedly, it only touches notes
# that are new or have changed since their last run (tracked in
# .link_notes_state.json, same convention as vault_sync.py's own state file).
#
# Usage (from the repo root, on aibox):
#   export LITELLM_MASTER_KEY=$(kubectl get secret -n ai-models litellm-secret -o jsonpath='{.data.master_key}' | base64 -d)
#   python3 scripts/link_notes.py                  # AI/ folder only (default)
#   python3 scripts/link_notes.py --folder Health   # a different single folder
#   python3 scripts/link_notes.py --all             # entire vault
#   python3 scripts/link_notes.py --dry-run         # preview, write nothing
#   python3 scripts/link_notes.py --reset           # ignore state, relink everything in scope
#   python3 scripts/link_notes.py --top-k 7         # more links per note (default 5)
#   python3 scripts/link_notes.py --min-score 0.75  # stricter similarity (default 0.70)
#
# NOTE: --min-score 0.70 is carried over from the old HuggingFace-embedding
# pipeline -- nomic-embed-text-v1.5's cosine similarity scores may cluster
# differently. Run --dry-run first on a folder you know well and eyeball
# whether 0.70 is finding real connections or missing/over-linking before
# trusting a full non-dry-run pass.

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

import aiohttp
from qdrant_client import QdrantClient

VAULT_PATH = Path("/data/obsidian_vault")
STATE_FILE = VAULT_PATH / ".link_notes_state.json"   # hidden dotfile -- not picked up by the *.md scan below
COLLECTION_NAME = "obsidian_vault"

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:30633")
LITELLM_URL = os.environ.get("LITELLM_URL", "http://localhost:30400")
EMBED_MODEL = "nomic-embed-text-v1.5"
LITELLM_MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY")

DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.70   # cosine similarity threshold (0-1) -- see NOTE above
EMBED_CHARS = 1500         # how many chars of the note to embed for the similarity search

# Same gid/convention as gmail_sync.py's chgrp_writable (kept as its own
# small copy here rather than a shared import -- see gmail_sync.py's own
# comment on why: every script in this repo stays self-contained). Without
# this, a note this script edits goes back to root:root the moment it's
# rewritten, undoing the fix already applied to the rest of the vault.
HERMES_GID = 10000

def chgrp_writable(path, file_mode=0o664):
    os.chown(path, -1, HERMES_GID)   # -1 leaves the owner unchanged, only the group changes
    path.chmod(file_mode)

RELATED_SECTION_RE = re.compile(r"\n\n## Related Notes\n(?:.*\n?)*?(?=\n##|\Z)", re.MULTILINE)


# Removes any existing "## Related Notes" section, so re-running this script
# doesn't keep appending duplicate sections and doesn't feed a note's own
# old links back into its next similarity search (self-reinforcing loop).
def strip_related_section(content):
    return RELATED_SECTION_RE.sub("", content).rstrip()


def build_related_section(titles):
    if not titles:
        return ""
    links = "\n".join(f"- [[{t}]]" for t in titles)
    return f"\n\n## Related Notes\n\n{links}"


def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:   # corrupt state file -- start fresh rather than crash
            return {}
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    chgrp_writable(STATE_FILE)


# nomic-embed-text-v1.5 wants "search_query: " on the QUERY side of a
# similarity search, matching "search_document: " on the indexing side
# (vault_sync.py's embed_texts) -- see that function's comment for why this
# matters. This note's own content is the "query" here (we're searching for
# notes similar to it), so it gets the query prefix, not the document one.
#
# sends this to LiteLLM's OpenAI-compatible embeddings endpoint:
#   {"model": "nomic-embed-text-v1.5", "input": ["search_query: note text up to EMBED_CHARS chars"]}
# returns a single vector, e.g. [0.012, -0.045, ...]
async def embed_query(text):
    headers = {"Authorization": f"Bearer {LITELLM_MASTER_KEY}"}
    payload = {"model": EMBED_MODEL, "input": [f"search_query: {text}"]}
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{LITELLM_URL}/v1/embeddings", json=payload, headers=headers,
                                 timeout=aiohttp.ClientTimeout(total=60)) as resp:
            resp.raise_for_status()
            result = await resp.json()
    return result["data"][0]["embedding"]


# Embeds the note and searches Qdrant for similar chunks from OTHER notes.
# Returns a list of unique (title, score) pairs (not including the note
# itself), ordered by descending similarity, restricted to notes that
# actually exist in candidate_titles (the same folder this note was scanned
# from -- see run_linker below for why). Deliberately returns everything
# found, not just what's above min_score -- run_linker applies that filter,
# so a dry-run can still show near-misses alongside what actually passed.
async def find_related(note_path, candidate_titles, client, top_k):
    content = note_path.read_text(encoding="utf-8")
    query_text = strip_related_section(content)[:EMBED_CHARS].strip()
    if not query_text:
        return []

    try:
        vector = await embed_query(query_text)
    except Exception as e:   # embedding call failed -- skip this note, don't crash the whole run
        print(f"  ✗ embed failed for {note_path.name}: {e}")
        return []

    try:
        # client.query_points() returns an object whose .points is a list of
        # ScoredPoint(score=0.83, payload={"note_title": "...", "source": "...", ...})
        # -- no score_threshold here (unlike the old ai-lab version): min_score
        # is applied below in Python instead, AFTER we've seen every
        # candidate's real score, so a dry-run can show what's just below the
        # cutoff too, not only what already passed it.
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=top_k * 4,   # fetch extra so duplicate note_titles (multiple chunks per note) can be deduplicated
            with_payload=True,
        )
        hits = results.points
    except Exception as e:   # Qdrant unreachable/misconfigured -- skip this note, don't crash the whole run
        print(f"  ✗ Qdrant search failed for {note_path.name}: {e}")
        return []

    self_title = note_path.stem
    seen = set()
    related = []   # list of (title, score), highest score first -- hits already come back sorted that way
    for hit in hits:
        title = hit.payload.get("note_title", "")
        if not title or title == self_title or title in seen:
            continue
        if title in candidate_titles:
            related.append((title, hit.score))
            seen.add(title)
        if len(related) >= top_k * 3:   # keep enough below-threshold candidates around for the dry-run printout
            break

    return related


# Finds related notes for everything new/changed among `notes` and rewrites
# each one's "## Related Notes" section. candidate_titles is the set of
# titles a link is allowed to point at -- kept equal to the scanned set
# (same folder scan, or the whole vault under --all) so this never links
# an AI note out to, say, a Health note nobody meant to cross-reference.
async def run_linker(notes, candidate_titles, top_k, min_score, dry_run, reset):
    state = {} if reset else load_state()

    to_process = [(n, n.stat().st_mtime) for n in notes if state.get(str(n)) != n.stat().st_mtime]
    print(f"[links] {len(notes) - len(to_process)} up to date, {len(to_process)} to process")
    if not to_process:
        print("[links] nothing to do")
        return

    client = QdrantClient(url=QDRANT_URL)
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        raise SystemExit(f"Qdrant collection '{COLLECTION_NAME}' not found -- run vault_sync.py first.")

    updated = 0
    for note_path, mtime in to_process:
        scored = await find_related(note_path, candidate_titles, client, top_k)
        passing = [(t, s) for t, s in scored if s >= min_score][:top_k]

        if dry_run:
            # shows every candidate found, not just the ones that passed --
            # the point of dry-run is judging whether min_score itself is
            # set right, which you can't do from only the survivors.
            if scored:
                print(f"  (dry-run) {note_path.name}")
                for t, s in scored:
                    mark = "✓" if s >= min_score else "✗ below threshold"
                    print(f"    {s:.3f} {mark}  → [[{t}]]")
            else:
                print(f"  (dry-run) {note_path.name} -- no candidates found at all")
            continue

        if not passing:
            # Qdrant found nothing above threshold this time -- keep whatever
            # links the note already has rather than erasing them.
            print(f"  – {note_path.name} (no links above threshold, keeping existing)")
            state[str(note_path)] = mtime
            continue

        titles = [t for t, s in passing]
        original = note_path.read_text(encoding="utf-8")
        new_content = strip_related_section(original) + build_related_section(titles) + "\n"

        if new_content != original:
            note_path.write_text(new_content, encoding="utf-8")
            chgrp_writable(note_path)
            score_range = f"{passing[-1][1]:.3f}-{passing[0][1]:.3f}"
            print(f"  ✓ {note_path.name} → {len(titles)} links (scores {score_range})")
            updated += 1
        else:
            print(f"  = {note_path.name} (unchanged)")

        # state keyed on the POST-write mtime, so the just-written change
        # doesn't make this note look "changed" again on the very next run
        state[str(note_path)] = note_path.stat().st_mtime

    if not dry_run:
        save_state(state)
        print(f"[links] done -- {updated} notes updated")
    else:
        print(f"[links] dry-run complete -- {len(to_process)} notes checked")


# runs one linking pass (AI/ folder by default, --folder/--all to change
# scope, --dry-run to preview, --reset to ignore state) and exits
if __name__ == "__main__":
    if not LITELLM_MASTER_KEY:
        raise SystemExit(
            "Set LITELLM_MASTER_KEY first, e.g.:\n"
            "  export LITELLM_MASTER_KEY=$(kubectl get secret -n ai-models litellm-secret "
            "-o jsonpath='{.data.master_key}' | base64 -d)"
        )

    parser = argparse.ArgumentParser(description="Add [[wiki-links]] between related Obsidian notes via Qdrant similarity search.")
    parser.add_argument("--folder", default="AI", help="vault subfolder to scan (default: AI)")
    parser.add_argument("--all", action="store_true", help="scan the entire vault instead of one folder")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help=f"max related notes per note (default: {DEFAULT_TOP_K})")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE, help=f"minimum cosine similarity 0-1 (default: {DEFAULT_MIN_SCORE})")
    parser.add_argument("--dry-run", action="store_true", help="preview links without writing files")
    parser.add_argument("--reset", action="store_true", help="ignore state file, relink every note in scope")
    args = parser.parse_args()

    if args.all:
        scan_notes = sorted(VAULT_PATH.rglob("*.md"))
        scope_label = "entire vault"
    else:
        scan_dir = VAULT_PATH / args.folder
        if not scan_dir.is_dir():
            raise SystemExit(f"Folder not found: {scan_dir}")
        scan_notes = sorted(scan_dir.glob("*.md"))
        scope_label = f"{args.folder}/"

    print(f"[links] {len(scan_notes)} notes in {scope_label}")
    candidate_titles = {n.stem for n in scan_notes}

    asyncio.run(run_linker(scan_notes, candidate_titles, args.top_k, args.min_score, args.dry_run, args.reset))
