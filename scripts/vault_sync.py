# vault_sync.py -- syncs the Obsidian vault into aibox's Qdrant
# "obsidian_vault" collection, so Hermes (once a RAG tool exists) can search it.
#
# Ported from the old Windows-lab pipeline (ai-lab/step10_n8n/vault_sync.py),
# with three changes: embeddings go through LiteLLM's nomic-embed-text-v1.5
# instead of a local sentence-transformers model, Qdrant talk uses the plain
# qdrant-client instead of langchain, and -- the actual point of this pass --
# each note's `tags`/`audience` YAML frontmatter is now parsed and carried
# into every chunk's Qdrant payload, which the old lab's pipeline never did
# (see the vault README's "KNOWN GAP" note).
#
# Standalone for now, run by hand or via cron -- no N8N wiring yet, that's
# the next step once this is proven working.
#
# Usage (from the repo root, on aibox):
#   pip install qdrant-client aiohttp pyyaml
#   export LITELLM_MASTER_KEY=$(kubectl get secret -n ai-models litellm-secret -o jsonpath='{.data.master_key}' | base64 -d)
#   python3 scripts/vault_sync.py             # incremental -- only new/changed notes
#   python3 scripts/vault_sync.py --resync    # re-ingest every note, regardless of mtime
#   python3 scripts/vault_sync.py --dry-run   # print what would happen, write nothing

import argparse
import asyncio
import hashlib
import json
import os
import re
import uuid
from pathlib import Path

import aiohttp
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

VAULT_PATH = Path("/data/obsidian_vault")
STATE_FILE = VAULT_PATH / ".vault_sync_state.json"   # hidden dotfile -- not picked up by the *.md scan below

# localhost+NodePort is only reachable when this script runs directly on
# the aibox host (its original design, a bare scripts/.venv process) --
# both env-var overridable so the SAME script also works run in-cluster
# (e.g. from the N8N Execute Command node, apps/n8n/Dockerfile), where
# "localhost" is the container's own netns, not the node's, and has to be
# pointed at the real in-cluster Service DNS instead (same convention
# vault-mcp/media-gen-mcp/hermes-agent already use for these two).
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:30633")
COLLECTION_NAME = "obsidian_vault"

LITELLM_URL = os.environ.get("LITELLM_URL", "http://localhost:30400")
EMBED_MODEL = "nomic-embed-text-v1.5"
LITELLM_MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY")

# Same block-splitting logic as the old lab's markdown_blockify.py -- ported
# inline since this vault only ever has real .md files, so that file's
# MarkItDown multi-format conversion step isn't needed here.
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
FENCE_RE = re.compile(r"^(```|~~~)")
TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


# Splits a frontmatter block off the top of a note, e.g.:
#   ---
#   tags: [legal, poland]
#   audience: everyone
#   ---
#   # rest of the note...
# returns (frontmatter_dict, remaining_markdown_text) -- frontmatter_dict is
# {} if the note has no frontmatter block.
def split_frontmatter(text):
    if not text.startswith("---"):
        return {}, text
    match = re.match(r"^---\s*\n(.*?\n)---\s*\n?", text, re.DOTALL)
    if not match:
        return {}, text
    try:
        front = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:   # malformed frontmatter -- treat the note as untagged rather than crash
        front = {}
    return front, text[match.end():]


# Real ceiling confirmed on aibox, 2026-09-14: nomic-embed-text-v1.5's
# actual GGUF here caps out at n_ctx=2048 regardless of any llama-server
# -c/--parallel setting -- confirmed via its own /props endpoint after
# trying both, so this is the model's real native context, not a config
# bug to tune around. A single big unbroken block (a long pasted
# blockquote, e.g. -- the real case that surfaced this: 10957 chars/4614
# tokens in one chunk, silently rejected) can blow past that in one
# embed call. MAX_CHUNK_CHARS is a conservative char-based stand-in for a
# token count -- the failing chunk measured ~2.4 chars/token, denser than
# the usual ~4 chars/token rule of thumb for English, so this is picked
# with real margin below 2048 tokens either way, not tuned to that one
# sample's exact ratio.
MAX_CHUNK_CHARS = 3000


# Splits one oversized block's text into pieces each under
# MAX_CHUNK_CHARS -- prefers breaking on a blank line, then any line
# break, before falling back to a hard cut, so pieces stay reasonably
# readable instead of splitting mid-sentence when avoidable.
def _split_oversized(text):
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    pieces = []
    remaining = text
    while len(remaining) > MAX_CHUNK_CHARS:
        cut = remaining.rfind("\n\n", 0, MAX_CHUNK_CHARS)
        if cut == -1:
            cut = remaining.rfind("\n", 0, MAX_CHUNK_CHARS)
        if cut == -1:
            cut = MAX_CHUNK_CHARS
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _flush_block(blocks, lines, block_type, headings):
    text = "\n".join(lines).strip()
    if text:
        for piece in _split_oversized(text):
            blocks.append({"text": piece, "chunk_type": block_type, "heading_path": " > ".join(headings)})
    lines.clear()


# Splits markdown into headings/paragraphs/lists/tables/code blocks, each
# tagged with the heading breadcrumb above it -- same chunker the old lab
# used, so retrieval quality carries over.
def blockify(markdown):
    blocks = []
    current_lines = []
    current_type = "paragraph"
    headings = []
    in_code = False

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if FENCE_RE.match(stripped):
            if in_code:
                current_lines.append(line)
                _flush_block(blocks, current_lines, "code", headings)
                current_type = "paragraph"
                in_code = False
            else:
                _flush_block(blocks, current_lines, current_type, headings)
                current_lines.append(line)
                current_type = "code"
                in_code = True
            continue

        if in_code:
            current_lines.append(line)
            continue

        heading_match = HEADING_RE.match(stripped)
        if heading_match:
            _flush_block(blocks, current_lines, current_type, headings)
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            headings = headings[: level - 1] + [title]
            blocks.append({"text": stripped, "chunk_type": "heading", "heading_path": " > ".join(headings)})
            current_type = "paragraph"
            continue

        if not stripped:
            _flush_block(blocks, current_lines, current_type, headings)
            current_type = "paragraph"
            continue

        line_type = "table" if TABLE_RE.match(line) else "list" if LIST_RE.match(line) else "paragraph"
        if current_lines and line_type != current_type:
            _flush_block(blocks, current_lines, current_type, headings)
        current_type = line_type
        current_lines.append(line)

    _flush_block(blocks, current_lines, current_type, headings)
    return blocks


def chunk_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# nomic-embed-text-v1.5 is trained to expect a "search_document: " prefix on
# anything being INDEXED (as opposed to "search_query: " on the text a
# search is later run with, see link_notes.py's embed_query) -- without it,
# the model still returns vectors, but they cluster much closer together
# regardless of actual topic, which is exactly what made link_notes.py's
# similarity scores come back nearly flat across unrelated notes (confirmed
# 2026-09-14: e.g. 14 completely different AI notes all scoring within
# 0.007 of each other against one query). Added here on the indexing side
# and in link_notes.py/vault-mcp's server.py on the query side, all three
# at once -- a document embedded with the prefix only lines up meaningfully
# against a query embedded with ITS prefix, so this only helps once every
# embedding call into this collection uses the matching one. Existing
# vectors were embedded without any prefix; run `vault_sync.py --resync`
# after deploying this to re-embed the whole vault, or old and new vectors
# will sit inconsistently in the same collection.
#
# sends this to LiteLLM's OpenAI-compatible embeddings endpoint:
#   {"model": "nomic-embed-text-v1.5", "input": ["search_document: chunk one text", "search_document: chunk two text"]}
# returns a list of vectors in the same order as texts, e.g.
#   [[0.012, -0.045, ...], [0.031, 0.002, ...]]
async def embed_texts(texts):
    headers = {"Authorization": f"Bearer {LITELLM_MASTER_KEY}"}
    payload = {"model": EMBED_MODEL, "input": [f"search_document: {t}" for t in texts]}
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{LITELLM_URL}/v1/embeddings", json=payload, headers=headers,
                                 timeout=aiohttp.ClientTimeout(total=60)) as resp:
            resp.raise_for_status()
            result = await resp.json()
    ordered = sorted(result["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in ordered]


def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:   # corrupt state file -- start fresh rather than crash
            return {}
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# Creates the collection on first run. vector_size comes from a real
# embedding call rather than being hardcoded, so this can't drift out of
# sync with whatever nomic-embed-text-v1.5 actually returns.
def ensure_collection(client, vector_size):
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"[qdrant] created collection '{COLLECTION_NAME}' (size={vector_size})")


# Removes every existing point for this note before re-adding its current
# chunks -- otherwise an edited note (fewer chunks, or changed text) leaves
# its old, now-wrong chunks sitting in Qdrant forever, since each chunk's
# point id is derived from its own text hash (see ingest_note below), so an
# edited chunk gets a brand new id rather than overwriting the old one.
def delete_notes_points(client, source):
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]),
    )


# Converts one note into Qdrant points and upserts them.
# Returns the number of chunks written (0 if the note had nothing to ingest).
async def ingest_note(client, path, dry_run=False):
    text = path.read_text(encoding="utf-8")
    frontmatter, body = split_frontmatter(text)
    tags = frontmatter.get("tags") or []
    audience = frontmatter.get("audience")   # None if not set in the note -- not assumed to mean "everyone"

    blocks = blockify(body)
    if not blocks:
        print(f"  (skip) no content in {path.name}")
        return 0

    source = str(path.relative_to(VAULT_PATH)).replace("\\", "/")
    note_title = path.stem

    if dry_run:
        print(f"  (dry-run) would ingest {len(blocks)} chunks from {source}  tags={tags} audience={audience}")
        return len(blocks)

    vectors = await embed_texts([b["text"] for b in blocks])

    points = []
    for block, vector in zip(blocks, vectors):
        h = chunk_hash(block["text"])
        points.append(PointStruct(
            # keyed by source+text hash, not text hash alone -- two different notes that happen to
            # share identical chunk text (a common heading, a repeated one-liner) used to collide onto
            # the same point id and silently overwrite each other, quietly dropping one note's chunk.
            # still deterministic per note (re-ingesting the same unchanged chunk in the same note
            # reuses the same id, which is all delete_notes_points below actually needs).
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}::{h}")),
            vector=vector,
            payload={
                "text": block["text"],
                "chunk_type": block["chunk_type"],
                "heading_path": block["heading_path"],
                "source": source,
                "note_title": note_title,
                "tags": tags,
                "audience": audience,
                "chunk_hash": h,
            },
        ))

    delete_notes_points(client, source)
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"  {len(points)} chunks  <-  {source}  tags={tags} audience={audience}")
    return len(points)


async def run_sync(resync, dry_run):
    notes = sorted(VAULT_PATH.rglob("*.md"))
    if not notes:
        print(f"No .md files found under {VAULT_PATH}")
        return

    state = load_state()
    if resync:
        to_process = [(n, n.stat().st_mtime) for n in notes]
        print(f"[resync] re-ingesting all {len(notes)} notes")
    else:
        to_process = [(n, n.stat().st_mtime) for n in notes if state.get(str(n)) != n.stat().st_mtime]
        print(f"[sync] {len(to_process)} new/changed notes out of {len(notes)} total")

    if not to_process:
        print("nothing to do")
        return

    client = QdrantClient(url=QDRANT_URL)
    if not dry_run:
        # embed one throwaway string just to learn the vector size, so the
        # collection is created with the right dimensions before anything upserts
        vector_size = len((await embed_texts(["test"]))[0])
        ensure_collection(client, vector_size)

    total_chunks = 0
    for note, mtime in to_process:
        try:
            n = await ingest_note(client, note, dry_run)
        except Exception as exc:   # one bad note shouldn't stop the whole sync
            print(f"  FAILED on {note.name}: {exc}")
            continue
        total_chunks += n
        if not dry_run:
            state[str(note)] = mtime

    if not dry_run:
        save_state(state)
    print(f"done -- {total_chunks} chunks across {len(to_process)} notes")


# runs one sync pass over the vault (incremental by default, --resync for a
# full re-ingest, --dry-run to preview without writing) and exits
if __name__ == "__main__":
    if not LITELLM_MASTER_KEY:
        raise SystemExit(
            "Set LITELLM_MASTER_KEY first, e.g.:\n"
            "  export LITELLM_MASTER_KEY=$(kubectl get secret -n ai-models litellm-secret "
            "-o jsonpath='{.data.master_key}' | base64 -d)"
        )

    parser = argparse.ArgumentParser(description="Sync the Obsidian vault into aibox's Qdrant obsidian_vault collection.")
    parser.add_argument("--resync", action="store_true", help="re-ingest every note, ignoring last-synced state")
    parser.add_argument("--dry-run", action="store_true", help="print what would happen, write nothing")
    args = parser.parse_args()

    asyncio.run(run_sync(args.resync, args.dry_run))
