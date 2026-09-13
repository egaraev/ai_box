"""vault-mcp: an HTTP MCP server exposing Obsidian-vault search and ingest to Hermes.

Same pattern as apps/media-gen-mcp/server.py (mcp>=2.0's MCPServer, HTTP
streamable-http transport on :8000/mcp, registered in Hermes's mcp_servers
config block -- see k8s/p3/hermes-agent.yaml). Two tools:

  search_vault  -- embed a query via LiteLLM, search Qdrant's obsidian_vault
                   collection, return the matching chunks as text.
  ingest_note   -- write text Eldar gives Hermes as a new vault note
                   (Inbox/<slug>.md), then chunk+embed+upsert it.
  ingest_file   -- same, but for a file already readable by this server
                   (image/PDF/other): extracts text (PDF) or generates a
                   caption (image) via the local vision model, copies the
                   original into Attachments/, links it from the note.

Chunking/embedding/point-id logic here is a deliberate DUPLICATE of
scripts/vault_sync.py's own copies, not a shared import -- that script runs
as a bare venv process on the host, this runs in a container, and there's
no clean way to share a literal module across that boundary without extra
plumbing (see vault_sync.py's own top comment). If you fix a bug in one
copy (e.g. the point-id collision bug fixed 2026-09-13, see
/areas/ai-box-k3s.md), fix it in the other too -- they must stay identical
in this specific regard or the two paths will silently diverge on how the
SAME vault ends up indexed.

Point ids are keyed off `source_path::text_hash` (not text hash alone) for
the same reason vault_sync.py's are: two different notes sharing identical
short chunk text (a repeated heading, a common one-liner) would otherwise
collide onto the same Qdrant point and silently overwrite each other.

ingest_file's file_path is checked against two allowed roots (CONFIRMED,
2026-09-13, by asking Hermes itself where it saves things -- not guessed):
  - /shared-output -- the hostPath media-gen-mcp already shares with
    hermes-agent for OUTGOING generated-image delivery.
  - /opt/data/cache -- the hermes-data PVC's cache/ subdirectory, where
    hermes-agent actually saves INCOMING Telegram attachments
    (cache/images/, cache/audio/). Mounted here read-only via subPath so
    this pod never sees the rest of that PVC (config.yaml/honcho.json/.env
    live at its root -- see k8s/p3/hermes-agent.yaml). Both pods run on
    this cluster's single node, so a second pod mounting the same PVC (even
    read-write at the API level, since ReadWriteOnce is node-scoped, not
    pod-scoped) is expected to work -- NOT YET CONFIRMED against a real
    pod start, check both pods reach Running after this deploys.

IMPORTANT, told directly to Eldar: don't let Hermes/Jarvis actually create
its own "/opt/data/obsidian_vault/" directory, even though it offered to in
chat. That path lives inside hermes-agent's own private PVC, entirely
separate from the real vault at /data/obsidian_vault -- anything saved
there would never be seen by scripts/vault_sync.py or by this server's own
search_vault, a silent duplicate "vault" that looks like it worked. The
real fix is this server's ingest_file tool reading from where Hermes
already saves things (above), not Hermes writing into a new location.
"""
import base64
import hashlib
import logging
import mimetypes
import os
import re
import shutil
import uuid
from pathlib import Path

import httpx
from mcp.server.mcpserver import MCPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s vault-mcp %(levelname)s: %(message)s")
logger = logging.getLogger("vault-mcp")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant.ai-memory.svc.cluster.local:6333")
COLLECTION_NAME = "obsidian_vault"

LITELLM_URL = os.environ.get("LITELLM_URL", "http://litellm.ai-models.svc.cluster.local:4000/v1")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")
EMBED_MODEL = "nomic-embed-text-v1.5"
VISION_MODEL = "qwen3.8-27b-Expert-Vision"   # same locked vision route Hermes's own image-in-chat path already uses (custom_router_hook.py's _has_image check)

VAULT_PATH = Path(os.environ.get("VAULT_PATH", "/data/obsidian_vault"))
INBOX_DIR = VAULT_PATH / "Inbox"             # everything this server writes lands here -- Eldar re-files it in Obsidian later if he wants
ATTACHMENTS_DIR = VAULT_PATH / "Attachments"

# Where ingest_file is allowed to read from -- see this file's top comment
# for how these two were confirmed. Anything outside both is refused rather
# than read, so a compromised/confused Hermes call can't point this server
# at an arbitrary host path.
ALLOWED_INPUT_ROOTS = [
    Path(os.environ.get("SHARED_INPUT_DIR", "/shared-output")),
    Path(os.environ.get("HERMES_CACHE_DIR", "/opt/data/cache")),
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

mcp = MCPServer("vault")

# ---- chunking -- identical to scripts/vault_sync.py's own copy of this
# (see this file's top comment for why it's duplicated, not imported) ----
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
FENCE_RE = re.compile(r"^(```|~~~)")
TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def _flush_block(blocks, lines, block_type, headings):
    text = "\n".join(lines).strip()
    if text:
        blocks.append({"text": text, "chunk_type": block_type, "heading_path": " > ".join(headings)})
    lines.clear()


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


def chunk_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.strip().lower()).strip("-")
    return slug or "untitled"


# sends this to LiteLLM's OpenAI-compatible embeddings endpoint:
#   {"model": "nomic-embed-text-v1.5", "input": ["chunk one text", "chunk two text"]}
# returns a list of vectors in the same order as texts
async def embed_texts(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    headers = {"Authorization": f"Bearer {LITELLM_API_KEY}"}
    resp = await client.post(
        f"{LITELLM_URL}/embeddings",
        json={"model": EMBED_MODEL, "input": texts},
        headers=headers,
        timeout=60.0,
    )
    resp.raise_for_status()
    result = resp.json()
    ordered = sorted(result["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in ordered]


# ---- Qdrant, plain REST -- no qdrant-client dependency, same reasoning as
# vault_sync.py dropping langchain: this container already pulls in httpx
# for LiteLLM, no reason to add a second HTTP-wrapping SDK just for Qdrant ----

async def qdrant_ensure_collection(client: httpx.AsyncClient, vector_size: int) -> None:
    resp = await client.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
    if resp.status_code == 200:
        return
    await client.put(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}",
        json={"vectors": {"size": vector_size, "distance": "Cosine"}},
        timeout=30.0,
    )
    logger.info("created Qdrant collection '%s' (size=%s)", COLLECTION_NAME, vector_size)


async def qdrant_delete_source(client: httpx.AsyncClient, source: str) -> None:
    await client.post(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/delete",
        json={"filter": {"must": [{"key": "source", "match": {"value": source}}]}},
        timeout=30.0,
    )


async def qdrant_upsert(client: httpx.AsyncClient, points: list[dict]) -> None:
    resp = await client.put(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
        json={"points": points},
        timeout=60.0,
    )
    resp.raise_for_status()


# sends this to Qdrant's search endpoint:
#   {"vector": [0.01, ...], "limit": 5, "with_payload": true, "filter": {...}}
# returns e.g. {"result": [{"id": "...", "score": 0.82, "payload": {...}}, ...]}
async def qdrant_search(client: httpx.AsyncClient, vector: list[float], top_k: int, tags) -> list[dict]:
    body = {"vector": vector, "limit": top_k, "with_payload": True}
    if tags:
        body["filter"] = {"must": [{"key": "tags", "match": {"any": tags}}]}
    resp = await client.post(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
        json=body,
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["result"]


# ---- writing + indexing a vault note -- shared by ingest_note and
# ingest_file, both of which just build the body text differently and hand
# it here ----

def _write_note(title: str, tags, audience, body: str) -> tuple[Path, str]:
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slugify(title)
    path = INBOX_DIR / f"{slug}.md"
    if path.exists():   # don't clobber an earlier note with the same title -- give this one its own file
        path = INBOX_DIR / f"{slug}-{uuid.uuid4().hex[:8]}.md"

    front_lines = []
    if tags:
        front_lines.append("tags: [" + ", ".join(tags) + "]")
    if audience:
        front_lines.append(f"audience: {audience}")
    frontmatter = "---\n" + "\n".join(front_lines) + "\n---\n\n" if front_lines else ""

    path.write_text(f"{frontmatter}# {title}\n\n{body}\n", encoding="utf-8")
    source = str(path.relative_to(VAULT_PATH))
    return path, source


async def _index_note(client: httpx.AsyncClient, path: Path, source: str, title: str, tags, audience) -> int:
    text = path.read_text(encoding="utf-8")
    body = FRONTMATTER_RE.sub("", text, count=1)   # frontmatter itself isn't searchable content
    blocks = blockify(body)
    if not blocks:
        return 0

    vectors = await embed_texts(client, [b["text"] for b in blocks])

    points = []
    for block, vector in zip(blocks, vectors):
        h = chunk_hash(block["text"])
        points.append({
            # same source+hash keying as vault_sync.py -- see this file's top comment
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}::{h}")),
            "vector": vector,
            "payload": {
                "text": block["text"],
                "chunk_type": block["chunk_type"],
                "heading_path": block["heading_path"],
                "source": source,
                "note_title": title,
                "tags": tags or [],
                "audience": audience,
                "chunk_hash": h,
            },
        })

    await qdrant_ensure_collection(client, len(vectors[0]))
    await qdrant_delete_source(client, source)
    await qdrant_upsert(client, points)
    return len(points)


@mcp.tool()
async def search_vault(query: str, top_k: int = 5, tags: list[str] | None = None) -> str:
    """Search Eldar's Obsidian vault (personal notes, reference material, saved articles) for content relevant to a query.

    Use this whenever Eldar asks something that might be answered by
    something he previously saved -- a fact, a decision, a reference note,
    a past article -- rather than guessing or saying you don't know.

    Args:
        query: What to search for, in natural language. Gets embedded and
            matched against the vault's indexed chunks -- phrase it like the
            thing you're looking for, not like a keyword search.
        top_k: Max number of results to return. Default 5.
        tags: Optional list of tags to restrict results to (matches notes
            carrying ANY of these tags). Leave unset to search everything.

    Returns:
        The matching chunks as text, each with its source note, heading,
        tags, and relevance score, most relevant first. "No matches found."
        if nothing came back.
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty")

    async with httpx.AsyncClient() as client:
        vector = (await embed_texts(client, [query]))[0]
        hits = await qdrant_search(client, vector, top_k, tags)

    if not hits:
        return "No matches found."

    lines = []
    for hit in hits:
        p = hit["payload"]
        lines.append(
            f"[{hit['score']:.2f}] {p['note_title']} -- {p['source']}\n"
            f"heading: {p['heading_path'] or '(top level)'}   tags: {p.get('tags') or []}\n\n"
            f"{p['text']}"
        )
    return "\n\n---\n\n".join(lines)


@mcp.tool()
async def ingest_note(content: str, title: str, tags: list[str] | None = None, audience: str | None = None) -> str:
    """Save text content as a new note in Eldar's Obsidian vault and index it for search.

    Use this when Eldar asks you to remember/save/note down something --
    pasted text, a summary, something he dictated -- as a durable,
    searchable note rather than only in this conversation. The note is
    written to the vault's Inbox/ folder; Eldar re-files it into his own
    folder structure later if he wants.

    Args:
        content: The note body, as markdown. Real headings (#, ##, ...) are
            picked up and improve how the note gets chunked for search.
        title: Short title for the note -- becomes both the filename and
            the note's top heading.
        tags: Optional list of freeform tags (e.g. ["legal", "poland"]).
        audience: Optional -- "everyone", "adults", or "eldar-only" if this
            note should eventually be scoped to specific family members.
            Leave unset for a normal note.

    Returns:
        Confirmation with the vault path and how many chunks were indexed.
    """
    if not content or not content.strip():
        raise ValueError("content must not be empty")
    if not title or not title.strip():
        raise ValueError("title must not be empty")

    path, source = _write_note(title, tags, audience, content.strip())

    async with httpx.AsyncClient() as client:
        chunk_count = await _index_note(client, path, source, title, tags, audience)

    return f"Saved to {source} and indexed ({chunk_count} chunks)."


# sends this to LiteLLM's vision-capable model:
#   {"model": "qwen3.8-27b-Expert-Vision", "messages": [{"role": "user",
#    "content": [{"type": "text", "text": "..."}, {"type": "image_url",
#    "image_url": {"url": "data:image/png;base64,..."}}]}], "max_tokens": 500}
# returns the usual OpenAI chat shape: {"choices": [{"message": {"content": "a photo of..."}}]}
async def _caption_image(client: httpx.AsyncClient, image_bytes: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    headers = {"Authorization": f"Bearer {LITELLM_API_KEY}"}
    resp = await client.post(
        f"{LITELLM_URL}/chat/completions",
        headers=headers,
        json={
            "model": VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail, including any visible text. "
                                              "This description will be used to make the image searchable "
                                              "later, so be specific and thorough."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                ],
            }],
            "max_tokens": 500,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _extract_pdf_text(path: Path) -> str:
    import pypdf   # only ingest_file's PDF branch needs this -- deferred import keeps the top-of-file import list honest about what every request pays for
    reader = pypdf.PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


@mcp.tool()
async def ingest_file(file_path: str, title: str | None = None, tags: list[str] | None = None, audience: str | None = None) -> str:
    """Save a file (image, PDF, or other document) into Eldar's Obsidian vault and index its content for search.

    file_path must be a path this server can already read -- either
    /opt/data/cache/images/... or /opt/data/cache/audio/... (where your own
    incoming Telegram attachments already land) or /shared-output/... (where
    generated media gets written). Not an arbitrary path on your own side --
    if you're not sure what path to pass, say so rather than guessing, and
    never create files under /opt/data/obsidian_vault -- that is NOT the
    real vault.

    What happens depends on the file type:
    - PDF: text is extracted and indexed.
    - Image: a detailed caption is generated (via the local vision model)
      and indexed, so the image becomes findable by what it shows.
    - Anything else: read as plain text if possible; otherwise the file is
      still saved and linked from a note, but won't be text-searchable.
    The original file is always copied into the vault's Attachments/ folder
    and linked from the note, so it's not just a caption -- Eldar can open
    the real file from Obsidian.

    Args:
        file_path: Absolute path to the file, already readable by this server.
        title: Optional title for the note. Defaults to the filename.
        tags: Optional list of freeform tags.
        audience: Optional -- "everyone", "adults", or "eldar-only".

    Returns:
        Confirmation with the vault paths (note + attachment) and whether
        text/caption extraction succeeded.
    """
    src = Path(file_path).resolve()
    if not any(src.is_relative_to(root.resolve()) for root in ALLOWED_INPUT_ROOTS):   # Path.is_relative_to, py3.9+ (base image is 3.12)
        allowed = " or ".join(str(r) for r in ALLOWED_INPUT_ROOTS)
        raise ValueError(f"file_path must be under {allowed} (got {file_path})")
    if not src.is_file():
        raise ValueError(f"no such file: {file_path}")

    display_title = title or src.stem
    mime_type = mimetypes.guess_type(src.name)[0] or "application/octet-stream"
    ext = src.suffix.lower()

    ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = ATTACHMENTS_DIR / src.name
    if dest.exists():
        dest = ATTACHMENTS_DIR / f"{src.stem}-{uuid.uuid4().hex[:8]}{src.suffix}"
    shutil.copy2(src, dest)
    attachment_ref = str(dest.relative_to(VAULT_PATH))

    extracted = ""
    status = "attachment only, no text extracted"

    async with httpx.AsyncClient() as client:
        if ext == ".pdf":
            try:
                extracted = _extract_pdf_text(dest)
                status = "PDF text extracted" if extracted else "PDF had no extractable text (likely scanned/image-only)"
            except Exception as exc:   # corrupted/encrypted PDF -- keep the attachment, just skip text extraction
                logger.warning("PDF extraction failed for %s: %s", dest, exc)
                status = f"PDF text extraction failed ({exc}) -- attachment saved anyway"
        elif ext in IMAGE_EXTENSIONS:
            try:
                extracted = await _caption_image(client, dest.read_bytes(), mime_type)
                status = "image captioned"
            except httpx.HTTPError as exc:   # vision model unreachable/erroring -- keep the attachment, just skip captioning
                logger.warning("image captioning failed for %s: %s", dest, exc)
                status = f"image captioning failed ({exc}) -- attachment saved anyway"
        else:
            try:
                extracted = dest.read_text(encoding="utf-8")
                status = "read as plain text"
            except (UnicodeDecodeError, OSError):   # binary/unreadable -- fine, attachment-only note
                pass

        link = f"![[{attachment_ref}]]" if ext in IMAGE_EXTENSIONS else f"[[{attachment_ref}]]"
        body = f"{link}\n\n{extracted}" if extracted else link

        path, source = _write_note(display_title, tags, audience, body)
        chunk_count = await _index_note(client, path, source, display_title, tags, audience)

    return f"Saved attachment to {attachment_ref}, note at {source} ({status}, {chunk_count} chunks indexed)."


if __name__ == "__main__":
    # exposes search_vault/ingest_note/ingest_file over streamable-http at
    # :8000/mcp for Hermes's mcp_servers.vault entry (k8s/p3/hermes-agent.yaml)
    # -- same transport/path as media-gen-mcp, no json_response kwarg (see
    # that server's own top-of-file comment for why: it silently drops
    # progress/streaming messages under load).
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
