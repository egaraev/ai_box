"""vault-mcp: an HTTP MCP server exposing Obsidian-vault search and ingest to Hermes.

Same pattern as apps/media-gen-mcp/server.py (mcp>=2.0's MCPServer, HTTP
streamable-http transport on :8000/mcp, registered in Hermes's mcp_servers
config block -- see k8s/p3/hermes-agent.yaml). Two tools:

  search_vault  -- embed a query via LiteLLM, search Qdrant's obsidian_vault
                   collection, return the matching chunks as text.
  ingest_note   -- write text Eldar gives Hermes as a new vault note
                   (<folder>/<slug>.md, folder defaults to Inbox), then
                   chunk+embed+upsert it.
  ingest_file   -- same, but for a file already readable by this server
                   (image/PDF/docx/other): extracts text, OCRs images and
                   scanned PDFs via the local vision model, copies the
                   original into Attachments/, links it from the note.

Folder routing (2026-09-14): both tools take an optional folder param so
Hermes can route a save straight into Family/Health/Legal/Household/
Reference/AI instead of everything always landing in Inbox/ for Eldar to
re-file by hand later. Restricted to a fixed whitelist (ALLOWED_FOLDERS
below) rather than any string Hermes sends, so a hallucinated/typo'd folder
name can't create new top-level folders Eldar never asked for.

OCR (2026-09-14): Eldar's real use case for images/PDFs here is photographed
notes/documents, not scenery -- so images are read via the vision model with
a verbatim-transcription prompt (not a scene-description caption), and a PDF
with no extractable text layer (a scanned/photographed PDF) falls back to
rasterizing each page and running the same vision-OCR path per page. Both
paths fall back further to local pytesseract if the vision model call itself
fails (LiteLLM down/unreachable) -- weaker on skewed/handwritten text, but
doesn't depend on anything else being up. docx is extracted directly via
python-docx (it's already digital text, no OCR needed).
Tables (2026-09-14): a PDF with a real text layer also gets run through
camelot for table structure -- pypdf's extract_text() flattens a table into
hard-to-read run-on text, so any tables camelot finds are appended as plain
pipe rows, same lightweight rendering _extract_docx_text uses for docx
tables. Only attempted on the pypdf (real text layer) path, not the OCR
fallback path -- camelot needs actual vector table structure to find, which
a scanned/photographed page doesn't have, so it would just find nothing
there anyway.
Deliberately still NOT added: ocrmypdf. Its system toolchain (ghostscript/
qpdf/tesseract) mostly duplicates what pymupdf+vision-OCR already covers
here for the scanned-PDF case. Revisit if a real need shows up.

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

ingest_file does NOT reach into hermes-agent's own storage to read a
freshly-received attachment (an earlier version of this tried that -- read
into the hermes-data PVC's cache/ subdirectory where incoming Telegram
attachments land -- and hit a real dead end, see below). Instead,
hermes-agent.yaml now ALSO mounts the real vault (/data/obsidian_vault) at
the identical path this server uses, read-write. So the flow is:
  1. Hermes copies the file it already has (from its own cache) into
     /data/obsidian_vault/Attachments/ itself, using its own file/code
     tools -- both source and destination are in ITS OWN pod, no cross-pod
     access of any kind needed for this step.
  2. Hermes calls ingest_file with that now-in-the-vault path.
  3. This server extracts text (PDF) / captions (image) / links the
     attachment from a new note, then indexes it -- exactly like before,
     just reading from the vault it already owns instead of reaching into
     someone else's pod.
ingest_file's file_path is therefore checked against VAULT_PATH itself
(plus /shared-output, for the rarer case of linking a media-gen output).

WHY NOT read hermes-agent's cache directly (tried first, reverted
2026-09-13): hermes-agent's own pod never stops running and already holds
the hermes-data PVC read-write as its primary volume; k3s's default
storage (local-path-provisioner) does not support a second pod mounting
that PVC concurrently, even read-only, even on the same node -- confirmed
live, the second pod sat Pending indefinitely. A hostPath straight into
local-path-provisioner's on-disk storage for that PVC was tried next and
ALSO stayed Pending for an unclear reason -- rather than keep debugging a
narrower fix for the wrong-shaped problem, Eldar's own suggestion (mount
the vault into Hermes's pod instead) sidesteps the whole class of issue:
this server now only ever touches volumes it fully owns.
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
INBOX_DIR = VAULT_PATH / "Inbox"             # default folder when ingest_note/ingest_file's folder param is left unset
ATTACHMENTS_DIR = VAULT_PATH / "Attachments"

# Folders ingest_note/ingest_file's folder param is allowed to route a note
# into -- a fixed whitelist, not any string Hermes sends, so a confused or
# hallucinated folder name can't scatter notes outside Eldar's real
# top-level vault structure or silently create a new folder he never asked
# for. Matches Eldar's actual Obsidian top-level folders as of 2026-09-14 --
# if he adds a new one in Obsidian, add it here too or Hermes won't be able
# to route into it.
ALLOWED_FOLDERS = {"Inbox", "Family", "Health", "Legal", "Household", "Reference", "AI"}

# This server runs as root in its container (see Dockerfile -- no USER set),
# same as most single-pod services on this cluster, but hermes-agent's own
# gateway process deliberately drops to a non-root "hermes" user (uid/gid
# 10000, see k8s/p3/hermes-agent.yaml) once it boots. That mismatch is real:
# found 2026-09-14 when Hermes could copy a photo into Attachments/ (its own
# writes, as itself) but then couldn't edit or delete the .md notes THIS
# server had written moments earlier, because they came out owned root:root
# and Hermes has no sudo in its container. Hermes worked around it by
# re-ingesting under new titles instead of fixing the original note --
# harmless but left duplicate hash-suffixed files behind every time.
# Keeping this GID here as a plain constant (not read from the environment)
# is a deliberate shortcut, not an oversight: it only has to match
# hermes-agent.yaml's fixed uid/gid 10000, which isn't expected to change,
# and adding a whole env-var/ConfigMap plumbing path for one number felt
# like more moving parts than the problem deserves. If hermes-agent's uid
# ever changes, update it here too.
HERMES_GID = 10000

def _chgrp_writable(path: Path, dir_mode=0o775, file_mode=0o664):
    # Group-owns path by Hermes's gid and makes it group-writable, so the
    # non-root hermes-agent process can edit or delete something this
    # (root) process just created. Root can chown to any group regardless
    # of who currently owns the file, so this is safe to call unconditionally.
    mode = dir_mode if path.is_dir() else file_mode
    os.chown(path, -1, HERMES_GID)   # -1 leaves the user (root) unchanged, only the group changes
    path.chmod(mode)

# Created at import time, not lazily inside ingest_note/ingest_file. Found
# 2026-09-14: on a vault where ingest_file had never yet succeeded even
# once, Attachments/ didn't exist -- but the tool's own docstring tells
# Hermes to copy a file INTO Attachments/ itself BEFORE calling ingest_file,
# so that first-ever call was guaranteed to fail with "No such file or
# directory" no matter what. Hermes misreported that as a permissions
# problem (see /areas/ai-box-k3s.md) -- creating every dir up front, as soon
# as this server starts, closes that chicken-and-egg gap for good (now
# covers all of ALLOWED_FOLDERS, not just Inbox, for the same reason).
# Also re-applies _chgrp_writable to each on every startup (not just at
# creation) so a restart alone heals any folder that somehow drifted back
# to root-only permissions -- cheap, idempotent, no reason not to.
for _folder_name in ALLOWED_FOLDERS:
    _folder_path = VAULT_PATH / _folder_name
    _folder_path.mkdir(parents=True, exist_ok=True)
    _chgrp_writable(_folder_path)
ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
_chgrp_writable(ATTACHMENTS_DIR)

# Where ingest_file is allowed to read from -- see this file's top comment
# for why it's just these two (both volumes this pod fully owns, no
# cross-pod reads into someone else's storage). Anything outside both is
# refused rather than read, so a compromised/confused Hermes call can't
# point this server at an arbitrary host path.
ALLOWED_INPUT_ROOTS = [
    VAULT_PATH,
    Path(os.environ.get("SHARED_INPUT_DIR", "/shared-output")),
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

# Hard cap on how many pages of a scanned/no-text-layer PDF get OCR'd via the
# vision model. The vision model route is locked --parallel 1 (see
# /areas/ai-box-k3s.md), so pages OCR this way are strictly sequential --
# one page can take 30-90s, and the vault MCP tool call itself times out at
# 180s (hermes-agent.yaml's mcp_servers.vault.timeout). Past this cap the
# note still gets the pages that were OCR'd plus a note that it was
# truncated, rather than the whole call failing/timing out.
MAX_OCR_PDF_PAGES = 3

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
# nomic-embed-text-v1.5 wants a "search_document: " prefix on text being
# indexed and "search_query: " on text a search is run with -- without it,
# vectors cluster too close together to tell related notes from unrelated
# ones (see scripts/vault_sync.py's embed_texts for the full story; this is
# the same fix applied here since this function serves both _index_note,
# the document side, and search_vault, the query side). prefix must be
# "search_document" or "search_query" -- both call sites below pass one
# explicitly rather than defaulting, so it's never accidentally left off.
async def embed_texts(client: httpx.AsyncClient, texts: list[str], prefix: str) -> list[list[float]]:
    headers = {"Authorization": f"Bearer {LITELLM_API_KEY}"}
    resp = await client.post(
        f"{LITELLM_URL}/embeddings",
        json={"model": EMBED_MODEL, "input": [f"{prefix}: {t}" for t in texts]},
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

def _write_note(title: str, tags, audience, body: str, folder: str = "Inbox") -> tuple[Path, str]:
    if folder not in ALLOWED_FOLDERS:
        raise ValueError(f"folder must be one of {sorted(ALLOWED_FOLDERS)} (got {folder!r})")
    target_dir = VAULT_PATH / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(title)
    path = target_dir / f"{slug}.md"
    if path.exists():   # don't clobber an earlier note with the same title -- give this one its own file
        path = target_dir / f"{slug}-{uuid.uuid4().hex[:8]}.md"

    front_lines = []
    if tags:
        front_lines.append("tags: [" + ", ".join(tags) + "]")
    if audience:
        front_lines.append(f"audience: {audience}")
    frontmatter = "---\n" + "\n".join(front_lines) + "\n---\n\n" if front_lines else ""

    path.write_text(f"{frontmatter}# {title}\n\n{body}\n", encoding="utf-8")
    _chgrp_writable(path)   # otherwise Hermes (non-root) can't edit or delete the note this (root) process just wrote
    source = str(path.relative_to(VAULT_PATH))
    return path, source


async def _index_note(client: httpx.AsyncClient, path: Path, source: str, title: str, tags, audience) -> int:
    text = path.read_text(encoding="utf-8")
    body = FRONTMATTER_RE.sub("", text, count=1)   # frontmatter itself isn't searchable content
    blocks = blockify(body)
    if not blocks:
        return 0

    vectors = await embed_texts(client, [b["text"] for b in blocks], "search_document")

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
        vector = (await embed_texts(client, [query], "search_query"))[0]
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
async def ingest_note(content: str, title: str, tags: list[str] | None = None, audience: str | None = None, folder: str = "Inbox") -> str:
    """Save text content as a new note in Eldar's Obsidian vault and index it for search.

    Use this when Eldar asks you to remember/save/note down something --
    pasted text, a summary, something he dictated -- as a durable,
    searchable note rather than only in this conversation. Route it into the
    most fitting folder below rather than defaulting to Inbox whenever the
    content clearly fits one -- Inbox is for anything that doesn't.

    Args:
        content: The note body, as markdown. Real headings (#, ##, ...) are
            picked up and improve how the note gets chunked for search.
        title: Short title for the note -- becomes both the filename and
            the note's top heading.
        tags: Optional list of freeform tags (e.g. ["legal", "poland"]).
        audience: Optional -- "everyone", "adults", or "eldar-only" if this
            note should eventually be scoped to specific family members.
            Leave unset for a normal note.
        folder: Which top-level vault folder to file this under -- one of
            "Inbox" (default; anything that doesn't clearly fit elsewhere),
            "Family", "Health", "Legal", "Household", "Reference", or "AI".
            Any other value is rejected.

    Returns:
        Confirmation with the vault path and how many chunks were indexed.
    """
    if not content or not content.strip():
        raise ValueError("content must not be empty")
    if not title or not title.strip():
        raise ValueError("title must not be empty")

    path, source = _write_note(title, tags, audience, content.strip(), folder)

    async with httpx.AsyncClient() as client:
        chunk_count = await _index_note(client, path, source, title, tags, audience)

    return f"Saved to {source} and indexed ({chunk_count} chunks)."


# Last-resort OCR when the vision model call itself fails (LiteLLM down/
# unreachable) -- classic tesseract via pytesseract, no LLM involved. Weaker
# than the vision model on skewed photos or handwriting, but doesn't depend
# on anything else being up.
def _ocr_image_locally(image_bytes: bytes) -> str:
    import io
    import pytesseract
    from PIL import Image
    image = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_string(image).strip()


# sends this to LiteLLM's vision-capable model:
#   {"model": "qwen3.8-27b-Expert-Vision", "messages": [{"role": "user",
#    "content": [{"type": "text", "text": "..."}, {"type": "image_url",
#    "image_url": {"url": "data:image/png;base64,..."}}]}], "max_tokens": 2000}
# returns the usual OpenAI chat shape: {"choices": [{"message": {"content": "..."}}]}
#
# Prompted for verbatim transcription, not a scene caption -- the real use
# case here is photographed notes/documents Eldar wants findable by their
# actual content, not "a photo of a piece of paper". Falls back to local
# pytesseract (see above) if the vision model call itself fails.
async def _ocr_image(client: httpx.AsyncClient, image_bytes: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    headers = {"Authorization": f"Bearer {LITELLM_API_KEY}"}
    prompt = (
        "Transcribe ALL visible text in this image, verbatim and in full, exactly as written "
        "(keep the original language -- don't translate it). Preserve line breaks and structure "
        "(headings, lists, tables) as plain text where it's clear. If there is little or no text, "
        "instead describe what the image shows in one or two sentences."
    )
    try:
        resp = await client.post(
            f"{LITELLM_URL}/chat/completions",
            headers=headers,
            json={
                "model": VISION_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                    ],
                }],
                "max_tokens": 2000,
            },
            timeout=90.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except httpx.HTTPError as exc:
        logger.warning("vision OCR failed (%s) -- falling back to local pytesseract", exc)
        return _ocr_image_locally(image_bytes)


def _extract_pdf_text(path: Path) -> str:
    import pypdf   # only ingest_file's PDF branch needs this -- deferred import keeps the top-of-file import list honest about what every request pays for
    reader = pypdf.PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


# Only meaningful on a PDF with a real text layer (see this file's top
# comment) -- pypdf's extract_text() flattens a table into hard-to-read
# run-on text, so any tables camelot finds get appended separately as plain
# pipe rows. camelot's "lattice" flavor (the default) needs visible table
# borders/lines to find a table; if Eldar's real PDFs turn out to have
# borderless tables, switch to flavor="stream" (whitespace-based instead).
def _extract_pdf_tables(path: Path) -> str:
    import camelot
    tables = camelot.read_pdf(str(path), pages="all")
    if len(tables) == 0:
        return ""
    parts = []
    for i, table in enumerate(tables):
        rows = table.df.values.tolist()
        lines = [" | ".join(str(cell).strip() for cell in row) for row in rows]
        parts.append(f"Table {i + 1} (page {table.page}):\n" + "\n".join(lines))
    return "\n\n".join(parts)


# Fallback for a PDF with no extractable text layer (scanned/photographed).
# Rasterizes each page to a PNG via pymupdf (no external system binary
# needed, unlike poppler/pdf2image) and runs it through the same vision-OCR
# path images use. Capped at MAX_OCR_PDF_PAGES -- see that constant's
# comment for why (the vision route is strictly sequential).
async def _extract_pdf_via_ocr(client: httpx.AsyncClient, path: Path) -> str:
    import fitz   # pymupdf
    doc = fitz.open(str(path))
    try:
        page_count = doc.page_count
        pages_to_ocr = min(page_count, MAX_OCR_PDF_PAGES)
        texts = []
        for i in range(pages_to_ocr):
            png_bytes = doc[i].get_pixmap(dpi=200).tobytes("png")
            texts.append(await _ocr_image(client, png_bytes, "image/png"))
        result = "\n\n---\n\n".join(t for t in texts if t)
        if page_count > MAX_OCR_PDF_PAGES:
            result += f"\n\n[OCR truncated -- {page_count - MAX_OCR_PDF_PAGES} more page(s) not processed]"
        return result
    finally:
        doc.close()


# docx is already digital text -- no OCR needed, just walk paragraphs and
# tables in document order. Simple pipe-row rendering for tables (not a
# proper markdown table, but good enough to be searchable and readable).
def _extract_docx_text(path: Path) -> str:
    import docx   # python-docx
    document = docx.Document(str(path))
    parts = []
    for para in document.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n\n".join(parts).strip()


@mcp.tool()
async def ingest_file(file_path: str, title: str | None = None, tags: list[str] | None = None, audience: str | None = None, folder: str = "Inbox") -> str:
    """Save a file (image, PDF, or other document) into Eldar's Obsidian vault and index its content for search.

    file_path must already be under the vault itself
    (/data/obsidian_vault/...) or under /shared-output/... (where generated
    media gets written). If you have a file from your own cache
    (/opt/data/cache/images/... or .../audio/...) that isn't in the vault
    yet, copy it into /data/obsidian_vault/Attachments/ yourself first (you
    have write access there), then call this with that new path. Not an
    arbitrary path on your own side otherwise -- if you're not sure what
    path to pass, say so rather than guessing.

    What happens depends on the file type:
    - PDF: text is extracted directly if the PDF has a real text layer, and
      any tables on that layer are also extracted and appended (as plain
      rows -- table structure that plain text extraction alone would
      flatten into a hard-to-read run-on). If there's no text layer (a
      scanned/photographed PDF), the first few pages are OCR'd via the
      vision model instead (capped -- see the code comment on
      MAX_OCR_PDF_PAGES; a very long scanned PDF only gets its first pages
      indexed) -- no table extraction on that path, there's no real table
      structure to find on a scanned page.
    - Image: transcribed verbatim via the vision model (this is meant for
      photos of notes/documents, not scenery -- it reads the actual text,
      not just a scene description).
    - .docx: text extracted directly (paragraphs + tables).
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
        folder: Which top-level vault folder to file the NOTE under (the
            attachment itself always goes to Attachments/ regardless) -- one
            of "Inbox" (default; anything that doesn't clearly fit
            elsewhere), "Family", "Health", "Legal", "Household",
            "Reference", or "AI". Any other value is rejected.

    Returns:
        Confirmation with the vault paths (note + attachment) and whether
        text/caption extraction succeeded.
    """
    if folder not in ALLOWED_FOLDERS:   # fail before touching the filesystem, not after copying the attachment
        raise ValueError(f"folder must be one of {sorted(ALLOWED_FOLDERS)} (got {folder!r})")

    src = Path(file_path).resolve()
    if not any(src.is_relative_to(root.resolve()) for root in ALLOWED_INPUT_ROOTS):   # Path.is_relative_to, py3.9+ (base image is 3.12)
        allowed = " or ".join(str(r) for r in ALLOWED_INPUT_ROOTS)
        # Spell out the fix, not just the rule -- this text is the whole thing
        # Hermes sees when the call fails, and it kept re-guessing paths or
        # inventing a different error instead of doing the copy the docstring
        # already asks for. Repeating the actual fix here (copy into
        # Attachments/, then retry with that path) where the caller can't miss it.
        raise ValueError(
            f"file_path must be under {allowed} (got {file_path}). "
            f"If this file is from your own cache, first copy it into {ATTACHMENTS_DIR} "
            f"yourself, then call ingest_file again with that new path under {ATTACHMENTS_DIR}. "
            f"Report this exact error if you're unsure what path to use -- don't guess a different one."
        )
    if not src.is_file():
        raise ValueError(f"no such file: {file_path}")

    display_title = title or src.stem
    mime_type = mimetypes.guess_type(src.name)[0] or "application/octet-stream"
    ext = src.suffix.lower()

    ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
    if src.parent == ATTACHMENTS_DIR:
        dest = src   # Hermes already placed it there itself -- use as-is, don't duplicate it under a new name
    else:
        dest = ATTACHMENTS_DIR / src.name
        if dest.exists() and dest.resolve() != src.resolve():
            dest = ATTACHMENTS_DIR / f"{src.stem}-{uuid.uuid4().hex[:8]}{src.suffix}"
        shutil.copy2(src, dest)
    _chgrp_writable(dest)   # copy2 preserves src's own mode/owner (root's, here) -- fix it up so Hermes can still touch its own attachment later
    attachment_ref = str(dest.relative_to(VAULT_PATH))

    extracted = ""
    status = "attachment only, no text extracted"

    async with httpx.AsyncClient() as client:
        if ext == ".pdf":
            try:
                extracted = _extract_pdf_text(dest)
                if extracted:
                    status = "PDF text extracted"
                    try:
                        tables = _extract_pdf_tables(dest)
                        if tables:
                            extracted = f"{extracted}\n\n{tables}"
                            status = "PDF text extracted (+ tables)"
                    except Exception as exc:   # table extraction is a bonus on top of already-successful text extraction -- don't lose the text over it
                        logger.warning("PDF table extraction failed for %s: %s", dest, exc)
                else:
                    extracted = await _extract_pdf_via_ocr(client, dest)
                    status = "PDF had no text layer -- OCR'd via vision model" if extracted else "PDF OCR found no text"
            except Exception as exc:   # corrupted/encrypted PDF, or OCR fallback itself failed -- keep the attachment, just skip text extraction
                logger.warning("PDF extraction failed for %s: %s", dest, exc)
                status = f"PDF text extraction failed ({exc}) -- attachment saved anyway"
        elif ext in IMAGE_EXTENSIONS:
            try:
                extracted = await _ocr_image(client, dest.read_bytes(), mime_type)
                status = "image OCR'd"
            except Exception as exc:   # vision model AND local OCR fallback both failed -- keep the attachment, just skip extraction
                logger.warning("image OCR failed for %s: %s", dest, exc)
                status = f"image OCR failed ({exc}) -- attachment saved anyway"
        elif ext == ".docx":
            try:
                extracted = _extract_docx_text(dest)
                status = "docx text extracted" if extracted else "docx had no extractable text"
            except Exception as exc:   # corrupted/unsupported docx -- keep the attachment, just skip text extraction
                logger.warning("docx extraction failed for %s: %s", dest, exc)
                status = f"docx text extraction failed ({exc}) -- attachment saved anyway"
        else:
            try:
                extracted = dest.read_text(encoding="utf-8")
                status = "read as plain text"
            except (UnicodeDecodeError, OSError):   # binary/unreadable -- fine, attachment-only note
                pass

        link = f"![[{attachment_ref}]]" if ext in IMAGE_EXTENSIONS else f"[[{attachment_ref}]]"
        body = f"{link}\n\n{extracted}" if extracted else link

        path, source = _write_note(display_title, tags, audience, body, folder)
        chunk_count = await _index_note(client, path, source, display_title, tags, audience)

    return f"Saved attachment to {attachment_ref}, note at {source} ({status}, {chunk_count} chunks indexed)."


if __name__ == "__main__":
    # exposes search_vault/ingest_note/ingest_file over streamable-http at
    # :8000/mcp for Hermes's mcp_servers.vault entry (k8s/p3/hermes-agent.yaml)
    # -- same transport/path as media-gen-mcp, no json_response kwarg (see
    # that server's own top-of-file comment for why: it silently drops
    # progress/streaming messages under load).
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
