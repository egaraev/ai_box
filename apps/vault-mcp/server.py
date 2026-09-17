"""vault-mcp: an HTTP MCP server exposing Obsidian-vault search and ingest to Hermes.

Same pattern as apps/media-gen-mcp/server.py (mcp>=2.0's MCPServer, HTTP
streamable-http transport on :8000/mcp, registered in Hermes's mcp_servers
config block -- see k8s/p3/hermes-agent.yaml). Three tools:

  search_vault  -- embed a query via LiteLLM, search Qdrant's obsidian_vault
                   collection, return the matching chunks as text.
  ingest_note   -- write text Eldar gives Hermes as a new vault note
                   (<folder>/<slug>.md, folder defaults to Inbox), then
                   chunk+embed+upsert it.
  ingest_file   -- same, but for a file already readable by this server
                   (image/PDF/docx/other): extracts text, OCRs images and
                   scanned PDFs via the local vision model, copies the
                   original into Attachments/, links it from the note.

REFACTORED 2026-09-17: all the actual vault/OCR/chunking/Qdrant logic that
used to live directly in this file now lives in vault_core.py instead --
same functions, same behavior, just pulled out so upload_web.py (the new
browser-facing upload page, same container, separate process on :8081) can
reuse it without duplicating a second copy of the OCR pipeline. This file
is now just the MCP tool wrappers: request validation + calling into
vault_core + formatting the reply string, same as before. See
vault_core.py's own top comment for why this is two processes instead of
one, and entrypoint.sh for how both get started.

Folder routing (2026-09-14): both ingest tools take an optional folder
param so Hermes can route a save straight into Family/Health/Legal/
Household/Reference/AI instead of everything always landing in Inbox/ for
Eldar to re-file by hand later. Restricted to a fixed whitelist
(vault_core.ALLOWED_FOLDERS) rather than any string Hermes sends, so a
hallucinated/typo'd folder name can't create new top-level folders Eldar
never asked for.

OCR (2026-09-14): Eldar's real use case for images/PDFs here is photographed
notes/documents, not scenery -- so images are read via the vision model with
a verbatim-transcription prompt (not a scene-description caption), and a PDF
with no extractable text layer (a scanned/photographed PDF) falls back to
rasterizing each page and running the same vision-OCR path per page. Both
paths fall back further to local pytesseract if the vision model call itself
fails (LiteLLM down/unreachable) -- weaker on skewed/handwritten text, but
doesn't depend on anything else being up. docx is extracted directly via
python-docx (it's already digital text, no OCR needed).

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
from pathlib import Path

import httpx
from mcp.server.mcpserver import MCPServer

import vault_core

mcp = MCPServer("vault")


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
        vector = (await vault_core.embed_texts(client, [query], "search_query"))[0]
        hits = await vault_core.qdrant_search(client, vector, top_k, tags)

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

    path, source = vault_core.write_note(title, tags, audience, content.strip(), folder)

    async with httpx.AsyncClient() as client:
        chunk_count = await vault_core.index_note(client, path, source, title, tags, audience)

    return f"Saved to {source} and indexed ({chunk_count} chunks)."


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
      vision model instead (capped -- see vault_core.MAX_OCR_PDF_PAGES; a
      very long scanned PDF only gets its first pages indexed) -- no table
      extraction on that path, there's no real table structure to find on
      a scanned page.
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
    if folder not in vault_core.ALLOWED_FOLDERS:   # fail before touching the filesystem, not after copying the attachment
        raise ValueError(f"folder must be one of {sorted(vault_core.ALLOWED_FOLDERS)} (got {folder!r})")

    src = Path(file_path).resolve()
    if not any(src.is_relative_to(root.resolve()) for root in vault_core.ALLOWED_INPUT_ROOTS):   # Path.is_relative_to, py3.9+ (base image is 3.12)
        allowed = " or ".join(str(r) for r in vault_core.ALLOWED_INPUT_ROOTS)
        # Spell out the fix, not just the rule -- this text is the whole thing
        # Hermes sees when the call fails, and it kept re-guessing paths or
        # inventing a different error instead of doing the copy the docstring
        # already asks for. Repeating the actual fix here (copy into
        # Attachments/, then retry with that path) where the caller can't miss it.
        raise ValueError(
            f"file_path must be under {allowed} (got {file_path}). "
            f"If this file is from your own cache, first copy it into {vault_core.ATTACHMENTS_DIR} "
            f"yourself, then call ingest_file again with that new path under {vault_core.ATTACHMENTS_DIR}. "
            f"Report this exact error if you're unsure what path to use -- don't guess a different one."
        )
    if not src.is_file():
        raise ValueError(f"no such file: {file_path}")

    dest = vault_core.copy_to_attachments(src)

    async with httpx.AsyncClient() as client:
        result = await vault_core.ingest_attachment(client, dest, title, tags, audience, folder)

    return (
        f"Saved attachment to {result['attachment_ref']}, note at {result['source']} "
        f"({result['status']}, {result['chunk_count']} chunks indexed)."
    )


if __name__ == "__main__":
    # exposes search_vault/ingest_note/ingest_file over streamable-http at
    # :8000/mcp for Hermes's mcp_servers.vault entry (k8s/p3/hermes-agent.yaml)
    # -- same transport/path as media-gen-mcp, no json_response kwarg (see
    # that server's own top-of-file comment for why: it silently drops
    # progress/streaming messages under load).
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
