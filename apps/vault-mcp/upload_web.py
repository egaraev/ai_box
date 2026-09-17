# upload_web.py -- plain browser upload page for the Obsidian vault, added
# 2026-09-17 (Eldar: "I need some webinterface which will allow me to
# upload data on aibox remotely... must do OCR, Markitdown or whatever is
# needed to create MD file and put it on obsidian storage").
#
# This is Option B from that discussion: instead of a new standalone
# service with its own copy of the OCR/chunking/Qdrant logic, this is a
# second small process in the SAME vault-mcp container, importing the exact
# same vault_core.py the MCP server (server.py) already uses and has
# proven live via Hermes/Telegram. So a photo/PDF/docx uploaded here goes
# through the identical extraction path (vision-model OCR with pytesseract
# fallback, PDF text+table extraction, docx extraction), gets written into
# the vault as a real .md note under Attachments/ + the chosen folder, and
# is chunk+embed+upserted into the same Qdrant obsidian_vault collection --
# not a separate pipeline to keep in sync.
#
# Why a second process instead of adding routes onto server.py itself: see
# vault_core.py's top comment. Runs on its own port (8081) so it can be
# NodePort-exposed independently of the MCP port (8000, ClusterIP-only,
# Hermes-internal) -- see k8s/p2/vault-mcp-deployment.yaml's new
# vault-uploader Service. Plain NodePort for now, same pattern as every
# other exposed service on aibox (comfyui, router-dashboard) -- an
# ingress/reverse proxy for LAN-friendly URLs + auth is a later step,
# Eldar's own call.
#
# Run: python upload_web.py (see entrypoint.sh for how this starts
# alongside server.py in the container)
import mimetypes
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

import vault_core

app = FastAPI(title="Vault Upload")

# Rough cap so a huge accidental upload (or an abusive request, if this
# ever gets exposed beyond the LAN) can't fill the vault's disk or hang the
# vision-OCR call for minutes on an enormous image. 25MB comfortably covers
# a multi-page phone-scanned PDF or a full-res photo.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

FOLDER_OPTIONS = "".join(
    f'<option value="{f}"{" selected" if f == "Inbox" else ""}>{f}</option>'
    for f in sorted(vault_core.ALLOWED_FOLDERS)
)

# Single self-contained page -- a form field can't crash the request the
# way a bad path can, so this stays plain HTML/JS with no build step,
# fetch()-based so the phone doesn't do a full page reload per upload
# (useful when uploading several photos of the same document in a row).
PAGE = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vault Upload</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 480px; margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ font-size: 1.3rem; }}
  label {{ display: block; margin-top: 1rem; font-weight: 600; }}
  input, select {{ width: 100%; padding: 0.5rem; margin-top: 0.25rem; box-sizing: border-box; font-size: 1rem; }}
  button {{ margin-top: 1.5rem; width: 100%; padding: 0.75rem; font-size: 1rem; cursor: pointer; }}
  #status {{ margin-top: 1rem; padding: 0.75rem; border-radius: 4px; white-space: pre-wrap; display: none; }}
  #status.ok {{ display: block; background: #e6f4ea; color: #1e4620; }}
  #status.err {{ display: block; background: #fce8e6; color: #611a15; }}
  #status.busy {{ display: block; background: #eee; color: #333; }}
</style>
</head>
<body>
<h1>Upload to Obsidian vault</h1>
<form id="f">
  <label for="file">File (photo, PDF, docx, ...)</label>
  <input type="file" id="file" name="file" accept="image/*,application/pdf,.docx,.txt,.md" required>

  <label for="folder">Folder</label>
  <select id="folder" name="folder">{FOLDER_OPTIONS}</select>

  <label for="title">Title (optional -- defaults to filename)</label>
  <input type="text" id="title" name="title" placeholder="e.g. Prescription 2026-09-17">

  <label for="tags">Tags (optional, comma-separated)</label>
  <input type="text" id="tags" name="tags" placeholder="e.g. health, prescription">

  <button type="submit">Upload &amp; index</button>
</form>
<div id="status"></div>
<script>
const form = document.getElementById("f");
const statusEl = document.getElementById("status");
form.addEventListener("submit", async (e) => {{
  e.preventDefault();
  const fileInput = document.getElementById("file");
  if (!fileInput.files.length) return;

  statusEl.className = "busy";
  statusEl.textContent = "Uploading and processing (OCR can take up to a minute for photos/scans)...";

  const body = new FormData();
  body.append("file", fileInput.files[0]);
  body.append("folder", document.getElementById("folder").value);
  body.append("title", document.getElementById("title").value);
  body.append("tags", document.getElementById("tags").value);

  try {{
    const resp = await fetch("/api/upload", {{ method: "POST", body }});
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "Upload failed");
    statusEl.className = "ok";
    statusEl.textContent = `Saved: ${{data.source}}\\n${{data.status}}\\n${{data.chunk_count}} chunks indexed.`;
    form.reset();
  }} catch (err) {{
    statusEl.className = "err";
    statusEl.textContent = "Error: " + err.message;
  }}
}});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def root():
    return PAGE


# Accepts a browser-uploaded file + form fields, saves it into the vault's
# Attachments/ folder, then runs it through vault_core's shared
# extract-and-index pipeline -- same code path server.py's MCP ingest_file
# tool uses, see vault_core.ingest_attachment.
@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    folder: str = Form("Inbox"),
    title: str = Form(""),
    tags: str = Form(""),
):
    if folder not in vault_core.ALLOWED_FOLDERS:
        raise HTTPException(status_code=400, detail=f"folder must be one of {sorted(vault_core.ALLOWED_FOLDERS)}")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"file too large ({len(data)} bytes, max {MAX_UPLOAD_BYTES})")
    if not data:
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    filename = file.filename or "upload"
    if not Path(filename).suffix:
        # browsers usually give photos a real name+extension, but a camera
        # capture blob sometimes doesn't -- guess one from the content-type
        # so extract_pdf_text/ocr_image/etc. can still dispatch on it
        guessed = mimetypes.guess_extension(file.content_type or "") or ""
        filename = f"{filename}{guessed}"

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] or None
    dest = vault_core.save_upload_to_attachments(filename, data)

    async with httpx.AsyncClient() as client:
        try:
            result = await vault_core.ingest_attachment(client, dest, title or None, tag_list, None, folder)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    return result


@app.get("/health")
def health():
    return {"status": "ok", "vault": str(vault_core.VAULT_PATH)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
