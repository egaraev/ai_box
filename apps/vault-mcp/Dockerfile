# vault-mcp: built and run locally on aibox, no registry involved.
# See k8s/p2/vault-mcp-deployment.yaml (imagePullPolicy: IfNotPresent) and
# the deploy notes at the top of that file for the build command and how
# to redeploy after an edit. Same pattern as apps/media-gen-mcp/Dockerfile.
FROM python:3.12-slim

WORKDIR /app

# tesseract-ocr: system engine pytesseract calls out to (the local OCR
# fallback used only when the vision model call itself fails -- see
# server.py's _ocr_image_locally). Language packs match what Eldar's
# documents actually show up in so far (English, Polish, Russian) --
# add more (apt list 'tesseract-ocr-*') if a document in another language
# needs this fallback path.
#
# ghostscript: camelot's own PDF page rendering (table extraction, real
# text-layer PDFs only -- see server.py's _extract_pdf_tables). libglib2.0-0:
# a runtime lib opencv-python-headless (camelot[cv]'s "lattice" table-finder)
# needs even in its "headless" build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-pol tesseract-ocr-rus \
        ghostscript libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# vault_core.py: shared OCR/chunking/Qdrant logic. server.py: the MCP
# server (unchanged behavior). upload_web.py: the new browser upload page,
# 2026-09-17 -- both server.py and upload_web.py import vault_core.py, see
# its top comment. entrypoint.sh runs both processes -- see its own comment.
COPY vault_core.py server.py upload_web.py entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000
EXPOSE 8081

CMD ["./entrypoint.sh"]
