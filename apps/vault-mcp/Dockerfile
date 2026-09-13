# vault-mcp: built and run locally on aibox, no registry involved.
# See k8s/p2/vault-mcp-deployment.yaml (imagePullPolicy: IfNotPresent) and
# the deploy notes at the top of that file for the build command and how
# to redeploy after an edit. Same pattern as apps/media-gen-mcp/Dockerfile.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

EXPOSE 8000

CMD ["python", "server.py"]
