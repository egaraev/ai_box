# media-gen-mcp: built and run locally on aibox, no registry involved.
# See k8s/media-gen/media-gen-mcp-deployment.yaml (imagePullPolicy:
# IfNotPresent) and the deploy notes at the top of that file for the build
# command and how to redeploy after an edit.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

EXPOSE 8000

CMD ["python", "server.py"]
