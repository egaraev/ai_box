# media-augment-mcp: built and run locally on aibox, no registry involved --
# same reasoning as media-gen-mcp's identical Dockerfile (one more moving
# part -- registry auth, a push step, image scanning -- with no real
# benefit on a single-user home cluster).
#
# BUILD / DEPLOY (run on aibox):
#   cd ~/repos/aibox-platform/apps/media-augment-mcp
#   docker build -t media-augment-mcp:latest .
#   docker save media-augment-mcp:latest | sudo k3s ctr images import -
# After editing server.py, repeat both of those, then:
#   kubectl rollout restart deployment/media-augment-mcp -n ai-mcp
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

EXPOSE 8000

CMD ["python", "server.py"]
