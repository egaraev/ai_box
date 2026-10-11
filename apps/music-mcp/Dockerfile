# music-mcp: built and run locally on aibox, no registry involved --
# same reasoning as movies-mcp's identical Dockerfile.
#
# BUILD / DEPLOY (run on aibox):
#   cd ~/repos/aibox-platform/apps/music-mcp
#   docker build -t music-mcp:latest .
#   docker save music-mcp:latest | sudo k3s ctr images import -
# After editing server.py, repeat both of those, then:
#   kubectl rollout restart deployment/music-mcp -n ai-mcp
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

EXPOSE 8000

CMD ["python", "server.py"]
