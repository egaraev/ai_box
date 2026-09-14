# n8n: custom image for the ingestion pipeline (Obsidian -> Qdrant sync, and
# eventually Gmail/Telegram -> Obsidian). Same "build once, run locally, no
# registry" pattern as apps/vault-mcp and apps/media-gen-mcp -- built and
# imported on aibox directly, k8s/n8n/n8n-deployment.yaml just references
# the resulting local image tag.
#
# WHY a custom image at all, instead of the stock n8nio/n8n: n8n's own
# Execute Command node can only run what's inside its own container -- it
# has no way to reach scripts/.venv on the host. So this bakes the 3
# ingestion scripts (scripts/vault_sync.py, gmail_sync.py, telegram_sync.py)
# and their own venv straight into the image, and a workflow's Execute
# Command node just calls that venv's python3 by full path. Confirmed via
# n8n's own Dockerfile (github.com/n8n-io/n8n/blob/master/docker/images/
# n8n/Dockerfile) that the official image is Alpine-based (n8nio/base,
# itself node:24-alpine) -- apk, not apt, below.
#
# ROOT, DELIBERATELY (same call as vault-mcp/media-gen-mcp, see their own
# Dockerfiles/comments): the official image drops to a non-root "node" user,
# which would hit the exact same host-permission wall hermes-agent's non-root
# "hermes" user just did against /data/obsidian_vault (see apps/vault-mcp/
# server.py's HERMES_GID/_chgrp_writable comment for that whole saga) --
# except this time it'd ALSO need to touch Gmail OAuth token files and a
# Telegram session file. Running this container as root sidesteps needing a
# second permission-matching dance for a third identity; consistent with
# how every other single-pod service on this single-user cluster is already
# run (vault-mcp, media-gen-mcp, Honcho's trust auth, etc).
#
# BUILD / DEPLOY (run on aibox, not from a repo checkout elsewhere -- same
# gap as vault-mcp/media-gen-mcp, the image has to exist as a local
# containerd/docker image on the node before the Deployment can start).
# ONE DIFFERENCE from vault-mcp/media-gen-mcp's build command: those `cd`
# into their own app dir first because everything they COPY lives right
# there. This image needs scripts/*.py from ../../scripts/ (the same files
# scripts/.venv runs on the host, deliberately not duplicated into this
# dir), so the build context has to be the repo ROOT instead, with -f
# pointing at this Dockerfile:
#   cd ~/repos/aibox-platform
#   docker build -f apps/n8n/Dockerfile -t aibox-n8n:latest .
#   docker save aibox-n8n:latest | sudo k3s ctr images import -
# After editing scripts/*.py or this Dockerfile, repeat both, then:
#   kubectl rollout restart deployment/n8n -n ai-integration
# (a plain `git push` + ArgoCD sync alone does NOT pick up a code/image
# change -- only k8s/n8n/n8n-deployment.yaml itself self-triggers via
# ArgoCD; the image has to be rebuilt and reimported by hand first, same
# gap as vault-mcp/media-gen-mcp.)
FROM n8nio/n8n:2.38.7

USER root

# python3/py3-pip: Alpine package names (this base is Alpine, not Debian --
# apt-get here would just fail with "command not found").
RUN apk add --no-cache python3 py3-pip

# Own venv, same reasoning scripts/.venv has on the host: Alpine's system
# python3 is externally-managed too (pip install without a venv refuses),
# and this also keeps these deps cleanly separated from whatever n8n's own
# Node runtime needs. Paths below are relative to the repo-root build
# context (see the build command above) -- scripts/, not just requirements.txt.
COPY scripts/requirements.txt /opt/scripts/requirements.txt
RUN python3 -m venv /opt/scripts-venv \
    && /opt/scripts-venv/bin/pip install --no-cache-dir -r /opt/scripts/requirements.txt

COPY scripts/vault_sync.py scripts/telegram_sync.py scripts/gmail_sync.py /opt/scripts/

# Stays root at runtime (see top comment) -- no USER line back to "node".
