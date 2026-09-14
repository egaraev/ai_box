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
# Command node just calls that venv's python3 by full path.
#
# WHY MULTI-STAGE (found live 2026-09-14, cost one failed build first):
# n8nio/n8n:2.38.7 is Alpine 3.24-based, but it's a Docker Hardened Image --
# `apk` itself is stripped out entirely, not just missing from PATH, so
# `apk add python3` fails outright ("apk: not found"). Fix: build Python +
# the venv in a NORMAL python:3.12-alpine stage (pinned to the same Alpine
# 3.24 base, so the binaries are musl-ABI-compatible) and COPY the finished
# install into the hardened final stage instead of trying to apk-install
# anything into it directly.
# Confirmed live what actually needs copying, rather than guessing twice:
# `ldd` on Alpine python3 showed it only directly links musl libc + its own
# libpython3.12.so (both copied below as part of the install). The hardened
# n8n image already ships libssl/libcrypto/libz/libbz2/liblzma/libexpat
# (presumably for Node's own use) -- confirmed via `ls /usr/lib` in that
# image -- so those don't need copying. libffi was the one real gap found
# by that first check (absent from that image, but needed at import time --
# not link time, so plain `ldd` on the interpreter won't show it -- by
# `cryptography`'s `cffi` backend, a transitive dep of both google-auth and
# telethon): copied explicitly below.
# SECOND gap, found live the same way one layer deeper (2026-09-14): `ldd`
# on the *interpreter* only shows what the interpreter itself links --  it
# says nothing about the lib-dynload/*.so extension modules that get
# dlopen'd on `import X`, each of which can need its own separate shared
# lib. `_sqlite3.cpython-312-*.so` (part of the stdlib, pulled in because
# qdrant_client unconditionally does `import sqlite3` at module load, even
# though only its *remote* client is used here, never the local/embedded
# mode that actually needs sqlite) needs libsqlite3.so.0, which isn't in
# the hardened n8n image either -- confirmed via `ldd` directly on that
# lib-dynload .so in the pybuild stage. Copied explicitly below, same
# pattern as libffi. Lesson for next time a similar gap shows up: check
# every lib-dynload/*.so this way, not just the interpreter binary.
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

# ── stage 1: build Python + the scripts' venv on a NORMAL (non-hardened)
#    Alpine image, where apk actually works ──
FROM python:3.12-alpine3.24 AS pybuild

COPY scripts/requirements.txt /opt/scripts/requirements.txt
RUN python3 -m venv /opt/scripts-venv \
    && /opt/scripts-venv/bin/pip install --no-cache-dir -r /opt/scripts/requirements.txt

COPY scripts/vault_sync.py scripts/telegram_sync.py scripts/gmail_sync.py /opt/scripts/

# ── stage 2: the real n8n image, with stage 1's Python install copied in
#    (not apk-installed -- see top comment for why that doesn't work here) ──
FROM n8nio/n8n:2.38.7

USER root

# The interpreter itself + its own shared lib + the stdlib (including the
# lib-dynload/*.so extension modules -- _ssl, _socket, etc. -- which is why
# the whole lib/python3.12 tree is copied, not just the bin/).
COPY --from=pybuild /usr/local/bin/python3* /usr/local/bin/
COPY --from=pybuild /usr/local/lib/libpython3.12.so* /usr/local/lib/
COPY --from=pybuild /usr/local/lib/python3.12 /usr/local/lib/python3.12
# libffi: the one confirmed-missing system lib (see top comment) -- needed
# by cffi/cryptography (google-auth + telethon both pull it in transitively)
# at import time via dlopen, not at link time, so it doesn't show up on a
# plain `ldd` of the python3 binary the way the already-present libs did.
COPY --from=pybuild /usr/lib/libffi.so* /usr/lib/
# libsqlite3: the second confirmed-missing lib (see top comment) -- needed
# by the stdlib's _sqlite3 extension module, which qdrant_client imports
# unconditionally even though only its remote client is used here.
COPY --from=pybuild /usr/lib/libsqlite3.so* /usr/lib/

# The actual venv + the 3 scripts.
COPY --from=pybuild /opt/scripts-venv /opt/scripts-venv
COPY --from=pybuild /opt/scripts /opt/scripts

# Stays root at runtime (see top comment) -- no USER line back to "node".
