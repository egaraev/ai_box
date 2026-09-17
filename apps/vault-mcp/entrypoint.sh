#!/bin/sh
# Starts both vault-mcp processes in this one container: server.py (the MCP
# server Hermes talks to, :8000) and upload_web.py (the new browser upload
# page, :8081) -- see vault_core.py's top comment for why these are two
# processes instead of one. Both run in the background; `wait -n` returns
# as soon as EITHER one exits, and the `exit 1` after it makes the whole
# container die in that case -- so if either process crashes, k8s sees the
# container as failed and restarts it (the normal recovery path), instead
# of silently limping along with only half the functionality working.
set -e
python server.py &
python upload_web.py &
wait -n
exit 1
