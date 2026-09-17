#!/bin/bash
# Starts both vault-mcp processes in this one container: server.py (the MCP
# server Hermes talks to, :8000) and upload_web.py (the new browser upload
# page, :8081) -- see vault_core.py's top comment for why these are two
# processes instead of one. Both run in the background; `wait -n` returns
# as soon as EITHER one exits, and the `exit 1` after it makes the whole
# container die in that case -- so if either process crashes, k8s sees the
# container as failed and restarts it (the normal recovery path), instead
# of silently limping along with only half the functionality working.
#
# MUST be bash, not sh: `wait -n` is a bash-only feature (bash 4.3+). The
# original version of this script used #!/bin/sh, which on this image
# (python:3.12-slim, Debian-based) resolves to dash -- dash's wait builtin
# doesn't understand -n at all and errors immediately ("wait: Illegal
# option -n"), which is why the very first deploy crash-looped instantly.
# python:3.12-slim ships /bin/bash by default (it's a base Debian package),
# so no extra install is needed for this fix.
set -e
python server.py &
python upload_web.py &
wait -n
exit 1
