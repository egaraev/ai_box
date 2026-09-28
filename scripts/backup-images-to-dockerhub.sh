#!/usr/bin/env bash
# Backs up every container image this platform depends on to Docker Hub,
# under egaraev/<name>:<tag> -- protects against a vendor deleting or
# breaking an image out from under us. This isn't hypothetical: it
# happened for real on 2026-09-28 (quay.io/minio/minio and quay.io/
# minio/mc both stopped allowing anonymous pulls, see
# k8s/p8/langfuse-minio.yaml's own top-of-file comment for the full
# story). All public repos, matching every other repo already on this
# Docker Hub account.
#
# Run this yourself whenever you want a fresh snapshot -- it's not on a
# schedule. `docker login` first (this script never touches credentials).
# Safe to re-run any time (e.g. after bumping a version in a k8s manifest
# -- just update the matching line below first).

DOCKERHUB_USER="egaraev"

# Pulled from a registry, then re-tagged and pushed. Format per line:
# "source_image|target_repo:tag" -- keep this in sync with the real
# image: lines across k8s/ and the FROM lines in apps/*/Dockerfile when
# either changes.
IMAGES=(
  "busybox:1.36|busybox:1.36"
  "cgr.dev/chainguard/minio:latest|chainguard-minio:latest"
  "cgr.dev/chainguard/minio-client:latest|chainguard-minio-client:latest"
  "clickhouse/clickhouse-server:24.8|clickhouse-server:24.8"
  "ghcr.io/berriai/litellm:main-latest|litellm:main-latest"
  "ghcr.io/mostlygeek/llama-swap:unified-cuda13|llama-swap:unified-cuda13"
  "ghcr.io/open-webui/open-webui:main|open-webui:main"
  "ghcr.io/plastic-labs/honcho:latest|honcho:latest"
  "grafana/grafana:11.4.0|grafana:11.4.0"
  "grafana/loki:3.7.6|loki:3.7.6"
  "grafana/promtail:3.3.0|promtail:3.3.0"
  "langfuse/langfuse-worker:3|langfuse-worker:3"
  "langfuse/langfuse:3|langfuse:3"
  "mmartial/comfyui-nvidia-docker:ubuntu24_cuda12.9-latest|comfyui-nvidia-docker:ubuntu24_cuda12.9-latest"
  # pinned by DIGEST, not tag, in k8s/p3/hermes-agent.yaml on purpose (see
  # its own comment on why) -- tagged here with the short digest so the
  # backup stays traceable to the exact pinned version, not just
  # whatever "latest" happens to mean later
  "nousresearch/hermes-agent@sha256:6305b1b16f945aa7201610d9e3ba69b9e26d57ee00693215cc8ff36c0b4c3616|hermes-agent:pinned-6305b1b1"
  "nvcr.io/nvidia/k8s/dcgm-exporter:4.6.0-4.8.3-distroless|dcgm-exporter:4.6.0-4.8.3-distroless"
  "pgvector/pgvector:pg15|pgvector:pg15"
  "prom/node-exporter:v1.12.1|node-exporter:v1.12.1"
  "prom/prometheus:v3.13.0|prometheus:v3.13.0"
  "python:3.13-slim|python:3.13-slim"
  "python:3.12-alpine3.24|python:3.12-alpine3.24"
  "python:3.12-slim|python:3.12-slim"
  "qdrant/qdrant:latest|qdrant:latest"
  "redis:8.2|redis:8.2"
  "registry.k8s.io/kube-state-metrics/kube-state-metrics:v2.19.1|kube-state-metrics:v2.19.1"
  "n8nio/n8n:2.38.7|n8n:2.38.7"
)

# Built locally from apps/*/Dockerfile, never pulled from a registry --
# no pull step, just tag+push whatever's already in the local image
# cache. Build each one first (docker build -t <name> apps/<dir>/) if
# it's not there -- this script won't build them for you.
LOCAL_IMAGES=(
  "aibox-n8n:latest|aibox-n8n:latest"
  "media-augment-mcp:latest|media-augment-mcp:latest"
  "media-gen-mcp:latest|media-gen-mcp:latest"
  "movies-mcp:latest|movies-mcp:latest"
  "vault-mcp:latest|vault-mcp:latest"
)

ok=()
failed=()

# pulls (unless local), tags, and pushes one image -- one bad image
# shouldn't stop the rest of the backup, so every step just records the
# failure and moves on rather than exiting the script
backup_one() {
  local src="$1" target="$2" is_local="$3"
  local dest="$DOCKERHUB_USER/$target"
  echo "=== $src -> $dest ==="
  if [ "$is_local" = "local" ]; then
    docker image inspect "$src" >/dev/null 2>&1 || {
      echo "MISSING LOCALLY: $src (build it first, this script doesn't build images)"
      failed+=("$src")
      return
    }
  else
    docker pull "$src" || { echo "PULL FAILED: $src"; failed+=("$src"); return; }
  fi
  docker tag "$src" "$dest" || { echo "TAG FAILED: $src"; failed+=("$src"); return; }
  docker push "$dest" || { echo "PUSH FAILED: $dest"; failed+=("$src"); return; }
  ok+=("$src")
}

for entry in "${IMAGES[@]}"; do
  backup_one "${entry%%|*}" "${entry##*|}" ""
done

for entry in "${LOCAL_IMAGES[@]}"; do
  backup_one "${entry%%|*}" "${entry##*|}" "local"
done

echo
echo "=== SUMMARY ==="
echo "${#ok[@]} succeeded, ${#failed[@]} failed"
if [ "${#failed[@]}" -gt 0 ]; then
  printf 'FAILED: %s\n' "${failed[@]}"
  exit 1
fi
