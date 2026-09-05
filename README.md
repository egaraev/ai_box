# aibox-platform

GitOps manifest repo for the aibox personal AI platform (k3s, single node).

ArgoCD watches this repo and auto-deploys on push. Manifests are organized by
deploy phase, matching the plan in `CLAUDEk3s.md`:

```
k8s/
├── bootstrap/    ArgoCD Application resources (the "app of apps" — one per phase,
│                 added as each phase is built out)
├── p0/           Cluster skeleton: namespaces
├── p1/           First working LLM: llama-server + llama-swap + Open WebUI
├── p2/           Memory layer: Qdrant + Honcho + MongoDB + MinIO
├── p3/           MCP bus + Hermes Agent
├── p4/           Telegram + session persistence
├── p5/           Voice pipeline
├── p6/           Specialized agents
├── p7/           Knowledge ingestion + WhatsApp
├── p8/           Istio + observability hardening
└── p9/           Unsloth-LoRA-DPO self-improvement
```

## Namespaces (created in p0)

| Namespace | Contents |
|-----------|----------|
| `ai-models` | llama-server + llama-swap + LiteLLM proxy + arch-router-1.5b |
| `ai-memory` | Honcho, Qdrant, MongoDB, ArangoDB, Redis, MinIO |
| `ai-mcp` | All MCP server pods |
| `ai-agents` | Hermes + specialized agent StatefulSets |
| `ai-integration` | N8N, LangFuse, LangServe wrappers |
| `ai-ingest` | Ingestion pipeline jobs (Docling, MarkItDown) |
| `ai-observe` | Prometheus, Grafana, Jaeger, LangFuse server |
| `ai-gateway` | Traefik ingress / external access |

## Bootstrapping

1. Install k3s (traefik disabled — MetalLB + ingress-nginx used instead) and ArgoCD.
2. Push this repo to a git remote ArgoCD can reach (a private GitHub repo is
   simplest for a single-node personal cluster — avoids mounting the host
   filesystem into the argocd-repo-server pod).
3. Set `repoURL` in `k8s/bootstrap/root-app.yaml` to that remote.
4. `kubectl apply -f k8s/bootstrap/root-app.yaml` — this is the one manifest
   you apply by hand; everything else flows through ArgoCD from here on.
5. Confirm: `kubectl get ns` shows all 8 namespaces, and the ArgoCD UI shows
   the `aibox-p0` Application as Synced/Healthy.
