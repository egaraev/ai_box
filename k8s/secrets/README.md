# Real secrets — never committed

Everything else in `k8s/secrets/` is git-ignored (see the repo root
`.gitignore`) and never leaves aibox. ArgoCD's Applications watch
`k8s/p0`, `k8s/p1`, `k8s/p2`, `k8s/p3`, ... only — this directory isn't one
of them, so `prune: true` never touches whatever you create here, and there
is nothing for GitHub's push protection to ever see.

The rule going forward: a committed manifest references a Secret **by
name** (`secretKeyRef`, `envFrom.secretRef`, a `secret.secretName` volume);
the Secret **object itself** — the one with the real value in it — gets
created straight on the cluster with `kubectl`, using the commands below.
Run each command once; re-run it (values will differ each time, since
they're freshly random) whenever you want to rotate.

## postgres-secret (namespace: ai-memory)

Backs `POSTGRES_PASSWORD` in `k8s/p2/postgres.yaml` and `DB_CONNECTION_URI`
in `k8s/p2/honcho.yaml` — both reference this one Secret now instead of a
literal `postgres/postgres` in the YAML. `POSTGRES_HOST_AUTH_METHOD: trust`
is still set (see the comment in postgres.yaml), so Postgres itself doesn't
actually check this password — this is about keeping the string out of git,
not about the password mattering yet.

```sh
PG_PASS=$(openssl rand -hex 20)
kubectl create namespace ai-memory --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic postgres-secret \
  --namespace ai-memory \
  --from-literal=password="$PG_PASS" \
  --from-literal=DB_CONNECTION_URI="postgresql+psycopg://postgres:${PG_PASS}@postgres:5432/postgres"
```

Note the two keys are used two different ways: `password` is read via an
explicit `secretKeyRef` in `postgres.yaml` (key names there can be anything),
while `DB_CONNECTION_URI` is picked up by `honcho.yaml` via a whole-secret
`envFrom` (which needs the key to already be a legal env var name, hence the
uppercase). `envFrom` also incidentally injects a stray `password=...` env
var into the Honcho containers alongside it — harmless, nothing reads it,
just don't be surprised to see it in `kubectl exec ... env`.

Rotating: after re-running the command above with a new `PG_PASS`, restart
both consumers so they pick it up —
`kubectl delete pod -n ai-memory -l app=postgres` and
`kubectl delete pod -n ai-memory -l app=honcho-api -l app=honcho-deriver`
(same kubelet-ConfigMap/Secret-cache timing as any other config change —
see the `aibox-config-restart-verify` skill).

## hermes-env (namespace: ai-agents)

Backs the API-server bearer key and the web dashboard's basic-auth
credentials in `k8s/p3/hermes-agent.yaml`, plus the `.env` file Hermes
expects at `/opt/data/.env` (empty here — Honcho is self-hosted via
`honcho.json`'s `baseUrl`, not `HONCHO_API_KEY`, so no key is needed there.
Hermes's actual LLM auth now comes from `litellm-secret` below, not this
Secret — see its `OPENAI_API_KEY` env var in `hermes-agent.yaml`).

```sh
API_KEY=$(openssl rand -hex 32)
DASH_PASS=$(openssl rand -hex 16)
kubectl create namespace ai-agents --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic hermes-env \
  --namespace ai-agents \
  --from-literal=API_SERVER_KEY="$API_KEY" \
  --from-literal=HERMES_DASHBOARD_BASIC_AUTH_USERNAME="eldar" \
  --from-literal=HERMES_DASHBOARD_BASIC_AUTH_PASSWORD="$DASH_PASS" \
  --from-literal=.env="# seeded by kubectl, see k8s/secrets/README.md"
```

Rotating: re-run with new random values, then
`kubectl delete pod -n ai-agents -l app=hermes-agent`.

### Adding Telegram

Two more keys in this same Secret turn on the Telegram gateway --
`TELEGRAM_BOT_TOKEN` (from @BotFather) and `TELEGRAM_ALLOWED_USERS` (your
numeric Telegram user ID, from @userinfobot -- comma-separated for more than
one person). No manifest change needed: the Deployment already does
`envFrom: secretRef: name: hermes-env`, and Hermes reads both as plain env
vars, so adding the keys here is the whole change.

```sh
kubectl -n ai-agents patch secret hermes-env --type merge -p \
  '{"stringData":{"TELEGRAM_BOT_TOKEN":"<token-from-botfather>","TELEGRAM_ALLOWED_USERS":"<your-numeric-id>"}}'
kubectl delete pod -n ai-agents -l app=hermes-agent
```

`kubectl patch --type merge` only touches the keys you name -- the API
server key and dashboard credentials already in the Secret are untouched.

## litellm-secret (namespaces: ai-models, ai-gateway, ai-agents, ai-memory, ai-mcp, ai-integration)

Backs LiteLLM's `general_settings.master_key` (`k8s/p1/litellm.yaml`) -- the
one bearer key every internal caller (Open WebUI, Hermes, Honcho,
media-gen-mcp's `_expand_character_references`, and now n8n's baked-in
`vault_sync.py`/`gmail_sync.py`/`telegram_sync.py`, see
`apps/media-gen-mcp/server.py` and `k8s/n8n/n8n-deployment.yaml`) needs to
talk to LiteLLM instead of hitting llama-swap directly. Kubernetes Secrets
don't cross namespaces, and each of those six callers lives in a different
one, so the *same* value has to exist as a same-named Secret in all six --
there's no single shared object to point at. (media-gen-mcp's own
Deployment/Service resources live in `k8s/media-gen/`, and n8n's in
`k8s/n8n/`, neither one of the `k8s/p0`..`p3` ArgoCD-synced directories,
but `ai-mcp`/`ai-integration` still each need their own copy of this
Secret since that's where the respective pod actually runs.)

```sh
LITELLM_KEY=$(openssl rand -hex 32)
for ns in ai-models ai-gateway ai-agents ai-memory ai-mcp ai-integration; do
  kubectl create namespace "$ns" --dry-run=client -o yaml | kubectl apply -f -
  kubectl create secret generic litellm-secret \
    --namespace "$ns" \
    --from-literal=master_key="$LITELLM_KEY"
done
```

Rotating: re-run the loop above with a fresh `LITELLM_KEY` (same value goes
into all six namespaces -- a partial rotation just means some callers get
rejected with a stale key), then restart everything that reads it:

```sh
kubectl delete pod -n ai-models -l app=litellm
kubectl delete pod -n ai-gateway -l app=open-webui
kubectl delete pod -n ai-agents -l app=hermes-agent
kubectl delete pod -n ai-memory -l app=honcho-api -l app=honcho-deriver
kubectl delete pod -n ai-mcp -l app=media-gen-mcp
kubectl delete pod -n ai-integration -l app=n8n
```

## openrouter-secret (namespace: ai-models)

Backs `OPENROUTER_API_KEY` in `k8s/p1/litellm.yaml`'s litellm Deployment --
used by the `openrouter-fallback` model_list entry (the uniform cloud
fallback `custom_router_hook.py` redirects to while the media-gen busy flag
is set -- see that file's own top comment). Only one namespace needs it,
unlike `litellm-secret` above -- OpenRouter auth is entirely internal to
the litellm pod, nothing else calls OpenRouter directly.

The real key already lives in a local `.env` file at the repo root on
aibox (`~/repos/aibox-platform/.env`, git-ignored) -- this reads it
straight from there so the value never has to be typed into a chat with
the assistant or committed anywhere:

```sh
OPENROUTER_KEY=$(grep '^OPENROUTER_API_KEY=' ~/repos/aibox-platform/.env | head -1 | cut -d= -f2- | tr -d '"')
kubectl create namespace ai-models --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic openrouter-secret \
  --namespace ai-models \
  --from-literal=OPENROUTER_API_KEY="$OPENROUTER_KEY"
```

Rotating: update the key in `.env` first, then re-run the block above
(`kubectl create secret` fails if the Secret already exists -- use
`kubectl create secret ... --dry-run=client -o yaml | kubectl apply -f -`
instead, or `kubectl delete secret openrouter-secret -n ai-models` first),
then `kubectl delete pod -n ai-models -l app=litellm`.

## grafana-secret (namespace: ai-observe)

Backs `GF_SECURITY_ADMIN_USER`/`GF_SECURITY_ADMIN_PASSWORD` in `k8s/p8/grafana.yaml` -- 2026-09-17 observability stack (Prometheus/Grafana/Loki/LangFuse, see `k8s/p8/`).

```sh
GF_PASS=$(openssl rand -hex 16)
kubectl create namespace ai-observe --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic grafana-secret \
  --namespace ai-observe \
  --from-literal=admin-user="eldar" \
  --from-literal=admin-password="$GF_PASS"
```

Rotating: re-run with a new `GF_PASS`, then `kubectl delete pod -n ai-observe -l app=grafana`.

## LangFuse secrets (namespace: ai-observe) -- 2026-09-17

LangFuse (`k8s/p8/langfuse.yaml`, `-clickhouse.yaml`, `-minio.yaml`) reuses the
EXISTING shared Postgres and Redis (`k8s/p2/postgres.yaml`/`redis.yaml`,
namespace ai-memory) rather than deploying new instances -- same
"no infra weight without a real reason" call as everywhere else in this
repo. ClickHouse and Minio are new dedicated deployments (nothing else on
this cluster needs either).

**1. One-time: create LangFuse's own database on the existing shared Postgres**
(can't use the `postgres-init` ConfigMap for this -- that only ever runs
against a fresh, empty data volume, and this Postgres already holds
Honcho's data):

```sh
kubectl exec -n ai-memory deploy/postgres -- psql -U postgres -c "CREATE DATABASE langfuse;"
```

**2. Add a `LANGFUSE_DATABASE_URL` key to the existing `postgres-secret`**
(patch, not recreate -- preserves the `password`/`DB_CONNECTION_URI` keys
Honcho already depends on):

```sh
PG_PASS=$(kubectl get secret postgres-secret -n ai-memory -o jsonpath='{.data.password}' | base64 -d)
kubectl -n ai-memory patch secret postgres-secret --type merge -p \
  "{\"stringData\":{\"LANGFUSE_DATABASE_URL\":\"postgresql://postgres:${PG_PASS}@postgres.ai-memory.svc.cluster.local:5432/langfuse\"}}"
```

**2b. Copy it into a `langfuse-postgres-secret` in `ai-observe`** -- FIXED
2026-09-17, first real deploy: `langfuse-web`/`langfuse-worker` (namespace
`ai-observe`) originally read `DATABASE_URL` straight from `postgres-secret`
above, but that Secret lives in `ai-memory` and Kubernetes Secrets can't be
read across namespaces -- confirmed live ("secret postgres-secret not
found"). Same cross-namespace problem the `litellm-secret` section above
already solves by duplicating into every namespace that needs it; this is
the same fix, just for one key instead of the whole master-key secret:

```sh
LANGFUSE_DB_URL=$(kubectl get secret postgres-secret -n ai-memory -o jsonpath='{.data.LANGFUSE_DATABASE_URL}' | base64 -d)
kubectl create secret generic langfuse-postgres-secret \
  --namespace ai-observe \
  --from-literal=DATABASE_URL="$LANGFUSE_DB_URL"
```

Rotating: if `postgres-secret`'s password ever changes, re-run step 2 above
first, then this block again, then
`kubectl delete pod -n ai-observe -l app=langfuse-web -l app=langfuse-worker`.

**3. langfuse-secret** -- LangFuse's own app-level secrets (session signing, API-key hashing, at-rest encryption). Real random values, generated once and never rotated casually (rotating `ENCRYPTION_KEY` breaks decryption of anything already encrypted with the old one):

```sh
kubectl create namespace ai-observe --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic langfuse-secret \
  --namespace ai-observe \
  --from-literal=NEXTAUTH_SECRET="$(openssl rand -hex 32)" \
  --from-literal=SALT="$(openssl rand -hex 32)" \
  --from-literal=ENCRYPTION_KEY="$(openssl rand -hex 32)"
```

**4. langfuse-clickhouse-secret**:

```sh
kubectl create secret generic langfuse-clickhouse-secret \
  --namespace ai-observe \
  --from-literal=CLICKHOUSE_USER="langfuse" \
  --from-literal=CLICKHOUSE_PASSWORD="$(openssl rand -hex 20)"
```

**5. langfuse-minio-secret**:

```sh
kubectl create secret generic langfuse-minio-secret \
  --namespace ai-observe \
  --from-literal=MINIO_ROOT_USER="langfuse" \
  --from-literal=MINIO_ROOT_PASSWORD="$(openssl rand -hex 20)"
```

Rotating any of 3-5: re-run with fresh values, then
`kubectl delete pod -n ai-observe -l app=langfuse-web -l app=langfuse-worker`
(clickhouse/minio pods too if their own secret changed).

## litellm-langfuse-secret (namespace: ai-models) -- 2026-09-17, MANUAL, post-login

Backs `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` in `k8s/p1/litellm.yaml`'s
litellm Deployment (see that file's `success_callback: ["langfuse"]`
comment). Unlike every other secret above, these values are NOT
self-generated random strings -- they're issued by LangFuse's own web UI,
so this secret can only be created AFTER LangFuse is deployed and Eldar has
logged in once:

1. Open `http://<aibox-lan-ip>:30350`, sign up (first account becomes the
   only org member -- single-user homelab, no invite flow needed), create
   one project (e.g. "aibox").
2. Project Settings -> API Keys -> Create new API key. Copy the Public Key
   (`pk-lf-...`) and Secret Key (`sk-lf-...`) shown once.
3. Create the Secret:

```sh
kubectl create secret generic litellm-langfuse-secret \
  --namespace ai-models \
  --from-literal=LANGFUSE_PUBLIC_KEY="pk-lf-..." \
  --from-literal=LANGFUSE_SECRET_KEY="sk-lf-..."
kubectl delete pod -n ai-models -l app=litellm
```

Until this secret exists, the litellm pod sits in
`CreateContainerConfigError` -- same as any other missing-Secret case (see
"Order of operations" below), so deploying `k8s/p1` and `k8s/p8` together
before this manual step is fine, nothing fails destructively.

## Order of operations for a fresh cluster

1. Run the three `kubectl create secret` blocks above (they also create the
   namespaces, so this can run before ArgoCD has synced anything).
2. Let ArgoCD sync `k8s/p1`, `k8s/p2`, and `k8s/p3` as normal — the
   Deployments come up referencing these Secrets by name.

If a Secret doesn't exist yet when ArgoCD syncs, the pod just sits in
`CreateContainerConfigError` until you create it — nothing fails
destructively, so the order isn't strict, just create the Secret before
expecting the pod to actually start.
