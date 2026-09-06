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
expects at `/opt/data/.env` (empty here — no external LLM provider keys are
needed since Hermes is routed to our own llama-swap, and Honcho is
self-hosted via `honcho.json`'s `baseUrl`, not `HONCHO_API_KEY`).

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

## Order of operations for a fresh cluster

1. Run the two `kubectl create secret` blocks above (they also create the
   namespaces, so this can run before ArgoCD has synced anything).
2. Let ArgoCD sync `k8s/p2` and `k8s/p3` as normal — the Deployments come up
   referencing these Secrets by name.

If a Secret doesn't exist yet when ArgoCD syncs, the pod just sits in
`CreateContainerConfigError` until you create it — nothing fails
destructively, so the order isn't strict, just create the Secret before
expecting the pod to actually start.
