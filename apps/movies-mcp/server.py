"""movies-mcp: an HTTP MCP server that answers movie questions (new
releases, recommendations, "what did we talk about last time") for
Hermes -- with PERSISTENT memory via Honcho, unlike a one-off LLM call.
Hermes's own conversation memory doesn't carry into a tool call, and a
fresh delegate_task subagent starts blank every time (see the
"specialist agents" design discussion, 2026-09-27) -- this tool is the
first of what's meant to become a small family of these (movies/
medical/legal/...), each its own tiny pod so a bug in one specialist
can't take the others down, same reasoning as media-augment-mcp's own
split from media-gen-mcp (see that file's top-of-file comment).

HONCHO -- the actual persistent-memory backend. Reuses the SAME Honcho
deployment already running for Hermes's own memory (k8s/p2/honcho.yaml),
just in its own workspace ("aibox-specialists") so it never collides
with whatever workspace Hermes itself uses. Two Honcho PEERS, one
long-lived SESSION between them, created once and reused forever (not
re-created per call, get-or-create is a no-op once they exist):
  - "eldar-movies" -- Eldar's side of the conversation, scoped to this
    specialist only. A separate peer per domain (not one shared "eldar"
    peer) is what keeps Honcho's medical representation of Eldar from
    ever bleeding into what the movies peer knows -- confirmed via
    Honcho's own docs: a peer's representation is built only from
    messages it has actually witnessed, and this tool only ever writes
    into the "movies-eldar" session.
  - "movies-agent" -- this tool's own past replies, so Honcho's
    reasoning has both sides of the conversation, not just Eldar's
    questions.
Auth: aibox's Honcho deployment runs with AUTH_USE_AUTH=false (see
honcho-config ConfigMap) -- no API key needed today. HONCHO_API_KEY below
exists only so this still works if that's ever turned on.

API SHAPE -- confirmed 2026-09-27 against aibox's actual running pod
(GET /openapi.json on honcho-api), not just Honcho's hosted docs, which
show both v2 and v3 surfaces and didn't fully agree with each other:
  POST /v3/workspaces                                get-or-create workspace
  POST /v3/workspaces/{ws}/peers                     get-or-create peer
  POST /v3/workspaces/{ws}/sessions                  get-or-create session (peers can join in the same call)
  POST /v3/workspaces/{ws}/sessions/{sid}/messages   append messages (batch)
  POST /v3/workspaces/{ws}/peers/{pid}/chat          dialectic recall -- "what do we already know"

LIVE MOVIE DATA -- NOT wired in yet, deliberately. This first version
answers from the model's own knowledge (via LiteLLM's "auto" route --
the same one ordinary chat messages use, so it inherits the same
busy-GPU/cloud-fallback behavior already proven for media-gen jobs, see
media-gen-mcp's own router comment) plus whatever Honcho remembers from
earlier conversations. That's enough to prove the memory loop end to
end; a real web-search tool for new releases/showtimes is a planned
fast-follow, not forgotten.
"""
import logging
import os

import httpx
from mcp.server.mcpserver import Context, MCPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s movies-mcp %(levelname)s: %(message)s")
logger = logging.getLogger("movies-mcp")

# Same base_url/auth pattern every other tool in this repo uses to reach
# LiteLLM -- Bearer auth against LITELLM_MASTER_KEY, sourced from the
# same litellm-secret Secret.
LITELLM_URL = os.environ.get("LITELLM_URL", "http://litellm.ai-models.svc.cluster.local:4000/v1")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")
# "auto" on purpose, not a specific local model name -- see the top-of-file
# LIVE MOVIE DATA note. Naming a specific llama-swap model directly (the
# way edit_image's tag-expansion call does) would bypass LiteLLM's busy/
# cloud-fallback router entirely and risk racing an in-flight ComfyUI job
# for the GPU, the exact class of bug already fixed elsewhere in this
# repo -- this tool has no real need to force a local model, so it just
# rides the same router path normal chat already uses.
LITELLM_MODEL = "auto"

HONCHO_URL = os.environ.get("HONCHO_URL", "http://honcho-api.ai-memory.svc.cluster.local:8000")
HONCHO_API_KEY = os.environ.get("HONCHO_API_KEY", "")
HONCHO_WORKSPACE = "aibox-specialists"
HONCHO_SESSION = "movies-eldar"
HONCHO_HUMAN_PEER = "eldar-movies"
HONCHO_AGENT_PEER = "movies-agent"

SYSTEM_PROMPT = """You are the movies specialist inside Eldar's personal assistant setup. You answer questions about films -- recommendations, new releases, what to watch, plot/cast questions, and so on.

If a block called RECALLED CONTEXT is included below, it's what Honcho (this tool's memory layer) remembers from earlier conversations with Eldar about movies -- his stated preferences, what you already recommended, what he said he watched or liked. Use it naturally, the way a person who remembered the last conversation would, instead of asking Eldar to repeat himself.

You do not have live internet access yet -- if the question depends on very recent release dates, showtimes, or anything that could have changed after your training, say so plainly rather than guessing at specifics.

Answer directly and conversationally, a few sentences unless more detail is clearly wanted."""

mcp = MCPServer("movies")


def _honcho_headers():
    return {"Authorization": f"Bearer {HONCHO_API_KEY}"} if HONCHO_API_KEY else {}


async def _honcho_ensure(client):
    # Get-or-create is idempotent -- fine to call every request. Best
    # effort: if Honcho is down, the tool still answers, just without
    # memory for this one call, rather than failing the whole request.
    try:
        headers = _honcho_headers()
        await client.post(f"{HONCHO_URL}/v3/workspaces", json={"id": HONCHO_WORKSPACE}, headers=headers, timeout=15.0)
        for peer_id in (HONCHO_HUMAN_PEER, HONCHO_AGENT_PEER):
            await client.post(
                f"{HONCHO_URL}/v3/workspaces/{HONCHO_WORKSPACE}/peers", json={"id": peer_id}, headers=headers, timeout=15.0
            )
        await client.post(
            f"{HONCHO_URL}/v3/workspaces/{HONCHO_WORKSPACE}/sessions",
            json={"id": HONCHO_SESSION, "peers": {HONCHO_HUMAN_PEER: {}, HONCHO_AGENT_PEER: {}}},
            headers=headers,
            timeout=15.0,
        )
        return True
    except Exception as exc:  # noqa: BLE001 -- Honcho being down shouldn't block an answer
        logger.warning("Honcho setup failed (%s: %s) -- answering without memory this time", type(exc).__name__, exc)
        return False


async def _honcho_recall(client, question):
    # Asks Honcho's own reasoning agent "what do we already know", scoped
    # to just this session -- not a raw transcript dump. Sends e.g.:
    #   POST /v3/workspaces/aibox-specialists/peers/eldar-movies/chat
    #   {"query": "any good sci-fi lately?", "session_id": "movies-eldar"}
    # returns e.g. {"content": "Eldar asked about sci-fi last week and liked the Dune recommendation."}
    try:
        resp = await client.post(
            f"{HONCHO_URL}/v3/workspaces/{HONCHO_WORKSPACE}/peers/{HONCHO_HUMAN_PEER}/chat",
            json={"query": question, "session_id": HONCHO_SESSION},
            headers=_honcho_headers(),
            timeout=30.0,
        )
        resp.raise_for_status()
        return (resp.json() or {}).get("content") or ""
    except Exception as exc:  # noqa: BLE001 -- degrade to no recalled context, don't fail the answer
        logger.warning("Honcho recall failed (%s: %s)", type(exc).__name__, exc)
        return ""


async def _honcho_save_turn(client, question, answer):
    try:
        await client.post(
            f"{HONCHO_URL}/v3/workspaces/{HONCHO_WORKSPACE}/sessions/{HONCHO_SESSION}/messages",
            json={
                "messages": [
                    {"content": question, "peer_id": HONCHO_HUMAN_PEER},
                    {"content": answer, "peer_id": HONCHO_AGENT_PEER},
                ]
            },
            headers=_honcho_headers(),
            timeout=15.0,
        )
    except Exception as exc:  # noqa: BLE001 -- the answer already went out, don't fail on the save
        logger.warning("Honcho save failed (%s: %s) -- this turn won't be remembered", type(exc).__name__, exc)


async def _ask_llm(client, question, recalled):
    # sends e.g.:
    #   {"model": "auto", "messages": [{"role": "system", ...},
    #    {"role": "user", "content": "RECALLED CONTEXT:\n...\n\nQUESTION:\nany good sci-fi lately?"}]}
    # returns e.g. {"choices": [{"message": {"content": "..."}}]}
    user_content = question
    if recalled:
        user_content = f"RECALLED CONTEXT (from earlier conversations):\n{recalled}\n\nQUESTION:\n{question}"
    resp = await client.post(
        f"{LITELLM_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LITELLM_API_KEY}"} if LITELLM_API_KEY else {},
        json={
            "model": LITELLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.4,
            "max_tokens": 600,
        },
        timeout=90.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


@mcp.tool()
async def ask_movies(question: str, ctx: Context) -> str:
    """Answer a movie question -- recommendations, new releases, what to watch, plot/cast questions.

    Remembers past conversations: what you've asked before, what this
    tool already recommended, preferences you've mentioned. You don't
    need to repeat context from an earlier chat.

    Does not have live internet access yet, so very recent release dates
    or showtimes may not be reliable -- the answer will say so rather
    than guess.

    Args:
        question: The movie question, in plain language, exactly as asked.

    Returns:
        A direct, conversational answer.
    """
    if not question or not question.strip():
        raise ValueError("question must not be empty")

    async with httpx.AsyncClient() as client:
        await _honcho_ensure(client)
        recalled = await _honcho_recall(client, question)
        answer = await _ask_llm(client, question, recalled)
        await _honcho_save_turn(client, question, answer)
        return answer


if __name__ == "__main__":
    # streamable_http_path defaults to "/mcp" -- json_response deliberately
    # NOT set, same reasoning as every other server.py in this repo (see
    # media-gen-mcp's own comment on this call).
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
