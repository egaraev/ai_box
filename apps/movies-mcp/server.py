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
    specialist only.
  - "movies-agent" -- this tool's own past replies, so Honcho's
    reasoning has both sides of the conversation, not just Eldar's
    questions.
Auth: aibox's Honcho deployment runs with AUTH_USE_AUTH=false (see
honcho-config ConfigMap) -- no API key needed today. HONCHO_API_KEY below
exists only so this still works if that's ever turned on.

WATCH HISTORY (added 2026-09-27, replaces the earlier "not wired in
yet" version of this file): a plain YAML file at WATCH_HISTORY_PATH
(five lists: interests, completed, on_hold, suggested_not_yet_watched,
anticipated), mounted from the Obsidian vault -- see
movies-mcp-deployment.yaml's own top comment for the hostPath history
and the directory-vs-file mount bug that was fixed there. IMPORTANT
HISTORY: an earlier version of movies-mcp-deployment.yaml and Hermes's
own SOUL.md were written assuming a SEPARATE update_watch_history tool
already existed here -- it never did (confirmed 2026-09-27 via git log,
a pod-code grep, and a line-count diff -- this file only ever had
ask_movies). That's the actual reason nothing ever got saved, not a
routing or prompt problem. Rather than add that second tool now, the
write capability is merged straight into ask_movies below: Hermes only
ever has one movie tool to pick, and the code -- not Hermes's own
judgment -- decides whether a given question describes a change worth
saving. See ask_movies's own comment for how.

check_anticipated (also added 2026-09-27, also never existed before
despite TMDB_API_KEY already being wired into the Deployment): live
TMDB lookups for "is X back yet" style questions about anything on the
anticipated list.

API SHAPE -- confirmed 2026-09-27 against aibox's actual running pod
(GET /openapi.json on honcho-api), not just Honcho's hosted docs, which
show both v2 and v3 surfaces and didn't fully agree with each other:
  POST /v3/workspaces                                get-or-create workspace
  POST /v3/workspaces/{ws}/peers                     get-or-create peer
  POST /v3/workspaces/{ws}/sessions                  get-or-create session (peers can join in the same call)
  POST /v3/workspaces/{ws}/sessions/{sid}/messages   append messages (batch)
  POST /v3/workspaces/{ws}/peers/{pid}/chat          dialectic recall -- "what do we already know"
"""
import logging
import os
import re

import httpx
import yaml
from mcp.server.mcpserver import Context, MCPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s movies-mcp %(levelname)s: %(message)s")
logger = logging.getLogger("movies-mcp")

# Same base_url/auth pattern every other tool in this repo uses to reach
# LiteLLM -- Bearer auth against LITELLM_MASTER_KEY, sourced from the
# same litellm-secret Secret.
LITELLM_URL = os.environ.get("LITELLM_URL", "http://litellm.ai-models.svc.cluster.local:4000/v1")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")
# "auto" on purpose, not a specific local model name -- rides LiteLLM's
# own busy/cloud-fallback router instead of racing an in-flight ComfyUI
# job for the GPU the way naming a specific llama-swap model would.
LITELLM_MODEL = "auto"

HONCHO_URL = os.environ.get("HONCHO_URL", "http://honcho-api.ai-memory.svc.cluster.local:8000")
HONCHO_API_KEY = os.environ.get("HONCHO_API_KEY", "")
HONCHO_WORKSPACE = "aibox-specialists"
HONCHO_SESSION = "movies-eldar"
HONCHO_HUMAN_PEER = "eldar-movies"
HONCHO_AGENT_PEER = "movies-agent"

WATCH_HISTORY_PATH = os.environ.get("WATCH_HISTORY_PATH", "/app/watch-history.yaml")
# all four lists a title can appear in. "interests" is a separate freeform
# list, not managed by the update logic below.
STATUS_LISTS = ("completed", "on_hold", "suggested_not_yet_watched", "anticipated")
# these three ARE mutually exclusive -- a title is either finished, paused,
# or queued, never more than one at once. "anticipated" is deliberately NOT
# in this group -- see _apply_watch_history_update's comment for why.
PROGRESS_LISTS = ("completed", "on_hold", "suggested_not_yet_watched")

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
TMDB_URL = "https://api.themoviedb.org/3"

# strips a trailing "season 3" / "s3" / "part 2" so the same show doesn't
# end up as two different-looking titles -- confirmed live 2026-09-27:
# "Silo season 3" (completed) and "Silo" (anticipated) were treated as
# unrelated entries because matching was exact-string. Applied once, in
# _apply_watch_history_update, so both comparison and what actually gets
# stored use the same clean title.
SEASON_SUFFIX_RE = re.compile(r"\s+(season\s*\d+|s\d+|part\s*\d+)$", re.IGNORECASE)


def _normalize_title(title):
    return SEASON_SUFFIX_RE.sub("", title.strip()).strip()

# The model is told to use one of the exact STATUS_LISTS words, but confirmed
# live 2026-09-27 (Fringe update) that it sometimes substitutes a natural
# synonym instead ("dropped" instead of "on_hold") -- that silently failed
# the old strict-match parsing (status came back None, update got dropped
# with no confirmation). Normalize instead of trusting the model to always
# hit the exact enum string.
STATUS_SYNONYMS = {
    "completed": "completed", "watched": "completed", "finished": "completed", "done": "completed",
    "on_hold": "on_hold", "onhold": "on_hold", "hold": "on_hold", "dropped": "on_hold",
    "stopped": "on_hold", "paused": "on_hold", "quit": "on_hold", "abandoned": "on_hold", "gave_up": "on_hold",
    "suggested_not_yet_watched": "suggested_not_yet_watched", "suggested": "suggested_not_yet_watched",
    "recommended": "suggested_not_yet_watched", "to_watch": "suggested_not_yet_watched",
    "anticipated": "anticipated", "waiting": "anticipated", "upcoming": "anticipated", "anticipating": "anticipated",
}


def _normalize_status(value):
    key = value.strip().lower().replace(" ", "_").replace("-", "_")
    return STATUS_SYNONYMS.get(key)

# The model is told to always answer in this exact ANSWER/UPDATE/... line
# format instead of JSON -- small local models (this runs on LiteLLM's
# "auto" route, which can land on an 8B model) are unreliable at strict
# JSON but do fine at "one field per line", and _parse_llm_response below
# just splits on lines rather than needing a real JSON parser.
SYSTEM_PROMPT = """You are the movies specialist inside Eldar's personal assistant setup. You answer questions about films and shows -- recommendations, new releases, what to watch, plot/cast questions -- AND you record changes to Eldar's watch history when he mentions one.

CURRENT WATCH HISTORY is included below -- what Eldar has already completed, put on hold, been suggested, or is anticipating.

If a block called RECALLED CONTEXT is included, it's what Honcho (this tool's memory layer) remembers from earlier conversations with Eldar about movies -- his stated preferences, what you already recommended, what he said he watched or liked. Use it naturally, instead of asking Eldar to repeat himself.

You do not have live internet access for most questions -- if something depends on very recent release dates or showtimes, say so plainly rather than guessing. (A separate tool handles live TMDB lookups for anticipated titles -- you don't need to guess at those either.)

First, decide: does this message describe a CHANGE to the watch history (something watched, dropped, stopped mid-way, or newly anticipated) -- or is it a question / recommendation request? "I finished X, loved it", "mark X as watched", "I stopped watching X at season 2", "I'm waiting for the next season of X" are changes. "What should I watch", "what happens in the finale of X", "any good sci-fi lately" are not. A message can describe two changes at once ("I finished season 3, waiting for season 4") -- "anticipated" is independent of completed/on_hold/suggested_not_yet_watched (finishing what's out now doesn't cancel anticipating what's next), but you can only record ONE status per reply. If a message clearly describes both, record the progress change (completed/on_hold/suggested_not_yet_watched) this time and mention in ANSWER that it'll need to be told separately to also mark it anticipated.

Always reply in EXACTLY this format, one field per line, nothing before or after it:
ANSWER: <your natural, conversational answer, a few sentences unless more is clearly wanted>
UPDATE: yes or no
TITLE: <the bare show/movie title only -- no "season 3" or episode numbers appended, those go in NOTE or STOPPED_AT instead -- or none if UPDATE is no>
STATUS: must be exactly one of these four words -- completed, on_hold, suggested_not_yet_watched, anticipated -- or none. Use on_hold for anything dropped, stopped, paused, or quit -- not a different word.
LIKED: <yes, no, or unknown>
STOPPED_AT: <where he stopped, e.g. "season 2 episode 4" -- or none>
REASON: <why he stopped/dropped it, if he said -- or none>
NOTE: <any other detail worth keeping -- or none>"""

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


def _load_watch_history():
    # missing file or empty file both come back as the five empty lists,
    # same shape as the seed file movies-mcp-deployment.yaml documents
    try:
        with open(WATCH_HISTORY_PATH) as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        data = {}
    for key in ("interests",) + STATUS_LISTS:
        data.setdefault(key, [])
    return data


def _save_watch_history(data):
    # atomic write -- temp file next to the real one, then os.replace()
    # onto it, so a crash mid-write never corrupts the file. This is why
    # the k8s volume has to be a DIRECTORY mount, not a single-file one --
    # see movies-mcp-deployment.yaml's own comment on the EBUSY bug that
    # caused before that fix.
    tmp_path = f"{WATCH_HISTORY_PATH}.tmp"
    with open(tmp_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    os.replace(tmp_path, WATCH_HISTORY_PATH)


def _watch_history_text(data):
    # turns the loaded dict into a short plain-text block for the LLM
    # prompt, e.g.:
    #   COMPLETED: Dune Part Two (liked); Oppenheimer
    #   ON HOLD: The Bear (stopped at season 2 episode 3, got busy)
    #   SUGGESTED: Poor Things
    #   ANTICIPATED: Silo
    lines = []
    if data["completed"]:
        parts = []
        for item in data["completed"]:
            title = item.get("title", "?")
            if item.get("liked") is True:
                parts.append(f"{title} (liked)")
            elif item.get("liked") is False:
                parts.append(f"{title} (didn't like)")
            else:
                parts.append(title)
        lines.append("COMPLETED: " + "; ".join(parts))
    if data["on_hold"]:
        parts = []
        for item in data["on_hold"]:
            bit = item.get("title", "?")
            extra = [item[k] for k in ("stopped_at", "reason") if item.get(k)]
            if extra:
                bit += f" ({', '.join(extra)})"
            parts.append(bit)
        lines.append("ON HOLD: " + "; ".join(parts))
    if data["suggested_not_yet_watched"]:
        lines.append("SUGGESTED: " + "; ".join(item.get("title", "?") for item in data["suggested_not_yet_watched"]))
    if data["anticipated"]:
        lines.append("ANTICIPATED: " + "; ".join(item.get("title", "?") for item in data["anticipated"]))
    return "\n".join(lines) if lines else "(empty -- nothing recorded yet)"


def _remove_from_progress_lists(data, title_norm):
    # the three PROGRESS_LISTS are mutually exclusive -- moving a title to
    # one drops any stale copy from the other two
    for key in PROGRESS_LISTS:
        data[key] = [item for item in data[key] if _normalize_title(item.get("title", "")).lower() != title_norm.lower()]


def _remove_from_anticipated(data, title_norm):
    data["anticipated"] = [item for item in data["anticipated"] if _normalize_title(item.get("title", "")).lower() != title_norm.lower()]


def _apply_watch_history_update(data, title, status, liked, stopped_at, reason, note):
    # confirmed live 2026-09-27: completing what's out now and still
    # anticipating the next season are both true at once for the same
    # show -- so a progress update (completed/on_hold/suggested) only
    # dedupes against the OTHER progress statuses, never touches
    # anticipated, and vice versa. Cross-clearing them was the original
    # bug (see PROGRESS_LISTS' own comment).
    title_norm = _normalize_title(title)
    if status == "anticipated":
        _remove_from_anticipated(data, title_norm)
    else:
        _remove_from_progress_lists(data, title_norm)
    entry = {"title": title_norm}
    if liked is not None:
        entry["liked"] = liked
    if stopped_at:
        entry["stopped_at"] = stopped_at
    if reason:
        entry["reason"] = reason
    if note:
        entry["note"] = note
    data[status].append(entry)
    _save_watch_history(data)


def _parse_llm_response(raw):
    # splits the model's ANSWER:/UPDATE:/... reply into fields -- see
    # SYSTEM_PROMPT for the format it's told to follow. Falls back to
    # treating the whole raw reply as the answer, with no update, if the
    # model didn't follow the format (rather than guessing at fields from
    # free text).
    fields = {"answer": None, "update": False, "title": None, "status": None,
              "liked": None, "stopped_at": None, "reason": None, "note": None}
    for line in raw.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().upper()
        value = value.strip()
        none_or = lambda v: None if v.lower() == "none" else v  # noqa: E731 -- tiny one-off, a def would be more to read
        if key == "ANSWER":
            fields["answer"] = value
        elif key == "UPDATE":
            fields["update"] = value.lower().startswith("y")
        elif key == "TITLE":
            fields["title"] = none_or(value)
        elif key == "STATUS":
            fields["status"] = _normalize_status(value)
        elif key == "LIKED":
            fields["liked"] = True if value.lower().startswith("y") else False if value.lower().startswith("n") else None
        elif key == "STOPPED_AT":
            fields["stopped_at"] = none_or(value)
        elif key == "REASON":
            fields["reason"] = none_or(value)
        elif key == "NOTE":
            fields["note"] = none_or(value)
    if not fields["answer"]:
        fields["answer"] = raw.strip()
        fields["update"] = False
    return fields


async def _ask_llm(client, question, watch_history_text, recalled):
    # sends e.g.:
    #   {"model": "auto", "messages": [{"role": "system", ...},
    #    {"role": "user", "content": "CURRENT WATCH HISTORY:\n...\n\nQUESTION:\nI stopped watching Foundation at season 2"}]}
    # returns e.g. {"choices": [{"message": {"content":
    #   "ANSWER: Got it, marked Foundation as on hold.\nUPDATE: yes\nTITLE: Foundation\n
    #    STATUS: on_hold\nLIKED: unknown\nSTOPPED_AT: season 2\nREASON: none\nNOTE: none"}}]}
    user_content = f"CURRENT WATCH HISTORY:\n{watch_history_text}\n\n"
    if recalled:
        user_content += f"RECALLED CONTEXT (from earlier conversations):\n{recalled}\n\n"
    user_content += f"QUESTION:\n{question}"
    resp = await client.post(
        f"{LITELLM_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LITELLM_API_KEY}"} if LITELLM_API_KEY else {},
        json={
            "model": LITELLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.3,
            "max_tokens": 500,
        },
        timeout=90.0,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    return _parse_llm_response(raw)


async def _tmdb_search(client, title):
    # GET https://api.themoviedb.org/3/search/multi?api_key=...&query=Silo
    # returns e.g. {"results": [{"id": 91363, "media_type": "tv", "name": "Silo", ...}, ...]}
    resp = await client.get(f"{TMDB_URL}/search/multi", params={"api_key": TMDB_API_KEY, "query": title}, timeout=15.0)
    resp.raise_for_status()
    for result in resp.json().get("results") or []:
        if result.get("media_type") in ("tv", "movie"):
            return result
    return None


async def _tmdb_tv_status(client, tmdb_id):
    # GET https://api.themoviedb.org/3/tv/91363?api_key=...
    # returns e.g. {"status": "Returning Series", "next_episode_to_air":
    #   {"air_date": "2027-01-15", "season_number": 3, "episode_number": 1}}
    resp = await client.get(f"{TMDB_URL}/tv/{tmdb_id}", params={"api_key": TMDB_API_KEY}, timeout=15.0)
    resp.raise_for_status()
    data = resp.json()
    status = data.get("status", "unknown")
    next_ep = data.get("next_episode_to_air")
    if next_ep:
        return f"{status} -- next episode S{next_ep.get('season_number')}E{next_ep.get('episode_number')} on {next_ep.get('air_date')}"
    return status


async def _tmdb_movie_status(client, tmdb_id):
    # GET https://api.themoviedb.org/3/movie/{id}?api_key=...
    # returns e.g. {"status": "Post Production", "release_date": "2027-03-01"}
    resp = await client.get(f"{TMDB_URL}/movie/{tmdb_id}", params={"api_key": TMDB_API_KEY}, timeout=15.0)
    resp.raise_for_status()
    data = resp.json()
    status = data.get("status", "unknown")
    release_date = data.get("release_date")
    return f"{status} -- release date {release_date}" if release_date else status


async def _tmdb_status_for(client, tmdb_id, media_type):
    return await _tmdb_tv_status(client, tmdb_id) if media_type == "tv" else await _tmdb_movie_status(client, tmdb_id)


async def _check_one_anticipated(client, entry):
    # entry is one item from watch_history["anticipated"], e.g.
    #   {"title": "Silo", "tmdb_id": 91363, "media_type": "tv"}
    # -- mutates entry in place to cache tmdb_id/media_type on first lookup,
    # so later checks don't need to re-search TMDB by title every time.
    title = entry.get("title", "?")
    if not entry.get("tmdb_id"):
        found = await _tmdb_search(client, title)
        if not found:
            return f"{title}: not found on TMDB"
        entry["tmdb_id"] = found["id"]
        entry["media_type"] = found["media_type"]
    try:
        status = await _tmdb_status_for(client, entry["tmdb_id"], entry.get("media_type", "tv"))
    except Exception as exc:  # noqa: BLE001 -- one bad TMDB lookup shouldn't blank out the whole answer
        return f"{title}: TMDB lookup failed ({type(exc).__name__})"
    return f"{title}: {status}"


@mcp.tool()
async def ask_movies(question: str, ctx: Context) -> str:
    """Ask about movies/shows, or tell it about a change to your watch history -- one tool for both.

    Handles recommendations, new releases, what to watch, plot/cast
    questions -- and also records changes: something you finished,
    dropped, stopped mid-way, or are now anticipating. The tool itself
    decides which kind of message this is and saves a change when one is
    described; you don't need a separate "update" call. Remembers past
    conversations via Honcho, so you don't need to repeat context from an
    earlier chat.

    For live release/season status on something you're anticipating
    ("is Silo back yet"), use check_anticipated instead -- this tool
    doesn't have internet access for that.

    Args:
        question: The movie question or statement, in plain language, exactly as asked.

    Returns:
        A direct, conversational answer. If the question described a
        change, the answer also confirms what was saved (e.g. "(saved:
        Foundation -> on_hold)") -- if that confirmation isn't there, it
        wasn't saved.
    """
    if not question or not question.strip():
        raise ValueError("question must not be empty")

    watch_history = _load_watch_history()
    watch_history_text = _watch_history_text(watch_history)

    async with httpx.AsyncClient() as client:
        await _honcho_ensure(client)
        recalled = await _honcho_recall(client, question)
        result = await _ask_llm(client, question, watch_history_text, recalled)

        answer = result["answer"]
        if result["update"] and result["title"] and result["status"]:
            title_norm = _normalize_title(result["title"])
            _apply_watch_history_update(
                watch_history, result["title"], result["status"],
                result["liked"], result["stopped_at"], result["reason"], result["note"],
            )
            logger.info("watch history updated: %s -> %s", title_norm, result["status"])
            answer += f"\n\n(saved: {title_norm} -> {result['status']})"
        elif result["update"]:
            # model said this was a change but didn't give a usable title/status --
            # log it instead of silently dropping what might be a real change
            logger.warning("update signaled but missing title/status, question=%r, raw fields=%r", question, result)

        await _honcho_save_turn(client, question, answer)
        return answer


@mcp.tool()
async def check_anticipated(title: str = None, ctx: Context = None) -> str:
    """Check the real current release/season status on TMDB for something you're anticipating.

    Pass a title to check just that one (it doesn't need to already be on
    your anticipated list -- it'll still look it up). Leave title empty to
    check everything currently on your anticipated list.

    Args:
        title: A specific movie/show title to check, or leave empty for the whole anticipated list.

    Returns:
        The current TMDB status for the title(s) checked -- e.g. "Returning
        Series -- next episode S3E1 on 2027-01-15", or "not found on TMDB".
    """
    if not TMDB_API_KEY:
        return "TMDB isn't configured (no API key) -- can't check live release status."

    watch_history = _load_watch_history()
    anticipated = watch_history["anticipated"]
    mutated = False

    async with httpx.AsyncClient() as client:
        if title:
            title_norm = _normalize_title(title)
            match = next((e for e in anticipated if _normalize_title(e.get("title", "")).lower() == title_norm.lower()), None)
            if match:
                before = dict(match)
                line = await _check_one_anticipated(client, match)
                mutated = match != before
            else:
                found = await _tmdb_search(client, title)
                if not found:
                    return f"{title}: not found on TMDB"
                status = await _tmdb_status_for(client, found["id"], found["media_type"])
                line = f"{title}: {status}"
        else:
            if not anticipated:
                return "Nothing on your anticipated list."
            lines = []
            for entry in anticipated:
                before = dict(entry)
                lines.append(await _check_one_anticipated(client, entry))
                mutated = mutated or entry != before
            line = "\n".join(lines)

    if mutated:
        _save_watch_history(watch_history)
    return line


if __name__ == "__main__":
    # streamable_http_path defaults to "/mcp" -- json_response deliberately
    # NOT set, same reasoning as every other server.py in this repo (see
    # media-gen-mcp's own comment on this call).
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
