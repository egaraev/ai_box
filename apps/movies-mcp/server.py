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
(six lists: interests, completed, on_hold, suggested_not_yet_watched,
wishlist, anticipated), mounted from the Obsidian vault -- see
movies-mcp-deployment.yaml's own top comment for the hostPath history
and the directory-vs-file mount bug that was fixed there.

wishlist (added 2026-09-30, alongside the Gmail movie-sync pipeline --
see scripts/gmail_sync.py's own top comment): the split from
suggested_not_yet_watched is deliberate and Eldar's own -- wishlist is
for something HE decided he wants to see (told the agent directly, or
planned himself), suggested_not_yet_watched is for candidates that
came from somewhere else (the agent's own recommendation, or now, a
"Movie: ..." email he sent himself). Both are in PROGRESS_LISTS (see
below), so a title can be in at most one of the two, never both.
gmail_sync.py's own dedup-reasoning step always targets
suggested_not_yet_watched for genuinely new titles -- it never writes
wishlist directly; that stays reserved for what Eldar tells the agent
in conversation. IMPORTANT
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

remove / update-in-place (added 2026-10-01): two real bugs found live the
same day. First, there was no way to take a title off the list(s)
entirely -- only to move it to a different status -- so asking Hermes to
"remove" a title sent it in circles (see git log for that incident).
STATUS now also accepts "remove" (see STATUS_SYNONYMS), which deletes the
title from every list instead of filing it anywhere.
Second, _apply_watch_history_update used to build a brand-new entry from
scratch every time, so a later message that only adds one new detail
("I watched season 2") and doesn't repeat everything else would silently
wipe fields an earlier message already saved (LIKED comes back "unknown"
when it isn't mentioned again, so "liked: true" from season 1 was lost).
Fixed by looking up the existing entry first and carrying its fields
forward when the new message doesn't restate them -- see
_apply_watch_history_update's own comment.

LIKED, take two (also 2026-10-01, same day): removed earlier in this
same session because the original field was ambiguous -- it looked like
a sentiment rating ("loved it") but Eldar had actually been using it to
mean something closer to "want to watch/rewatch", a leftover from the
old source-site import method, with no real use in this tool. Eldar then
gave it an actual, deliberate meaning: a real favorites marker, used so
the agent can lean on it as a taste signal for recommendations, and
handed over a list of ~490 titles to mark that way (see the "Marvel
universe", "Форсаж all movies" etc. franchise sweeps in the commit that
did this -- those are substring matches against the completed list, not
hand-picked one by one). So LIKED is back, but it means this now: a
title the model should treat as evidence of what Eldar actually enjoys,
not just a label to store. See SYSTEM_PROMPT's own LIKED line and the
RECALLED CONTEXT paragraph above it.

check_anticipated (also added 2026-09-27, also never existed before
despite TMDB_API_KEY already being wired into the Deployment): live
TMDB lookups for "is X back yet" style questions about anything on the
anticipated list.

BILINGUAL TITLES (added 2026-10-02, Eldar's request): every entry gets
title_en + title_ru instead of a single "title" -- so neither this file
nor hermes_voice.py's TTS pipeline ever has to translate a title again at
reply/speech time. Resolved ONCE, the first time a title is ever recorded
(_resolve_bilingual_title), via a real TMDB lookup by ID, not an LLM
guessing a translation -- deliberately, because most of Eldar's existing
Russian titles are distributor localizations with no literal relationship
to the English title ("1+1" is the Russian release title for
"Intouchables", nothing to do with the words themselves), and guessing
from the words is exactly what produced the Man of Steel -> Iron Man /
Terminator Salvation mistranslation bugs hermes_voice.py hit the same day.
The local qwen3.8-27b-Expert model only gets asked as a last resort, when
TMDB has no match at all, and that guess gets tagged "not confirmed via
TMDB" in the entry's note so it's visible, not silently trusted. The
~1268 pre-existing entries were backfilled once via migrate_bilingual_titles.py
(same repo, run-once, see its own top comment) -- this file's own code
only ever resolves a title the FIRST time it's recorded; a plain status
update ("watched season 2") always reuses what's already stored, never
re-resolves or touches it again. The extraction model's own job (TITLE:
field in SYSTEM_PROMPT) is UNCHANGED by any of this -- still a single
bare title in whichever language Eldar used, still whichever model
LITELLM_MODEL/_wants_recommendation already picks; this only adds a
deterministic Python step after that, same place _apply_watch_history_update
already did its other post-processing.

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
# CHANGED 2026-10-02 (was "auto" for every call): naming qwen3-1.7b-gpu
# directly (same pattern gmail_sync.py/telegram_sync.py already use for
# THEIR internal calls) skips litellm.yaml's "auto" classifier, which was
# landing this call on "reasoning" (thinking ON, qwen3.8-27b-Expert) just
# because a watch-history dump + question reads as substantial -- ~20s of
# needless chain-of-thought for what's usually a simple structured-
# extraction task (temperature 0.3, max_tokens 500 below). Safe to bypass
# "auto" here specifically because qwen3-1.7b-gpu runs pinned resident on
# the 3060 (see llama-swap-configmap.yaml), a card ComfyUI is never even
# given (NVIDIA_VISIBLE_DEVICES=1 only, in its own deployment) -- the
# busy/cloud-fallback protection "auto" provides only matters for the
# 5070 Ti models ComfyUI actually shares with llama-swap.
#
# CONFIRMED LIVE the same day this isn't free, though: asked "what do you
# recommend similar to Spider-Man" and qwen3-1.7b-gpu just pattern-matched
# more Spider-Man entries out of the watch history instead of reasoning
# about genre/vibe -- Hermes caught the bad tool answer and fell back to
# its own general knowledge to actually answer. So recommendation-style
# questions ("what should I watch", "similar to X") still need the bigger
# model's real reasoning -- see _wants_recommendation() below, which
# sends just those through "auto" instead. Lookups/updates ("have I
# watched X", status changes) are plain text-matching against data
# that's already in the prompt -- the small model handles those fine and
# stays on the fast direct path.
LITELLM_MODEL = "qwen3-1.7b-gpu"

# A false positive here just means an ordinary lookup pays for the slower/
# pricier "auto" path instead of the fast one -- same safe-failure-
# direction trade-off litellm.yaml's own router hook accepts for its
# reasoning triggers, so the list below is deliberately a bit generous
# rather than tightly scoped.
RECOMMENDATION_TRIGGERS = (
    "recommend", "suggest", "similar to", "anything good", "what should i watch",
    "what to watch", "something to watch", "посоветуй", "порекомендуй", "похож",
    "что посмотреть",
)


def _wants_recommendation(question):
    normalized = question.lower()
    return any(trigger in normalized for trigger in RECOMMENDATION_TRIGGERS)

HONCHO_URL = os.environ.get("HONCHO_URL", "http://honcho-api.ai-memory.svc.cluster.local:8000")
HONCHO_API_KEY = os.environ.get("HONCHO_API_KEY", "")
HONCHO_WORKSPACE = "aibox-specialists"
HONCHO_SESSION = "movies-eldar"
HONCHO_HUMAN_PEER = "eldar-movies"
HONCHO_AGENT_PEER = "movies-agent"

WATCH_HISTORY_PATH = os.environ.get("WATCH_HISTORY_PATH", "/app/watch-history.yaml")
# all five lists a title can appear in. "interests" is a separate freeform
# list, not managed by the update logic below.
STATUS_LISTS = ("completed", "on_hold", "suggested_not_yet_watched", "wishlist", "anticipated")
# these four ARE mutually exclusive -- a title is finished, paused, queued
# (suggested to Eldar), or wishlisted (Eldar's own pick), never more than
# one of those four at once. "anticipated" is deliberately NOT in this
# group -- see _apply_watch_history_update's comment for why.
PROGRESS_LISTS = ("completed", "on_hold", "suggested_not_yet_watched", "wishlist")

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
TMDB_URL = "https://api.themoviedb.org/3"

# strips a trailing "season 3" / "s3" / "part 2" so the same show doesn't
# end up as two different-looking titles -- confirmed live 2026-09-27:
# "Silo season 3" (completed) and "Silo" (anticipated) were treated as
# unrelated entries because matching was exact-string. Applied once, in
# _apply_watch_history_update, so both comparison and what actually gets
# stored use the same clean title.
SEASON_SUFFIX_RE = re.compile(r"\s+(season\s*\d+|s\d+|part\s*\d+)$", re.IGNORECASE)

# BILINGUAL TITLES (added 2026-10-02, Eldar's request): every entry gets
# title_en + title_ru instead of a single "title", resolved ONCE when a
# title is first recorded and never touched again after that -- see
# _resolve_bilingual_title's own comment for why this is TMDB-first, local-
# model-as-last-resort, not the other way round. The old single "title"
# field is still read as a fallback below (via _title_matches/_entry_label)
# for anything that predates the one-time migrate_bilingual_titles.py run,
# but nothing new is ever written with just "title" again.
CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")
# some entries were already hand-written as "Russian Title (English Title)"
# (mostly the newer TV-show adds) -- split those for free instead of
# spending a TMDB/model call on a title that's already bilingual
BILINGUAL_PAREN_RE = re.compile(r"^(.+?)\s*\((.+)\)$")


def _normalize_title(title):
    return SEASON_SUFFIX_RE.sub("", title.strip()).strip()


def _title_matches(item, title_norm):
    # compares against EVERY language stored on this entry -- title_norm
    # might be in Russian or English depending on which one Eldar just said,
    # and the entry carries both since the bilingual migration (falls back
    # to the legacy single "title" field for anything not yet migrated)
    norm_lower = title_norm.lower()
    for candidate in (item.get("title_en"), item.get("title_ru"), item.get("title")):
        if candidate and _normalize_title(candidate).lower() == norm_lower:
            return True
    return False


def _entry_label(item):
    # "English / Russian" when both are known and differ, so whichever
    # language a reply is in, the right-language title is already sitting
    # in the prompt -- this is the whole point of storing both up front
    # instead of translating at reply time (see hermes_voice.py's
    # localize_for_speech, which this is meant to make mostly unnecessary)
    title_en, title_ru = item.get("title_en"), item.get("title_ru")
    if title_en and title_ru and title_en != title_ru:
        return f"{title_en} / {title_ru}"
    return title_en or title_ru or item.get("title", "?")

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
    # "wishlist" is Eldar's own pick (told the agent directly), as opposed to
    # suggested_not_yet_watched which is for something suggested TO him (by
    # the agent, or by gmail_sync.py's movie-email pipeline) -- see
    # STATUS_LISTS' own comment for the distinction.
    "wishlist": "wishlist", "wish_list": "wishlist", "want_to_watch": "wishlist",
    "wanted": "wishlist", "planned": "wishlist", "plan_to_watch": "wishlist",
    "anticipated": "anticipated", "waiting": "anticipated", "upcoming": "anticipated", "anticipating": "anticipated",
    # "remove" isn't one of the five real lists -- it's a sentinel meaning
    # "take this title off everything, don't file it anywhere else" (added
    # 2026-10-01: Eldar asked to remove a title and there was no way to do
    # that, only to move it to a different status). _apply_watch_history_update
    # special-cases this value instead of appending to data["remove"].
    "remove": "remove", "removed": "remove", "delete": "remove", "deleted": "remove",
    "erase": "remove", "forget": "remove", "forget_it": "remove",
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

CURRENT WATCH HISTORY is included below -- what Eldar has already completed, put on hold, been suggested, wishlisted himself, or is anticipating. Most titles are shown as "English Title / Russian Title" (added 2026-10-02, so neither you nor any other part of this pipeline ever has to translate a title again) -- when you write ANSWER, use whichever side matches the language you're replying in, instead of the other one or a literal translation of your own. Completed titles marked "(liked)" are ones Eldar specifically told the agent he loved -- treat these as real signal about his taste (genres, franchises, tone, directors/actors he gravitates to) and actively use them when he asks for a recommendation, not just as a label sitting in the list. If several liked titles share something (same franchise, same genre, same kind of tone), it's fair to name that pattern and recommend accordingly.

If a block called RECALLED CONTEXT is included, it's what Honcho (this tool's memory layer) remembers from earlier conversations with Eldar about movies -- his stated preferences, what you already recommended, what he said he watched or liked. Use it naturally, instead of asking Eldar to repeat himself.

You do not have live internet access for most questions -- if something depends on very recent release dates or showtimes, say so plainly rather than guessing. (A separate tool handles live TMDB lookups for anticipated titles -- you don't need to guess at those either.)

First, decide: does this message describe a CHANGE to the watch history (something watched, dropped, stopped mid-way, newly wanted, newly anticipated, or to be removed entirely) -- or is it a question / recommendation request? "I finished X, loved it", "mark X as watched", "I stopped watching X at season 2", "I'm waiting for the next season of X", "I want to watch X", "remove X from the list", "delete X, I added it by mistake" are changes. "What should I watch", "what happens in the finale of X", "any good sci-fi lately" are not. A message can describe two changes at once ("I finished season 3, waiting for season 4") -- "anticipated" is independent of completed/on_hold/suggested_not_yet_watched/wishlist (finishing what's out now doesn't cancel anticipating what's next), but you can only record ONE status per reply. If a message clearly describes both, record the progress change (completed/on_hold/suggested_not_yet_watched/wishlist) this time and mention in ANSWER that it'll need to be told separately to also mark it anticipated.

Always reply in EXACTLY this format, one field per line, nothing before or after it:
ANSWER: <your natural, conversational answer, a few sentences unless more is clearly wanted>
UPDATE: yes or no
TITLE: <the bare show/movie title only -- no "season 3" or episode numbers appended, those go in NOTE or STOPPED_AT instead -- or none if UPDATE is no>
STATUS: must be exactly one of these six words -- completed, on_hold, suggested_not_yet_watched, wishlist, anticipated, remove -- or none. Use on_hold for anything dropped, stopped, paused, or quit -- not a different word. Use wishlist when ELDAR HIMSELF says he wants to watch something (his own pick, stated directly) -- use suggested_not_yet_watched only when YOU are the one recommending it to him in this reply, or when he mentions something suggested to him some other way (e.g. an email he sent himself about a movie). Use remove when Eldar wants a title taken off his list(s) ENTIRELY -- added by mistake, a duplicate, or he just doesn't want it tracked anymore -- not for pausing or changing status (that's on_hold/wishlist/etc, the title stays recorded).
LIKED: yes if Eldar said he loved/really liked it ("loved it", "one of my favorites", "really enjoyed it") -- this is a real signal used later for recommendations, not a throwaway label, so only say yes when he's actually expressing that, not just finishing something neutrally. no if he said he disliked it. unknown if he didn't say either way, or if this reply isn't recording completed/anything that had an opinion attached (e.g. a plain status move like "put X on hold" with no liked/disliked said).
STOPPED_AT: <where he stopped, e.g. "season 2 episode 4" -- or none>
REASON: <why he stopped/dropped it, if he said -- or none>
NOTE: <any other detail worth keeping, e.g. a season/episode he just watched ("through season 2") for a title that's already recorded -- or none>"""

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
    #   WISHLIST: Severance
    #   ANTICIPATED: Silo
    lines = []
    if data["completed"]:
        parts = []
        for item in data["completed"]:
            title = _entry_label(item)
            parts.append(f"{title} (liked)" if item.get("liked") is True else title)
        lines.append("COMPLETED: " + "; ".join(parts))
    if data["on_hold"]:
        parts = []
        for item in data["on_hold"]:
            bit = _entry_label(item)
            extra = [item[k] for k in ("stopped_at", "reason") if item.get(k)]
            if extra:
                bit += f" ({', '.join(extra)})"
            parts.append(bit)
        lines.append("ON HOLD: " + "; ".join(parts))
    if data["suggested_not_yet_watched"]:
        lines.append("SUGGESTED: " + "; ".join(_entry_label(item) for item in data["suggested_not_yet_watched"]))
    if data["wishlist"]:
        lines.append("WISHLIST: " + "; ".join(_entry_label(item) for item in data["wishlist"]))
    if data["anticipated"]:
        lines.append("ANTICIPATED: " + "; ".join(_entry_label(item) for item in data["anticipated"]))
    return "\n".join(lines) if lines else "(empty -- nothing recorded yet)"


def _remove_from_progress_lists(data, title_norm):
    # the four PROGRESS_LISTS are mutually exclusive -- moving a title to
    # one drops any stale copy from the other three
    for key in PROGRESS_LISTS:
        data[key] = [item for item in data[key] if not _title_matches(item, title_norm)]


def _remove_from_anticipated(data, title_norm):
    data["anticipated"] = [item for item in data["anticipated"] if not _title_matches(item, title_norm)]


def _remove_title_everywhere(data, title_norm):
    # for an explicit "remove" request -- strips title_norm out of every
    # list it could possibly be in (all five STATUS_LISTS), and reports
    # whether anything was actually found/removed so the caller can tell
    # Eldar when a title wasn't on any list in the first place
    found = False
    for key in STATUS_LISTS:
        before = len(data[key])
        data[key] = [item for item in data[key] if not _title_matches(item, title_norm)]
        if len(data[key]) != before:
            found = True
    return found


def _find_existing_entry(data, title_norm):
    # searches every list a title could be in (added 2026-10-01, alongside
    # the merge-on-update fix below) -- returns the entry dict plus which
    # list it was found in, or (None, None) if it's nowhere yet
    for key in STATUS_LISTS:
        for item in data[key]:
            if _title_matches(item, title_norm):
                return item, key
    return None, None


async def _tmdb_localized_title(client, tmdb_id, media_type, language):
    # GET https://api.themoviedb.org/3/movie/550?api_key=...&language=ru-RU
    # returns e.g. {"title": "Бойцовский клуб", ...} -- TMDB falls back to
    # the original title on its own when no translation is registered for
    # that language, which just means title_ru below could come back equal
    # to title_en -- not wrong, just not actually bilingual for that one.
    path = "tv" if media_type == "tv" else "movie"
    resp = await client.get(f"{TMDB_URL}/{path}/{tmdb_id}", params={"api_key": TMDB_API_KEY, "language": language}, timeout=15.0)
    resp.raise_for_status()
    data = resp.json()
    return data.get("title") or data.get("name")


async def _translate_title_via_local_model(client, title, target_language):
    # LAST RESORT ONLY -- called when TMDB has no match at all for a brand
    # new title (see _resolve_bilingual_title). Direct call to the resident
    # qwen3.8-27b-Expert, not litellm's "auto" classifier -- same reasoning
    # as hermes_voice.py's LOCALIZE_MODEL: this is a one-line lookup, not a
    # question, so skip the classifier round trip and force thinking off.
    # Real risk, same one that caused the Man of Steel -> Iron Man bug: a
    # model "translating" a title it doesn't actually know produces a
    # confident wrong answer, not an error -- that's why the caller always
    # tags whatever comes back here as unconfirmed rather than trusting it.
    if not LITELLM_API_KEY:
        return None
    prompt = (f'Give the official {target_language}-language title for this movie or TV show: "{title}". '
              f"Reply with ONLY the title, nothing else. If you don't actually know the official "
              f"{target_language} title, reply with exactly: UNKNOWN")
    payload = {
        "model": "qwen3.8-27b-Expert",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = await client.post(
            f"{LITELLM_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
            json=payload,
            timeout=20.0,
        )
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()
        return None if not result or result.upper() == "UNKNOWN" else result
    except Exception as exc:  # noqa: BLE001 -- degrade to "couldn't translate", don't fail the save
        logger.warning("local-model title translation failed for %r (%s): %s", title, target_language, exc)
        return None


async def _resolve_bilingual_title(client, title):
    # called ONCE per title, only the first time it's ever recorded (see
    # _apply_watch_history_update) -- returns (title_en, title_ru,
    # confirmed). confirmed=False means at least one side came from the
    # local-model fallback above rather than a real TMDB lookup, so the
    # caller tags the saved entry for Eldar to spot-check later instead of
    # silently trusting a guess.
    already_split = BILINGUAL_PAREN_RE.match(title)
    if already_split:
        # the hand-written convention elsewhere in this file is
        # "Russian (English)" -- group 1 is the Russian half
        return already_split.group(2).strip(), already_split.group(1).strip(), True

    try:
        found = await _tmdb_search(client, title)
    except Exception as exc:  # noqa: BLE001 -- fall through to the local-model path below
        logger.warning("TMDB search failed for %r during bilingual resolve: %s", title, exc)
        found = None

    title_en = title_ru = None
    if found:
        title_en = found.get("title") or found.get("name")
        try:
            title_ru = await _tmdb_localized_title(client, found["id"], found.get("media_type", "movie"), "ru-RU")
        except Exception as exc:  # noqa: BLE001 -- same, fall through
            logger.warning("TMDB ru-RU detail lookup failed for %r: %s", title, exc)
    if title_en and title_ru:
        return title_en, title_ru, True

    # TMDB came up short -- whichever language Eldar actually said is
    # already known for certain; only the OTHER side needs a guess
    is_cyrillic = bool(CYRILLIC_RE.search(title))
    known_en = title_en or (title if not is_cyrillic else None)
    known_ru = title_ru or (title if is_cyrillic else None)
    if not known_en:
        known_en = await _translate_title_via_local_model(client, title, "English") or title
    if not known_ru:
        known_ru = await _translate_title_via_local_model(client, title, "Russian") or title
    return known_en, known_ru, False


async def _apply_watch_history_update(client, data, title, status, liked, stopped_at, reason, note):
    # confirmed live 2026-09-27: completing what's out now and still
    # anticipating the next season are both true at once for the same
    # show -- so a progress update (completed/on_hold/suggested/wishlist) only
    # dedupes against the OTHER progress statuses, never touches
    # anticipated, and vice versa. Cross-clearing them was the original
    # bug (see PROGRESS_LISTS' own comment).
    #
    # returns True if the title ended up recorded/removed as asked, False
    # only for the "remove" case when the title wasn't found on any list
    title_norm = _normalize_title(title)
    if status == "remove":
        found = _remove_title_everywhere(data, title_norm)
        _save_watch_history(data)
        return found

    # look up the existing entry BEFORE wiping it out below -- a later
    # message that only adds one new detail ("watched season 2") and
    # doesn't restate everything (LIKED comes back "unknown" when it isn't
    # repeated) would otherwise silently erase fields the earlier message
    # already recorded (added 2026-10-01, same day LIKED got its real
    # meaning back -- see this file's top comment, "LIKED, take two")
    existing, existing_status = _find_existing_entry(data, title_norm)

    if status == "anticipated":
        _remove_from_anticipated(data, title_norm)
    else:
        _remove_from_progress_lists(data, title_norm)

    # bilingual resolve happens ONCE, only the first time this title is ever
    # recorded -- an existing entry already has both languages settled (by a
    # real TMDB lookup, or a prior guess), so a plain status update ("watched
    # season 2") never re-resolves or overwrites them, it just carries them
    # forward. See _resolve_bilingual_title's own comment for why this is
    # TMDB-first, local-model-as-last-resort.
    unconfirmed_tag = None
    if existing and existing.get("title_en") and existing.get("title_ru"):
        title_en, title_ru = existing["title_en"], existing["title_ru"]
    else:
        title_en, title_ru, confirmed = await _resolve_bilingual_title(client, title_norm)
        if not confirmed:
            # could mean a local-model guess was used, or that TMDB AND the
            # local model both came up empty (network/key down) and the one
            # known-language title just got copied into both fields --
            # either way it's not a real confirmed translation, so flag it
            logger.info("bilingual title for %r not confirmed via TMDB -- flagged for Eldar to double check", title_norm)
            unconfirmed_tag = "title auto-translated, not confirmed via TMDB -- double check"

    entry = {"title_en": title_en, "title_ru": title_ru}
    # liked carries over regardless of status change -- "watched season 2"
    # shouldn't erase "liked: true" recorded when season 1 was marked completed
    if liked is not None:
        entry["liked"] = liked
    elif existing and existing.get("liked") is not None:
        entry["liked"] = existing["liked"]
    # stopped_at/reason are on_hold-specific -- only carry them over on a
    # plain detail update (status unchanged), never across a real
    # transition like on_hold -> completed where they'd just be stale
    if stopped_at:
        entry["stopped_at"] = stopped_at
    elif existing_status == status and existing and existing.get("stopped_at"):
        entry["stopped_at"] = existing["stopped_at"]
    if reason:
        entry["reason"] = reason
    elif existing_status == status and existing and existing.get("reason"):
        entry["reason"] = existing["reason"]
    # note is general-purpose -- append the new detail onto the old one
    # instead of overwriting, so "watched season 2" on top of an earlier
    # "slow start" note keeps both instead of losing the first. The
    # unconfirmed-translation tag (set above, brand-new titles only) is
    # just one more thing to append onto the same note, same pattern.
    note_parts = [part for part in (
        existing.get("note") if existing and existing.get("note") else None,
        note,
        unconfirmed_tag,
    ) if part]
    if note_parts:
        entry["note"] = "; ".join(note_parts)

    data[status].append(entry)
    _save_watch_history(data)
    return True


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
    # sends e.g. (model is "qwen3-1.7b-gpu" for a lookup/update, "auto" for a
    # recommendation-style question -- see RECOMMENDATION_TRIGGERS above):
    #   {"model": "qwen3-1.7b-gpu", "messages": [{"role": "system", ...},
    #    {"role": "user", "content": "CURRENT WATCH HISTORY:\n...\n\nQUESTION:\nI stopped watching Foundation at season 2"}]}
    # returns e.g. {"choices": [{"message": {"content":
    #   "ANSWER: Got it, marked Foundation as on hold.\nUPDATE: yes\nTITLE: Foundation\n
    #    STATUS: on_hold\nLIKED: unknown\nSTOPPED_AT: season 2\nREASON: none\nNOTE: none"}}]}
    user_content = f"CURRENT WATCH HISTORY:\n{watch_history_text}\n\n"
    if recalled:
        user_content += f"RECALLED CONTEXT (from earlier conversations):\n{recalled}\n\n"
    user_content += f"QUESTION:\n{question}"
    # recommendation-style questions need the 27B model's real reasoning (see
    # RECOMMENDATION_TRIGGERS comment above) -- everything else stays on the
    # fast direct-to-3060 path
    model = "auto" if _wants_recommendation(question) else LITELLM_MODEL
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
        "max_tokens": 500,
    }
    if model != "auto":
        # explicit, not just relying on qwen3-1.7b-gpu's own launch-time
        # default -- this is a structured-extraction task, not one that
        # benefits from a reasoning pass, and the whole point of calling
        # this model by name is speed. Left unset for "auto" -- litellm.yaml's
        # own router hook decides thinking on/off for that path, and a
        # recommendation question is exactly the case it should land on
        # "reasoning" (thinking ON).
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    resp = await client.post(
        f"{LITELLM_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LITELLM_API_KEY}"} if LITELLM_API_KEY else {},
        json=payload,
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
    title = _entry_label(entry)
    if not entry.get("tmdb_id"):
        found = await _tmdb_search(client, entry.get("title_en") or entry.get("title", "?"))
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
    dropped, stopped mid-way, are now anticipating, or want removed
    entirely (added by mistake, duplicate, no longer want it tracked).
    Titles you say you loved get marked liked, which this tool then uses
    as a taste signal for future recommendations -- not just stored. The
    tool itself decides which kind of message this is and saves a
    change when one is described; you don't need a separate "update" or
    "delete" call. Remembers past conversations via Honcho, so you don't
    need to repeat context from an earlier chat.

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
            found_or_saved = await _apply_watch_history_update(
                client, watch_history, result["title"], result["status"],
                result["liked"], result["stopped_at"], result["reason"], result["note"],
            )
            if result["status"] == "remove":
                if found_or_saved:
                    logger.info("watch history entry removed: %s", title_norm)
                    answer += f"\n\n(removed: {title_norm})"
                else:
                    logger.info("remove requested but title not found: %s", title_norm)
                    answer += f"\n\n(couldn't find \"{title_norm}\" on any list -- nothing removed)"
            else:
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
            match = next((e for e in anticipated if _title_matches(e, title_norm)), None)
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
