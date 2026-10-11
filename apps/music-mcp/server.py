"""music-mcp: an HTTP MCP server that knows Eldar's whole MP3 collection,
so he can ask "do I have this song" (usually just heard via Shazam in the
car) and have it say yes/no/maybe instead of him manually scrolling a
1200+ song Winamp playlist. Third of the "specialist agent" family after
movies-mcp/vault-mcp -- same tiny-pod-per-specialist reasoning (see
movies-mcp/server.py's top-of-file comment for the full writeup), same
Honcho-backed persistent memory, reusing the SAME Honcho deployment/
workspace ("aibox-specialists") movies-mcp already uses, just its own
peers/session so the two specialists' conversations never mix.

DATA SOURCE (Eldar's decision, 2026-10): export the Winamp playlist to
M3U/M3U8 and parse it, rather than scanning file tags or hitting a
streaming-service API -- his whole library already has clean folder-based
genre tags (numbered folders like "3. Rock") and he already knows how to
re-export an updated M3U when he adds songs. music_import.py (a standalone
script, run by hand -- NOT a live MCP tool, same reasoning as
vault_sync.py vs vault-mcp) does the actual parsing and writes
music-library.yaml; this file only ever READS that file and appends single
new entries, it never re-parses a playlist itself.

MATCHING DESIGN -- the core decision this file makes on every call, and
the one Eldar pushed back on getting right (see conversation: "cannot
qwen3.8-27B distinguish between similar names??"). A first prototype tried
pure difflib string-similarity scoring to catch "same song, different
spelling" cases -- it worked for the one real example in Eldar's library
(Bon Jovi's "Living on a Prayer" vs "Livin' On A Prayer", 0.96 ratio) but
threw false positives whenever two DIFFERENT songs shared a long artist
name ("Within Temptation - Frozen" vs "Within Temptation - Iron" scored
0.91 despite being unrelated songs) -- a long shared substring inflates a
character-overlap ratio regardless of whether the actual song differs.
Raw fuzzy-string-ratio is never the real decision-maker here because of
that. The fix, mirroring gmail_sync.py's already-proven decide_movie_status
pattern for movies: a cheap, instant, FREE exact-normalized-string match
first (handles the overwhelming majority of real lookups -- Shazam's
title usually lands close enough to exact once junk/case/accents are
stripped), and only when that comes up empty does a real reasoning model
(qwen3.8-27b-Expert via LiteLLM's "auto" route) read the WHOLE library as
text plus the query and make an actual semantic judgment -- MATCH,
POSSIBLE_MATCH, or NOT_FOUND -- the same three-way split
decide_movie_status uses (DECISION/MATCHED_TITLE/REASON), just renamed for
songs. A 27B model reading "Within Temptation - Frozen" against "Within
Temptation - Iron" immediately knows those are different songs, where the
string-overlap ratio got fooled by the shared artist name.

AMBIGUOUS MATCHES (Eldar's decision): when the fallback comes back
POSSIBLE_MATCH rather than a confident MATCH, this tool tells Eldar it's a
possible match and asks him to confirm, rather than either silently
accepting it (risk: masks a genuinely new song as a duplicate, or worse,
refuses to add something he actually wants added) or silently rejecting it
(risk: tells him "you don't have it" when he actually does, under a
slightly different spelling).

normalize_for_matching BUG (found testing music_import.py against the
real 1257-entry playlist, same function duplicated here on purpose -- see
that file's own comment on why there's no shared import across the
venv-script/container-pod boundary): the original keep-pattern only kept
a-z0-9, which silently drops Cyrillic entirely and collapses every
Cyrillic song to the same empty string, "matching" all of them against
each other. Fixed in both copies by also keeping the Cyrillic range.

ADD flow: "I added X to my playlist, update your info" runs the exact
same fast-match-then-27B-fallback check before writing (to avoid filing a
genuine duplicate under a slightly different spelling), then appends to
music-library.yaml and confirms with a "(added: ...)" sentinel in the
reply -- same convention movies-mcp's ask_movies uses for its "(saved:
...)" confirmation, so Hermes has one reliable substring to check rather
than trusting its own paraphrase of what happened.
"""
import logging
import os
import re
import unicodedata
from datetime import date

import httpx
import yaml
from mcp.server.mcpserver import Context, MCPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s music-mcp %(levelname)s: %(message)s")
logger = logging.getLogger("music-mcp")

# same base_url/auth pattern every other tool in this repo uses to reach
# LiteLLM -- Bearer auth against LITELLM_MASTER_KEY, sourced from the same
# litellm-secret Secret.
LITELLM_URL = os.environ.get("LITELLM_URL", "http://litellm.ai-models.svc.cluster.local:4000/v1")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")
# the small, fast, resident-on-the-3060 model -- same one movies-mcp uses
# for its own structured-extraction step (see that file's own comment on
# why naming it directly, instead of going through "auto", is safe and
# fast for this kind of task). Only the ambiguous-match fallback below
# goes through "auto" (-> qwen3.8-27b-Expert, real reasoning).
EXTRACT_MODEL = "qwen3-1.7b-gpu"

HONCHO_URL = os.environ.get("HONCHO_URL", "http://honcho-api.ai-memory.svc.cluster.local:8000")
HONCHO_API_KEY = os.environ.get("HONCHO_API_KEY", "")
HONCHO_WORKSPACE = "aibox-specialists"  # SAME workspace movies-mcp uses -- just its own peers/session below
HONCHO_SESSION = "music-eldar"
HONCHO_HUMAN_PEER = "eldar-music"
HONCHO_AGENT_PEER = "music-agent"

LIBRARY_PATH = os.environ.get("LIBRARY_PATH", "/app/data/music-library.yaml")

CYRILLIC_RE = re.compile(r"[а-яёіїєґ]", re.IGNORECASE)


def normalize_for_matching(text):
    # lowercase, strip accents, drop everything but letters/digits/spaces --
    # ONLY for the exact-match lookup below, never what gets stored or
    # shown. DELIBERATE DUPLICATE of music_import.py's own copy (see that
    # file's comment) -- fix the same bug in both if found in either.
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9а-яёіїєґ ]", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def match_key(artist, title):
    if not title:
        return None
    key = normalize_for_matching(f"{artist or ''} {title}")
    return key or None  # empty key should never "match" another empty key


def _song_label(song):
    base = f"{song['artist']} - {song['title']}" if song.get("artist") else song["title"]
    return f"{base} [{song['genre']}]" if song.get("genre") else base


def _library_text(songs):
    # plain one-line-per-song list for the fallback LLM prompt -- e.g.
    #   Bon Jovi - Livin' On A Prayer [Rock]
    #   Within Temptation - Frozen [Rock]
    # ~1257 songs lands in roughly the same token range as movies-mcp's
    # own ~1268-entry watch history (~6.4k tokens), already proven workable
    # at this scale by decide_movie_status.
    return "\n".join(_song_label(song) for song in songs) if songs else "(empty -- no songs in the library yet)"


def _load_library():
    try:
        with open(LIBRARY_PATH) as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        data = {}
    data.setdefault("songs", [])
    return data


def _save_library(data):
    # atomic write -- temp file next to the real one, then os.replace(),
    # same reasoning (and same required DIRECTORY mount, not a single-file
    # mount) as movies-mcp's _save_watch_history.
    tmp_path = f"{LIBRARY_PATH}.tmp"
    with open(tmp_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    os.replace(tmp_path, LIBRARY_PATH)


mcp = MCPServer("music")


def _honcho_headers():
    return {"Authorization": f"Bearer {HONCHO_API_KEY}"} if HONCHO_API_KEY else {}


async def _honcho_ensure(client):
    # get-or-create is idempotent -- fine to call every request. Best
    # effort: if Honcho is down, the tool still answers, just without
    # memory for this one call.
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
    # POST /v3/workspaces/aibox-specialists/peers/eldar-music/chat
    #   {"query": "do I have Frozen by Within Temptation", "session_id": "music-eldar"}
    # returns e.g. {"content": "Eldar said he's into 80s/90s rock, mostly."}
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


# small, fast model (EXTRACT_MODEL) just pulls structure out of the
# message -- it never decides whether the song is actually in the
# library, that's the exact-match / fallback logic in ask_music itself.
SYSTEM_PROMPT_EXTRACT = """You help identify a song mentioned in a message about Eldar's personal MP3 collection. He might be asking whether he already has a song (often just heard via Shazam in the car), or telling you he's added a new one and wants the library's records updated to match.

Reply in EXACTLY this format, one field per line, nothing before or after it:
INTENT: check or add -- check when he's asking whether he has a song (or anything ambiguous), add only when he's clearly telling you a song is now in his collection (e.g. "I added X", "update your info to include X", "yes, it's different, add it")
ARTIST: <the artist/band name as given, or none if not stated>
TITLE: <the song title as given, or none if not stated>
GENRE: <genre/style only if he explicitly says one (e.g. "classic rock", "trance") -- otherwise none>"""


def _none_or(value):
    return None if value.strip().lower() == "none" else value.strip()


def _parse_extract_response(raw):
    fields = {"intent": "check", "artist": None, "title": None, "genre": None}
    for line in raw.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().upper()
        if key == "INTENT":
            fields["intent"] = "add" if value.strip().lower().startswith("add") else "check"
        elif key == "ARTIST":
            fields["artist"] = _none_or(value)
        elif key == "TITLE":
            fields["title"] = _none_or(value)
        elif key == "GENRE":
            fields["genre"] = _none_or(value)
    return fields


async def _ask_extract_llm(client, question):
    # sends e.g.:
    #   {"model": "qwen3-1.7b-gpu", "messages": [{"role": "system", ...},
    #    {"role": "user", "content": "do I have Frozen by Within Temptation"}]}
    # returns e.g. {"choices": [{"message": {"content":
    #   "INTENT: check\nARTIST: Within Temptation\nTITLE: Frozen\nGENRE: none"}}]}
    payload = {
        "model": EXTRACT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_EXTRACT},
            {"role": "user", "content": question},
        ],
        "temperature": 0.3,
        "max_tokens": 150,
        # explicit, same reasoning as movies-mcp's own direct-model calls --
        # this is structured extraction, not a task that benefits from a
        # reasoning pass, and the whole point of naming this model is speed
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = await client.post(
        f"{LITELLM_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LITELLM_API_KEY}"} if LITELLM_API_KEY else {},
        json=payload,
        timeout=60.0,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    return _parse_extract_response(raw)


# the reasoning-model fallback, only ever called once the fast exact-match
# lookup in ask_music comes up empty. Mirrors gmail_sync.py's
# decide_movie_status DECISION/MATCHED_TITLE/REASON format -- see this
# file's own top comment for why this exists instead of a string-similarity
# score.
SYSTEM_PROMPT_MATCH = """You are checking whether a specific song is already present in Eldar's music library, given below as a plain list of "Artist - Title [Genre]" lines, one song per line. The song to check might be the exact same recording filed under a slightly different name -- a translated or alternately-spelled title, the artist's name formatted differently, a remix/radio-edit tag added or missing -- or it might genuinely not be in the list at all. A shared artist name between the query and a library entry does NOT by itself mean it's the same song -- judge by the actual song, not just a matching artist.

Reply in EXACTLY this format, one field per line, nothing before or after it:
DECISION: MATCH, POSSIBLE_MATCH, or NOT_FOUND -- MATCH only when you're confident it's the same recording under a different name/spelling; POSSIBLE_MATCH when it could plausibly be the same song but you're not fully sure; NOT_FOUND when nothing in the list is plausibly the same song.
MATCHED_SONG: <the exact line from the list it matches, or none if DECISION is NOT_FOUND>
REASON: <one short sentence explaining the decision>"""


def _parse_match_response(raw):
    fields = {"decision": "NOT_FOUND", "matched_song": None, "reason": None}
    for line in raw.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().upper()
        if key == "DECISION":
            decision = value.strip().upper()
            fields["decision"] = decision if decision in ("MATCH", "POSSIBLE_MATCH", "NOT_FOUND") else "NOT_FOUND"
        elif key == "MATCHED_SONG":
            fields["matched_song"] = _none_or(value)
        elif key == "REASON":
            fields["reason"] = _none_or(value)
    return fields


async def _ask_match_llm(client, library_text, artist, title):
    # sends e.g. (model "auto" -> litellm.yaml's router hook -> real
    # reasoning on qwen3.8-27b-Expert, same route decide_movie_status uses):
    #   {"model": "auto", "messages": [{"role": "system", ...}, {"role": "user",
    #    "content": "MUSIC LIBRARY:\n...\n\nSONG TO CHECK:\nWithin Temptation - Frozen"}]}
    # returns e.g. {"choices": [{"message": {"content":
    #   "DECISION: NOT_FOUND\nMATCHED_SONG: none\nREASON: The library has
    #    Within Temptation - Iron, a different song by the same band."}}]}
    query = f"{artist} - {title}" if artist else title
    user_content = f"MUSIC LIBRARY:\n{library_text}\n\nSONG TO CHECK:\n{query}"
    payload = {
        "model": "auto",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_MATCH},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
        "max_tokens": 400,
    }
    # 120s -- this is a big prompt (the whole library, ~1257 lines) going
    # to the 27B reasoning model, not the quick 3060 lookup above. Keep
    # this comfortably under the 150s Hermes itself allows this tool in
    # hermes-agent.yaml's mcp_servers.music.timeout, so a slow backend
    # fails this one call cleanly instead of Hermes's own timeout cutting
    # the whole request off first.
    resp = await client.post(
        f"{LITELLM_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LITELLM_API_KEY}"} if LITELLM_API_KEY else {},
        json=payload,
        timeout=120.0,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    return _parse_match_response(raw)


def _find_exact(songs, artist, title):
    key = match_key(artist, title)
    if not key:
        return None
    return next((s for s in songs if match_key(s.get("artist"), s.get("title")) == key), None)


@mcp.tool()
async def ask_music(question: str, ctx: Context) -> str:
    """Ask whether a song is already in your MP3 collection, or tell it you added one -- one tool for both.

    Typical use: you heard something nice (often via Shazam) and want to
    know if it's already in your car playlist -- just say the artist and
    title, or paste what Shazam gave you. The tool checks for an exact
    match first (instant); if nothing exact turns up, it reads your whole
    library and makes a real judgment about whether a close match is
    actually the same song under a different spelling, a possible match
    worth confirming, or genuinely new. For a possible match, it asks you
    to confirm rather than guessing either way.

    To record a new song ("I added X to my playlist, update your info" /
    "yes, that's different, add it"), say so in plain language -- the tool
    decides this is an add, runs the same duplicate check first (so you
    don't end up with the same song filed twice under slightly different
    spellings), then saves it.

    Args:
        question: Your question or statement about a song, in plain language, exactly as asked.

    Returns:
        A direct answer. If a song was actually added, the reply also
        contains "(added: Artist - Title)" -- if that substring isn't
        there, nothing was saved.
    """
    if not question or not question.strip():
        raise ValueError("question must not be empty")

    library = _load_library()

    async with httpx.AsyncClient() as client:
        await _honcho_ensure(client)
        recalled = await _honcho_recall(client, question)
        extract = await _ask_extract_llm(client, question)

        artist, title, genre, intent = extract["artist"], extract["title"], extract["genre"], extract["intent"]
        if not title:
            answer = "I couldn't tell which song you mean -- can you give me the artist and title?"
            await _honcho_save_turn(client, question, answer)
            return answer

        exact = _find_exact(library["songs"], artist, title)

        if intent == "add":
            if exact:
                answer = f"Already in your library as {_song_label(exact)} -- no need to add it again."
            else:
                match = await _ask_match_llm(client, _library_text(library["songs"]), artist, title)
                if match["decision"] == "MATCH":
                    answer = (
                        f"Looks like you already have this under a different name: {match['matched_song']} "
                        f"({match['reason']}) -- skipping the add so you don't end up with a duplicate. "
                        f"Tell me it's actually a different song and I'll add it."
                    )
                elif match["decision"] == "POSSIBLE_MATCH":
                    answer = (
                        f"This might already be close to {match['matched_song']} ({match['reason']}) -- "
                        f"is that the same song, or a different one? Say it's different and I'll add it."
                    )
                else:
                    new_entry = {
                        "artist": artist, "title": title, "genre": genre,
                        "duration_sec": None, "source": "manual", "file_path": None, "added": date.today().isoformat(),
                    }
                    library["songs"].append(new_entry)
                    _save_library(library)
                    logger.info("song added: %s", _song_label(new_entry))
                    answer = f"Added: {_song_label(new_entry)}.\n\n(added: {artist + ' - ' if artist else ''}{title})"
        else:  # check
            if exact:
                answer = f"Yes, you have it: {_song_label(exact)}."
            else:
                match = await _ask_match_llm(client, _library_text(library["songs"]), artist, title)
                if match["decision"] == "MATCH":
                    answer = f"Yes -- looks like you have it under a different name: {match['matched_song']} ({match['reason']})."
                elif match["decision"] == "POSSIBLE_MATCH":
                    answer = f"Possible match: {match['matched_song']} -- {match['reason']}. Is that the one, or a different song?"
                else:
                    answer = "Doesn't look like you have this one yet -- want me to add it?"

        await _honcho_save_turn(client, question, answer)
        return answer


if __name__ == "__main__":
    # streamable_http_path defaults to "/mcp" -- json_response deliberately
    # NOT set, same convention as every other server.py in this repo.
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
