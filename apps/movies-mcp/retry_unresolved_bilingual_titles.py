# movies-mcp: follow-up to migrate_bilingual_titles.py -- retries ONLY the
# entries that script left single-language (title_en == title_ru is how
# those are identified; a real bilingual pair never happens to match by
# coincidence). 2026-10-02, same day, after the first migration run came
# back with 27 unresolved and a few of them (Django Unchained, The Green
# Mile, Star Trek II-IV, Pirates of the Caribbean...) are famous enough
# that TMDB obviously has them -- the plain /search/multi call in the
# first script just didn't surface the right result for some longer/
# punctuated queries. This tries harder for those, and explicitly SKIPS
# anything that isn't a real indexed title to begin with (the raw YouTube
# "Full Movie Cinematic" links, a regional film with no TMDB entry) --
# forcing a translation onto a YouTube video's title wouldn't mean
# anything, so those are left exactly as they are rather than guessed at.
#
# RUN THE SAME WAY as the first script:
#   POD=$(kubectl get pod -n ai-mcp -l app=movies-mcp -o jsonpath='{.items[0].metadata.name}')
#   kubectl cp apps/movies-mcp/retry_unresolved_bilingual_titles.py ai-mcp/$POD:/app/retry_unresolved_bilingual_titles.py
#   kubectl exec -n ai-mcp $POD -- python3 /app/retry_unresolved_bilingual_titles.py
import asyncio
import os
import re

import httpx
import yaml

WATCH_HISTORY_PATH = os.environ.get("WATCH_HISTORY_PATH", "/app/data/watch-history.yaml")
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
LITELLM_URL = os.environ.get("LITELLM_URL", "http://litellm.ai-models.svc.cluster.local:4000/v1")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")
TMDB_URL = "https://api.themoviedb.org/3"

TITLED_LISTS = ("completed", "on_hold", "suggested_not_yet_watched", "wishlist", "anticipated")
# a Russian/English colon- or dash-separated subtitle ("Пираты Карибского
# моря: Проклятие Черной жемчужины") sometimes throws off TMDB's plain
# search -- stripping it and searching just the main title catches these
SUBTITLE_SPLIT_RE = re.compile(r"\s*[:–—-]\s*")
# signs this was never a real indexed title to begin with -- a YouTube
# compilation's own video title, not an official movie/show name. Skipped
# entirely rather than fed to TMDB or the local model, since there's no
# real "official title" to resolve to.
JUNK_SIGNS_RE = re.compile(r"full movie|4k ultra hd|cinematic|bədii filmi", re.IGNORECASE)


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _save(data, path):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    os.replace(tmp_path, path)


async def _tmdb_search_one(client, path, query, language):
    # GET https://api.themoviedb.org/3/search/movie?api_key=...&query=Shaft&language=ru-RU
    # returns e.g. {"results": [{"id": 310, "title": "Шафт", ...}, ...]}
    resp = await client.get(f"{TMDB_URL}/search/{path}", params={"api_key": TMDB_API_KEY, "query": query, "language": language}, timeout=15.0)
    resp.raise_for_status()
    results = resp.json().get("results") or []
    return results[0] if results else None


async def _tmdb_search_harder(client, title):
    # tries, in order: movie search / tv search, each in en-US then ru-RU,
    # each with the full title then with any subtitle stripped off -- more
    # expensive than migrate_bilingual_titles.py's single /search/multi
    # call, but this only ever runs for the handful of entries that already
    # failed the cheap version, so the extra calls are worth it. is_exact
    # tracks whether this candidate is the real full title or a subtitle-
    # stripped shortcut -- "Чужой: Завет -- Пролог: Тайная Вечеря" stripped
    # down to just "Чужой" ("Alien") would most likely match the ORIGINAL
    # Alien (1979), a completely different entry, not this specific
    # promotional short -- so the caller treats a stripped-query match as a
    # franchise-level guess, not a confirmed one, same tagging as a local-
    # model guess gets.
    candidates = [(title, True)]
    short = SUBTITLE_SPLIT_RE.split(title)[0].strip()
    if short and short != title:
        candidates.append((short, False))

    for query, is_exact in candidates:
        for media_path in ("movie", "tv"):
            for language in ("en-US", "ru-RU"):
                try:
                    found = await _tmdb_search_one(client, media_path, query, language)
                except Exception as exc:  # noqa: BLE001 -- try the next combination
                    print(f"  search/{media_path} {language!r} for {query!r} failed: {type(exc).__name__}: {exc}")
                    continue
                if found:
                    return found, media_path, is_exact
    return None, None, False


async def _tmdb_localized_title(client, tmdb_id, media_type, language):
    resp = await client.get(f"{TMDB_URL}/{media_type}/{tmdb_id}", params={"api_key": TMDB_API_KEY, "language": language}, timeout=15.0)
    resp.raise_for_status()
    data = resp.json()
    return data.get("title") or data.get("name")


async def _local_model_guess(client, title, target_language):
    # same last-resort reasoning as server.py's _translate_title_via_local_model
    # -- only reached when TMDB (tried much harder than the first pass) still
    # has nothing, and the result gets tagged unconfirmed either way
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
        resp = await client.post(f"{LITELLM_URL}/chat/completions", headers={"Authorization": f"Bearer {LITELLM_API_KEY}"}, json=payload, timeout=20.0)
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()
        return None if not result or result.upper() == "UNKNOWN" else result
    except Exception as exc:  # noqa: BLE001 -- degrade to "couldn't translate"
        print(f"  local-model guess failed for {title!r} ({target_language}): {type(exc).__name__}: {exc}")
        return None


async def _retry_one(client, title):
    # returns (title_en, title_ru, resolved) -- resolved=False still means
    # "leave it tagged, don't trust it outright", same as the first
    # script's convention
    found, media_path, is_exact = await _tmdb_search_harder(client, title)
    if not found:
        return title, title, False

    title_en = found.get("title") or found.get("name") or title
    try:
        title_ru = await _tmdb_localized_title(client, found["id"], media_path, "ru-RU")
    except Exception as exc:  # noqa: BLE001 -- got the English side at least, still better than nothing
        print(f"  ru-RU detail lookup failed for {title!r}: {type(exc).__name__}: {exc}")
        title_ru = None

    if title_ru and title_ru != title_en and is_exact:
        return title_en, title_ru, True
    if title_ru and title_ru != title_en and not is_exact:
        # only matched after stripping a subtitle -- right franchise,
        # not necessarily the right specific entry (this is exactly the
        # "Чужой" -> original Alien instead of a specific short risk noted
        # above), so flag it for a human look instead of trusting it
        return title_en, title_ru, False

    # found an English title but no distinct Russian one -- last resort,
    # same risk as always, tagged unconfirmed by the caller
    guessed_ru = await _local_model_guess(client, title_en, "Russian")
    if guessed_ru:
        return title_en, guessed_ru, False
    return title_en, title_en, False


async def main():
    if not TMDB_API_KEY:
        print("TMDB_API_KEY is not set -- can't run this, exiting.")
        return

    data = _load(WATCH_HISTORY_PATH)
    retried = 0
    still_unresolved = []
    skipped_junk = []

    async with httpx.AsyncClient() as client:
        for list_name in TITLED_LISTS:
            for entry in data[list_name]:
                title_en, title_ru = entry.get("title_en"), entry.get("title_ru")
                if not title_en or title_en != title_ru:
                    continue  # already properly bilingual, or not migrated at all -- leave alone

                if JUNK_SIGNS_RE.search(title_en):
                    skipped_junk.append(title_en)
                    continue

                new_en, new_ru, resolved = await _retry_one(client, title_en)
                if resolved:
                    entry["title_en"], entry["title_ru"] = new_en, new_ru
                    entry.pop("note", None) if entry.get("note") == "title auto-translated, not confirmed via TMDB -- double check" else None
                    retried += 1
                    print(f"resolved: {title_en} -> {new_en} / {new_ru}")
                elif new_en != title_en or new_ru != title_ru:
                    # local-model fallback produced something -- keep it, but
                    # make sure it's tagged (first script already tagged most
                    # of these; this covers the couple that weren't)
                    entry["title_en"], entry["title_ru"] = new_en, new_ru
                    if entry.get("note") != "title auto-translated, not confirmed via TMDB -- double check":
                        entry["note"] = ("; ".join([entry["note"], "title auto-translated, not confirmed via TMDB -- double check"])
                                         if entry.get("note") else "title auto-translated, not confirmed via TMDB -- double check")
                    still_unresolved.append(title_en)
                else:
                    still_unresolved.append(title_en)

    _save(data, WATCH_HISTORY_PATH)
    print(f"\ndone -- {retried} newly resolved via TMDB, {len(still_unresolved)} still unresolved, {len(skipped_junk)} skipped (not real indexed titles)")
    if still_unresolved:
        print("still unresolved:")
        for title in still_unresolved:
            print(f"  - {title}")
    if skipped_junk:
        print("skipped (YouTube/non-indexed, left untouched):")
        for title in skipped_junk:
            print(f"  - {title}")


if __name__ == "__main__":
    # one-time follow-up: retries just the entries the first migration left
    # single-language, with a harder TMDB search (movie+tv endpoints
    # separately, en-US and ru-RU, subtitle stripped) before falling back
    # to a tagged local-model guess -- never touches anything already
    # bilingual, and skips anything that isn't a real indexed title at all
    asyncio.run(main())
