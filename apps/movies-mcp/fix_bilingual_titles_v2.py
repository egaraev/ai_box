# movies-mcp: corrective pass, 2026-10-02, same day as the first two
# bilingual scripts. Both migrate_bilingual_titles.py and the first
# version of server.py's _resolve_bilingual_title had the same bug: they
# accepted TMDB's "ru-RU detail" call as a real translation whenever it
# returned ANY non-empty string, without checking it was actually
# different from the English title. TMDB doesn't error when a title has
# no registered Russian translation -- it silently echoes the original
# title back, which is indistinguishable from a real translation unless
# you check for it. That silently miscounted 40 entries as "resolved" in
# the first migration's own success count (2012, 65, Адреналин, Изгой,
# Красавица и чудовище, Механик, Никто, and more -- none of them flagged).
#
# This fixes it by re-resolving EVERY entry where title_en == title_ru
# (the real signal, regardless of which script produced it or whether it
# got flagged), using the corrected rule: whichever language the entry
# already has is kept EXACTLY as-is, never overwritten -- TMDB is only
# ever asked to fill in the missing OTHER side, and whatever it returns
# has to be both genuinely different AND in the right script (Cyrillic
# for ru-RU, not Cyrillic for en-US) or it's rejected as an echo, not
# trusted. Tries the exact title first, then (flagged lower-confidence if
# it's what resolves it) a subtitle-stripped retry, then the local model
# as a last resort -- same order server.py's own corrected
# _resolve_bilingual_title uses for any brand-new title from here on.
#
# RUN THE SAME WAY as the other two:
#   POD=$(kubectl get pod -n ai-mcp -l app=movies-mcp -o jsonpath='{.items[0].metadata.name}')
#   kubectl cp apps/movies-mcp/fix_bilingual_titles_v2.py ai-mcp/$POD:/app/fix_bilingual_titles_v2.py
#   kubectl exec -n ai-mcp $POD -- python3 /app/fix_bilingual_titles_v2.py
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
CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")
SUBTITLE_SPLIT_RE = re.compile(r"\s*[:–—-]\s*")
JUNK_SIGNS_RE = re.compile(r"full movie|4k ultra hd|cinematic|bədii filmi", re.IGNORECASE)
UNCONFIRMED_NOTE = "title auto-translated, not confirmed via TMDB -- double check"
NO_TRANSLATION_NOTE = "no distinct Russian/English title found (checked TMDB + local model) -- likely the same in both languages"


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _save(data, path):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    os.replace(tmp_path, path)


def _looks_cyrillic(text):
    return bool(CYRILLIC_RE.search(text))


async def _tmdb_search_one(client, path, query, language):
    resp = await client.get(f"{TMDB_URL}/search/{path}", params={"api_key": TMDB_API_KEY, "query": query, "language": language}, timeout=15.0)
    resp.raise_for_status()
    results = resp.json().get("results") or []
    return results[0] if results else None


async def _tmdb_search_harder(client, title):
    # same as retry_unresolved_bilingual_titles.py's version -- tries movie
    # then tv, en-US then ru-RU, full title then (as a lower-confidence
    # fallback) with any subtitle stripped off. is_exact tells the caller
    # which one actually matched, since a stripped-subtitle match could be
    # the wrong specific entry in a franchise.
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


async def _resolve(client, title):
    # returns (title_en, title_ru, confirmed) -- mirrors server.py's fixed
    # _resolve_bilingual_title exactly, plus the harder multi-endpoint
    # search and subtitle-stripped fallback this one-time pass can afford
    # to spend extra calls on. The known side (title itself) is NEVER
    # overwritten; only the missing side ever gets filled in, and only
    # with something that's both genuinely different and in the right
    # script -- an echo or wrong-script result is rejected, not trusted.
    is_cyrillic = _looks_cyrillic(title)
    wants_cyrillic = not is_cyrillic  # the side we're filling in

    found, media_path, is_exact = await _tmdb_search_harder(client, title)
    other = None
    if found:
        target_language = "ru-RU" if wants_cyrillic else "en-US"
        try:
            candidate = await _tmdb_localized_title(client, found["id"], media_path, target_language)
        except Exception as exc:  # noqa: BLE001 -- fall through to local model below
            print(f"  {target_language} detail lookup failed for {title!r}: {type(exc).__name__}: {exc}")
            candidate = None
        if candidate and candidate != title and _looks_cyrillic(candidate) == wants_cyrillic:
            other = candidate

    if other:
        confirmed = is_exact  # a subtitle-stripped match might be the wrong specific entry in a franchise
        return (other, title, confirmed) if is_cyrillic else (title, other, confirmed)

    guessed = await _local_model_guess(client, title, "Russian" if wants_cyrillic else "English")
    guessed = guessed if guessed and guessed != title and _looks_cyrillic(guessed) == wants_cyrillic else title
    return (guessed, title, False) if is_cyrillic else (title, guessed, False)


async def main():
    if not TMDB_API_KEY:
        print("TMDB_API_KEY is not set -- can't run this, exiting.")
        return

    data = _load(WATCH_HISTORY_PATH)
    fixed, unconfirmed, no_translation, skipped_junk = 0, [], [], []

    async with httpx.AsyncClient() as client:
        for list_name in TITLED_LISTS:
            for entry in data[list_name]:
                title_en, title_ru = entry.get("title_en"), entry.get("title_ru")
                if not title_en or title_en != title_ru:
                    continue  # already genuinely bilingual, or not migrated -- leave alone

                if JUNK_SIGNS_RE.search(title_en):
                    skipped_junk.append(title_en)
                    continue

                new_en, new_ru, confirmed = await _resolve(client, title_en)
                entry["title_en"], entry["title_ru"] = new_en, new_ru

                # clear out any stale unconfirmed-tag text before deciding
                # whether to re-add it, so a re-run never doubles it up
                if entry.get("note") in (UNCONFIRMED_NOTE, NO_TRANSLATION_NOTE):
                    entry.pop("note", None)

                if confirmed:
                    fixed += 1
                    print(f"fixed: {title_en} -> {new_en} / {new_ru}")
                elif new_en != title_en or new_ru != title_ru:
                    entry["note"] = "; ".join([entry["note"], UNCONFIRMED_NOTE]) if entry.get("note") else UNCONFIRMED_NOTE
                    unconfirmed.append(f"{title_en} -> {new_en} / {new_ru}")
                else:
                    entry["note"] = "; ".join([entry["note"], NO_TRANSLATION_NOTE]) if entry.get("note") else NO_TRANSLATION_NOTE
                    no_translation.append(title_en)

    _save(data, WATCH_HISTORY_PATH)
    print(f"\ndone -- {fixed} genuinely fixed via TMDB, {len(unconfirmed)} resolved but flagged unconfirmed, "
          f"{len(no_translation)} confirmed to have no distinct translation, {len(skipped_junk)} skipped (not real indexed titles)")
    if unconfirmed:
        print("resolved but flagged unconfirmed (spot-check these):")
        for line in unconfirmed:
            print(f"  - {line}")
    if no_translation:
        print("no distinct translation found (probably fine as single-language, e.g. bare years/numbers):")
        for title in no_translation:
            print(f"  - {title}")
    if skipped_junk:
        print("skipped (YouTube/non-indexed, left untouched):")
        for title in skipped_junk:
            print(f"  - {title}")


if __name__ == "__main__":
    # one-time corrective pass: re-resolves every entry currently showing
    # title_en == title_ru (the real "not actually bilingual" signal) with
    # the fixed algorithm that rejects TMDB's silent fallback-to-original
    # instead of accepting it as a real translation
    asyncio.run(main())
