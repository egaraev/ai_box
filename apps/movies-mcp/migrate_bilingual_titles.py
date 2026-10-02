# movies-mcp: ONE-TIME migration adding title_en/title_ru to every entry in
# watch-history.yaml (2026-10-02, Eldar's request -- store both languages up
# front so hermes_voice.py/movies-mcp never need to translate a title at
# reply time again).
#
# Deliberately does NOT ask any LLM to translate -- that's the same mistake
# that caused the Man of Steel -> Iron Man / Terminator Salvation bugs
# (a model "translating" a title it doesn't actually know, producing a
# confident wrong answer). Most of the Russian titles here are distributor
# localizations, not literal translations ("1+1" is the Russian release
# title for "Intouchables" -- totally unrelated words), so this looks the
# real answer up on TMDB (same API check_anticipated already uses) by ID,
# instead of guessing from the words.
#
# RUN ONCE, INSIDE THE POD, so it reuses the already-configured
# TMDB_API_KEY/WATCH_HISTORY_PATH env vars and the already-mounted file --
# no new secrets, no new mounts. From your machine:
#   POD=$(kubectl get pod -n ai-mcp -l app=movies-mcp -o jsonpath='{.items[0].metadata.name}')
#   kubectl cp apps/movies-mcp/migrate_bilingual_titles.py ai-mcp/$POD:/app/migrate_bilingual_titles.py
#   kubectl exec -n ai-mcp $POD -- python3 /app/migrate_bilingual_titles.py
#
# Safe to re-run: skips any entry that already has both title_en and
# title_ru set, so Ctrl+C and re-running later just picks up where it left
# off. Saves the file after every batch (not just at the end), so an
# interruption loses at most one batch's worth of lookups, never more.
import asyncio
import os
import shutil
import re

import httpx
import yaml

WATCH_HISTORY_PATH = os.environ.get("WATCH_HISTORY_PATH", "/app/data/watch-history.yaml")
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
TMDB_URL = "https://api.themoviedb.org/3"

TITLED_LISTS = ("completed", "on_hold", "suggested_not_yet_watched", "wishlist", "anticipated")
# how many entries to resolve before saving progress to disk -- small enough
# that Ctrl+C never loses much, big enough that saving isn't the bottleneck
BATCH_SIZE = 50
# how many TMDB lookups to run at once -- TMDB has no hard documented limit
# today but there's no reason to hammer it for a one-time job
CONCURRENT_LOOKUPS = 5

# a bunch of entries already got hand-written as "Russian Title (English
# Title)" (mostly the newer TV-show adds) -- split those for free, no TMDB
# call needed, instead of searching TMDB for the combined string (which
# would just confuse the search)
BILINGUAL_PAREN_RE = re.compile(r"^(.+?)\s*\((.+)\)$")


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _save(data, path):
    # same atomic-write pattern as server.py's _save_watch_history --
    # temp file next to the real one, then os.replace() onto it
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    os.replace(tmp_path, path)


async def _tmdb_search(client, title):
    # GET https://api.themoviedb.org/3/search/multi?api_key=...&query=1%2B1&language=en-US
    # returns e.g. {"results": [{"id": 77, "media_type": "movie",
    #   "title": "The Intouchables", "original_title": "Intouchables"}, ...]}
    resp = await client.get(
        f"{TMDB_URL}/search/multi",
        params={"api_key": TMDB_API_KEY, "query": title, "language": "en-US"},
        timeout=15.0,
    )
    resp.raise_for_status()
    for result in resp.json().get("results") or []:
        if result.get("media_type") in ("tv", "movie"):
            return result
    return None


async def _tmdb_localized_title(client, tmdb_id, media_type, language):
    # GET https://api.themoviedb.org/3/movie/77?api_key=...&language=ru-RU
    # returns e.g. {"title": "1+1", ...} -- TMDB falls back to the original
    # title on its own if no translation is registered for that language,
    # which is an acceptable fallback here (better than nothing)
    path = "tv" if media_type == "tv" else "movie"
    resp = await client.get(f"{TMDB_URL}/{path}/{tmdb_id}", params={"api_key": TMDB_API_KEY, "language": language}, timeout=15.0)
    resp.raise_for_status()
    data = resp.json()
    return data.get("title") or data.get("name")


async def _resolve_bilingual_title(client, sem, title):
    # returns (title_en, title_ru, resolved) -- resolved=False means TMDB had
    # nothing and the caller is about to get the same string back in both
    # fields, not a real translation
    already_split = BILINGUAL_PAREN_RE.match(title)
    if already_split:
        # convention used elsewhere in this file is "Russian (English)" --
        # group 1 is the Russian half, group 2 the English half
        return already_split.group(2).strip(), already_split.group(1).strip(), True

    async with sem:
        try:
            found = await _tmdb_search(client, title)
        except Exception as exc:  # noqa: BLE001 -- one bad lookup shouldn't stop the whole migration
            print(f"  TMDB search failed for {title!r}: {type(exc).__name__}: {exc}")
            found = None
        if not found:
            return title, title, False
        title_en = found.get("title") or found.get("name") or title
        try:
            title_ru = await _tmdb_localized_title(client, found["id"], found.get("media_type", "movie"), "ru-RU")
        except Exception as exc:  # noqa: BLE001 -- same as above, degrade instead of aborting
            print(f"  TMDB ru-RU detail lookup failed for {title!r}: {type(exc).__name__}: {exc}")
            title_ru = title
        return title_en, title_ru or title, True


async def _migrate_list(client, sem, data, list_name):
    entries = data[list_name]
    todo = [e for e in entries if not (e.get("title_en") and e.get("title_ru"))]
    if not todo:
        print(f"{list_name}: already fully migrated ({len(entries)} entries)")
        return 0, 0

    resolved_count = 0
    unresolved = []
    for batch_start in range(0, len(todo), BATCH_SIZE):
        batch = todo[batch_start:batch_start + BATCH_SIZE]
        results = await asyncio.gather(*(_resolve_bilingual_title(client, sem, e.get("title", "?")) for e in batch))
        for entry, (title_en, title_ru, resolved) in zip(batch, results):
            entry["title_en"] = title_en
            entry["title_ru"] = title_ru
            if resolved:
                resolved_count += 1
            else:
                unresolved.append(entry.get("title", "?"))
        _save(data, WATCH_HISTORY_PATH)  # checkpoint after every batch, not just at the end
        done = min(batch_start + BATCH_SIZE, len(todo))
        print(f"{list_name}: {done}/{len(todo)} resolved this run ({len(entries) - len(todo) + done}/{len(entries)} total)")

    if unresolved:
        print(f"{list_name}: {len(unresolved)} entries had no TMDB match, left as single-language (title_en == title_ru):")
        for title in unresolved:
            print(f"  - {title}")
    return resolved_count, len(unresolved)


async def main():
    if not TMDB_API_KEY:
        print("TMDB_API_KEY is not set in this environment -- can't run the migration, exiting.")
        return

    backup_path = f"{WATCH_HISTORY_PATH}.backup-2026-10-02"
    if not os.path.exists(backup_path):
        shutil.copy2(WATCH_HISTORY_PATH, backup_path)
        print(f"backed up original file to {backup_path}")
    else:
        print(f"backup already exists at {backup_path} (not overwriting -- this looks like a resumed run)")

    data = _load(WATCH_HISTORY_PATH)
    sem = asyncio.Semaphore(CONCURRENT_LOOKUPS)

    total_resolved = 0
    total_unresolved = 0
    async with httpx.AsyncClient() as client:
        for list_name in TITLED_LISTS:
            resolved, unresolved = await _migrate_list(client, sem, data, list_name)
            total_resolved += resolved
            total_unresolved += unresolved

    print(f"\ndone -- {total_resolved} entries resolved via TMDB, {total_unresolved} left single-language (no TMDB match found)")


if __name__ == "__main__":
    # one-time pass: adds title_en/title_ru to every entry in all five
    # titled lists (not "interests", which stays freeform text), resolving
    # against TMDB by ID rather than guessing a translation. Safe to re-run.
    asyncio.run(main())
