# music_import.py -- one-time (and re-runnable) import of a Winamp M3U/M3U8
# playlist export into music-library.yaml, the structured store music-mcp's
# ask_music tool reads from. Mirrors vault_sync.py's standalone-script
# pattern rather than a live MCP tool, since this is a bulk job run by hand
# after exporting a playlist, not something Hermes calls conversationally.
#
# Fixes the two real bugs found in the throwaway prototype parser
# (parse_playlist.py) that was only ever meant to validate the approach:
#   - junk-stripping left dangling empty "()" behind when the junk pattern
#     (e.g. "official video") matched text INSIDE real parentheses that the
#     pattern's own optional "\[?...\]?" only covers for square brackets
#   - a junk phrase sitting exactly where " - " would be between artist and
#     title (e.g. "Band - [official video] - Song") left a stray doubled
#     dash once the junk was removed, which broke the plain " - " split
#
# Usage: python3 music_import.py <path-to-playlist.m3u> [--resync|--dry-run]
#   (no flags)  -- only adds songs whose file_path isn't already in the
#                  library yet (safe to re-run after exporting an updated
#                  playlist from Winamp -- never deletes anything)
#   --resync    -- wipes music-library.yaml and rebuilds it from this
#                  playlist alone
#   --dry-run   -- parses and reports what would happen, writes nothing
import os
import re
import sys
import unicodedata

import yaml

LIBRARY_PATH = "/data/obsidian_vault/_system/specialist-agents/music/music-library.yaml"

# junk tokens that are rip-tool/site artifacts, not real song info -- found
# by actually reading through Eldar's real playlist, not guessed in the
# abstract. Deliberately does NOT include things like "(Live)", "(Radio
# Edit)", "(Remastered)", "feat. X" -- those are real information about the
# recording and should stay.
JUNK_PATTERNS = [
    r"\[\s*ezmp3\.cc\s*\]",
    r"\(\s*ezmp3\.cc\s*\)",
    r"\(muzmino\.org\)",
    r"\(zaycev\.net\)",
    r"-\s*easymp3s",
    r"\[?\s*official\s+(music\s+)?video\s*\]?",
    r"\[?\s*official\s+lyric\s+video\s*\]?",
    r"\[?\s*official\s+performance\s+video\s*\]?",
    r"\[?\s*official\s+visualizer\s*\]?",
    r"\(\s*music\s+video\s*\)",
    r"\(\s*lyric\s+video\s*\)",
    r"\(\s*lyric\s+visualizer\s*\)",
    r"\[?\s*lyrics\s*\]?",
    r"\bmv\b",
]
JUNK_RE = re.compile("|".join(JUNK_PATTERNS), re.IGNORECASE)

GENRE_FOLDER_RE = re.compile(r"^\d+\.\s*")  # strips the "3. " off "3. Rock"


def strip_junk(text):
    cleaned = JUNK_RE.sub("", text)
    cleaned = re.sub(r"\(\s*\)|\[\s*\]", "", cleaned)  # junk removal leaving an empty pair behind, e.g. "(Official Video)" -> "()"
    cleaned = re.sub(r"[\[\]]", "", cleaned)  # any leftover single bracket from a pattern that only matched on one side
    cleaned = re.sub(r"\s*-\s*-\s*", " - ", cleaned)  # junk that sat between artist and title collapses to a doubled dash -- fold back to one
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned.strip(" -_")


def genre_from_path(windows_path):
    # "D:\Music\3. Rock\Bon Jovi - ...\.mp3" -> "Rock"
    folder = windows_path.split("\\")[-2]
    return GENRE_FOLDER_RE.sub("", folder).strip()


def split_artist_title(name):
    # "Artist - Title" -> ("Artist", "Title"). No " - " at all -> artist
    # unknown, whole thing is the title (rare, usually a mislabeled rip).
    if " - " not in name:
        return None, name.strip()
    artist, title = name.split(" - ", 1)
    artist, title = artist.strip(), title.strip()
    # some rips repeat the artist name on both sides of " - "
    if title.lower().startswith(artist.lower() + " - "):
        title = title[len(artist) + 3:].strip()
    return artist, title


def normalize_for_matching(text):
    # lowercase, strip accents, drop everything but letters/digits/spaces --
    # this is ONLY for duplicate-detection here (and the same-shaped
    # exact-match check ask_music does at query time), never what gets
    # stored or shown. Cyrillic doesn't decompose under NFKD, so a Cyrillic
    # title just won't match a LATIN one here -- correct, those are
    # different scripts, not different spellings of the same thing.
    #
    # BUG FOUND testing this against the real playlist: the keep-pattern
    # below originally only kept a-z0-9, which silently strips Cyrillic
    # letters entirely -- every Cyrillic song collapsed to the same empty
    # string and got reported as a "duplicate" of every other Cyrillic
    # song (61 bogus pairs on the real library). Fixed by also keeping the
    # Cyrillic range plus the extra Ukrainian letters (i/i-with-dots/ie/g)
    # that aren't in the plain Russian alphabet.
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9а-яёіїєґ ]", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def match_key(artist, title):
    key = normalize_for_matching(f"{artist or ''} {title}")
    return key or None  # empty key (e.g. no title at all) should never "match" another empty key


def parse_playlist(path):
    songs = []
    pending_extinf = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line == "#EXTM3U":
                continue
            if line.startswith("#EXTINF:"):
                # "#EXTINF:381,Armin van Buuren - Burned With Desire"
                duration, _, name = line[len("#EXTINF:"):].partition(",")
                pending_extinf = (name.strip(), int(duration) if duration.isdigit() else None)
                continue
            # a real file path line
            filename = line.split("\\")[-1]
            stem = re.sub(r"\.mp3$", "", filename, flags=re.IGNORECASE)
            if pending_extinf:
                name_source, duration = pending_extinf
                source = "extinf"
            else:
                name_source, duration = stem, None
                source = "filename"
            pending_extinf = None

            cleaned = strip_junk(name_source)
            artist, title = split_artist_title(cleaned)
            songs.append({
                "artist": artist,
                "title": title,
                "genre": genre_from_path(line),
                "duration_sec": duration,
                "source": source,
                "file_path": line,
            })
    return songs


def load_library():
    try:
        with open(LIBRARY_PATH) as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        data = {}
    data.setdefault("songs", [])
    return data


def save_library(data):
    # atomic write -- same temp-file-then-os.replace() pattern movies-mcp's
    # _save_watch_history uses, for the same reason (never leaves a
    # half-written file if something crashes mid-write)
    tmp_path = f"{LIBRARY_PATH}.tmp"
    with open(tmp_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    os.replace(tmp_path, LIBRARY_PATH)


def main():
    if len(sys.argv) < 2:
        print("usage: python3 music_import.py <playlist.m3u> [--resync|--dry-run]")
        sys.exit(1)
    playlist_path = sys.argv[1]
    resync = "--resync" in sys.argv[2:]
    dry_run = "--dry-run" in sys.argv[2:]

    parsed = parse_playlist(playlist_path)
    library = {"songs": []} if resync else load_library()
    existing_paths = {s["file_path"] for s in library["songs"]}

    added = 0
    for song in parsed:
        if song["file_path"] in existing_paths:
            continue
        library["songs"].append(song)
        existing_paths.add(song["file_path"])
        added += 1

    print(f"parsed {len(parsed)} entries from {playlist_path}, {added} new (library now has {len(library['songs'])})")

    # flag exact-normalized duplicates -- two songs under the SAME
    # normalized artist+title, usually a rip sitting in two genre folders
    # at once. Not auto-merged (too risky to delete something without a
    # person deciding), just surfaced so Eldar can clean them up by hand.
    # Real example found in the prototype run: Bon Jovi's "Livin on a
    # Prayer" filed under two different folders/spellings.
    seen = {}
    dupes = []
    for song in library["songs"]:
        key = match_key(song["artist"], song["title"])
        if key is None:  # nothing usable to compare on -- never counts as a match
            continue
        if key in seen:
            dupes.append((seen[key], song))
        else:
            seen[key] = song
    if dupes:
        print(f"\n{len(dupes)} exact-duplicate pair(s) found (same normalized artist+title):")
        for a, b in dupes:
            print(f"  [{a['genre']}] {a['file_path'].split(chr(92))[-1]}  <->  [{b['genre']}] {b['file_path'].split(chr(92))[-1]}")

    no_artist = [s for s in library["songs"] if not s["artist"]]
    if no_artist:
        print(f"\n{len(no_artist)} song(s) with no artist detected (no ' - ' separator) -- these only match on title")

    if dry_run:
        print("\n--dry-run: nothing written")
        return

    save_library(library)
    print(f"\nwrote {LIBRARY_PATH}")


if __name__ == "__main__":
    # parses the given M3U/M3U8 playlist and merges any new songs (by file
    # path) into music-library.yaml -- safe to re-run after exporting an
    # updated playlist from Winamp, only ever adds, never deletes
    main()
