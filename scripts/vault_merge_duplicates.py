# vault_merge_duplicates.py -- finds and merges duplicate Obsidian notes
# that came from the same source item synced more than once under a
# different filename, then archives the extras and rewrites [[links]].
#
# Ported from the old Windows-lab pipeline (ai-lab/step10_n8n/
# vault_merge_duplicates.py), with one real change: that version merged a
# hand-curated list of ~25 stem pairs (MERGE_GROUPS), written for a vault
# that no longer exists in this form -- none of those stems match anything
# here. This version detects duplicates automatically instead, using a
# pattern specific to this pipeline: gmail_sync.py/telegram_sync.py name
# every note "<slugified-title>-YYYY-MM-DD-HHMM.md", where the date suffix
# comes from the SOURCE MESSAGE's own date, not from when the file was
# written. So if the same email/message gets synced twice -- e.g. because
# clean_title()/slugify() changed and produced a different truncated title
# the second time, which is exactly what happened here on 2026-09-14 --
# both copies carry the identical date suffix. Two notes in the same
# folder sharing that exact suffix, with titles that clearly overlap, are
# almost certainly the same source item. The merge logic itself (richest
# note wins as canonical, unique content from the rest gets appended,
# duplicates get archived not deleted, [[references]] get rewritten) is
# unchanged from the original.
#
# Run:
#   python3 scripts/vault_merge_duplicates.py                 # dry run (default) -- AI/ folder
#   python3 scripts/vault_merge_duplicates.py --folder Health # a different single folder
#   python3 scripts/vault_merge_duplicates.py --all           # entire vault
#   python3 scripts/vault_merge_duplicates.py --apply         # execute merges (modifies/moves files)
#
# NOTE: a note with no date suffix at all (a hand-created note, or one
# whose source date failed to parse) can't be matched this way -- e.g.
# "Grill-Me-Skill-Viral.md" has an obvious duplicate
# ("Grill-Me-Skill-Went-Viral-2026-06-26-1512.md") that this script won't
# find, since only one of the two carries a suffix to match on. Worth a
# manual look at the vault after running this, not just relying on it.

import argparse
import os
import re
import shutil
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

VAULT_PATH = Path("/data/obsidian_vault")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:30633")
COLLECTION_NAME = "obsidian_vault"

# Same gid/convention as gmail_sync.py's chgrp_writable -- kept as its own
# small copy here rather than a shared import, matching every other script
# in this repo (see gmail_sync.py's own comment on why).
HERMES_GID = 10000

def chgrp_writable(path, dir_mode=0o775, file_mode=0o664):
    mode = dir_mode if path.is_dir() else file_mode
    os.chown(path, -1, HERMES_GID)
    path.chmod(mode)

# Matches gmail_sync.py's title_to_filename() suffix exactly: "-YYYY-MM-DD-HHMM"
# right before ".md". Captures just the date part.
DATE_SUFFIX_RE = re.compile(r"-(\d{4}-\d{2}-\d{2}-\d{4})$")

# Two titles in the same suffix-group only get merged if they share at
# least this many leading characters -- guards against the (very unlikely
# for a single personal inbox, but not impossible) case of two unrelated
# messages landing in the same source minute and getting matched purely by
# coincidence.
MIN_COMMON_PREFIX = 10


def read_note(path):
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_body(content):
    return re.sub(r"^---.*?---\s*", "", content, flags=re.DOTALL).strip()


# How many leading characters two strings share, case-insensitive
def common_prefix_len(a, b):
    a, b = a.lower(), b.lower()
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


# Returns duplicate-note content that isn't already in the canonical note
# (rough dedup -- line-by-line, not a real diff)
def unique_sections(canonical_body, duplicate_body):
    canonical_lower = canonical_body.lower()
    lines = []
    for line in duplicate_body.splitlines():
        stripped = line.strip()
        if len(stripped) < 20:
            continue
        if stripped.lower() in canonical_lower:
            continue
        if stripped.startswith("**Source:**") or stripped.startswith("date:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


# Groups notes by (matching date suffix + overlapping title) -- see the
# file header for why this is a reliable signal for THIS pipeline's naming
# convention. Returns a list of lists of Path, each inner list one
# suspected-duplicate group (2+ notes).
def find_duplicate_groups(notes):
    by_suffix = {}
    for note in notes:
        match = DATE_SUFFIX_RE.search(note.stem)
        if not match:
            continue   # no suffix to match on -- see the NOTE in the file header
        by_suffix.setdefault(match.group(1), []).append(note)

    groups = []
    for suffix, candidates in by_suffix.items():
        if len(candidates) < 2:
            continue
        # within one suffix, only cluster notes that also share a real
        # title prefix -- two unrelated notes landing on the same minute
        # shouldn't get merged just because the timestamp matches
        remaining = list(candidates)
        while remaining:
            seed = remaining.pop(0)
            cluster = [seed]
            still_remaining = []
            for other in remaining:
                if common_prefix_len(seed.stem, other.stem) >= MIN_COMMON_PREFIX:
                    cluster.append(other)
                else:
                    still_remaining.append(other)
            remaining = still_remaining
            if len(cluster) >= 2:
                groups.append(cluster)

    return groups


# Deletes this note's existing Qdrant points (keyed by its vault-relative
# path, same as vault_sync.py's ingest_note) -- called after a duplicate is
# archived, so its vectors don't linger under a source path that no note
# lives at anymore.
def delete_qdrant_points(client, source):
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]),
    )


# Replaces every [[old_stem]] with [[new_stem]] across the WHOLE vault
# (regardless of merge scope -- a duplicate could be linked-to from any
# folder). Returns how many files were changed.
def update_references(old_stem, new_stem, dry_run):
    pattern = re.compile(r"\[\[" + re.escape(old_stem) + r"(\|[^\]]+)?\]\]", re.IGNORECASE)
    count = 0
    for md_file in VAULT_PATH.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8", errors="ignore")
        if pattern.search(content):
            count += 1
            if not dry_run:
                md_file.write_text(pattern.sub(f"[[{new_stem}]]", content), encoding="utf-8")
                chgrp_writable(md_file)
    return count


# Processes every detected group: picks the richest file as canonical,
# appends unique content from the rest, archives them, deletes their stale
# Qdrant points, and rewrites references.
def run(notes, archive_dir, apply):
    groups = find_duplicate_groups(notes)

    print(f"Scope  : {len(notes)} notes scanned")
    print(f"Groups : {len(groups)} suspected duplicate group(s) found")
    print(f"Mode   : {'APPLY -- files will be modified/archived!' if apply else 'dry run  (add --apply to execute)'}")
    print("=" * 65)

    if apply:
        archive_dir.mkdir(parents=True, exist_ok=True)
        chgrp_writable(archive_dir)
        client = QdrantClient(url=QDRANT_URL)

    total_archived = 0

    for group in groups:
        group_sorted = sorted(group, key=lambda p: len(read_note(p).splitlines()), reverse=True)
        canonical_path = group_sorted[0]
        canonical_stem = canonical_path.stem
        dupes = group_sorted[1:]

        canonical_content = read_note(canonical_path)
        canonical_body = extract_body(canonical_content)

        print(f"\n📄 {canonical_path.relative_to(VAULT_PATH)}  ({len(canonical_content.splitlines())} lines)  ← canonical")

        additions = []
        for dupe_path in dupes:
            dupe_content = read_note(dupe_path)
            dupe_body = extract_body(dupe_content)
            unique = unique_sections(canonical_body, dupe_body)
            has_unique = len(unique) > 50

            print(f"   ↳ {dupe_path.relative_to(VAULT_PATH)}  ({len(dupe_content.splitlines())} lines)"
                  + (f"  +{len(unique)} chars unique content" if has_unique else "  (no unique content)"))

            if has_unique:
                additions.append((dupe_path.stem, unique))
            total_archived += 1

        if not apply:
            continue

        new_content = canonical_content.rstrip()
        for dupe_stem, unique in additions:
            new_content += f"\n\n---\n*Merged from [[{dupe_stem}]]:*\n\n{unique}"
        canonical_path.write_text(new_content + "\n", encoding="utf-8")
        chgrp_writable(canonical_path)

        for dupe_path in dupes:
            dupe_source = str(dupe_path.relative_to(VAULT_PATH))
            dest = archive_dir / dupe_path.name
            shutil.move(str(dupe_path), str(dest))
            chgrp_writable(dest)
            print(f"   ✓ archived {dupe_path.name} → {archive_dir.relative_to(VAULT_PATH)}/")

            delete_qdrant_points(client, dupe_source)

            refs = update_references(dupe_path.stem, canonical_stem, dry_run=False)
            if refs:
                print(f"   ✓ updated {refs} [[{dupe_path.stem}]] → [[{canonical_stem}]] reference(s)")

    print("\n" + "=" * 65)
    print(f"Total: {total_archived} note(s) to archive across {len(groups)} group(s)")
    if not apply:
        print("Run with --apply to execute. Re-run vault_sync.py afterward to re-index the vault.")
    else:
        print(f"Done. Archived notes are in: {archive_dir.relative_to(VAULT_PATH)}/")
        print("Run vault_sync.py to re-index (the canonical notes changed, and stale duplicate vectors were already cleared above).")


# parses CLI flags, finds duplicate groups in the scanned scope (AI/ by
# default, --folder/--all to change it), and merges them (dry-run unless
# --apply -- this is the one script in the repo where dry-run is the
# default rather than opt-in, since it moves/rewrites files across the
# whole vault, not just the scanned folder)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and merge duplicate Obsidian notes from repeat syncs of the same source item.")
    parser.add_argument("--folder", default="AI", help="vault subfolder to scan (default: AI)")
    parser.add_argument("--all", action="store_true", help="scan the entire vault instead of one folder")
    parser.add_argument("--apply", action="store_true", help="execute merges (modifies and moves files)")
    args = parser.parse_args()

    if args.all:
        scan_notes = [n for n in sorted(VAULT_PATH.rglob("*.md")) if "_archive" not in n.parts]
        archive_dir = VAULT_PATH / "_archive"
    else:
        scan_dir = VAULT_PATH / args.folder
        if not scan_dir.is_dir():
            raise SystemExit(f"Folder not found: {scan_dir}")
        scan_notes = [n for n in sorted(scan_dir.glob("*.md")) if "_archive" not in n.parts]
        archive_dir = scan_dir / "_archive"

    run(scan_notes, archive_dir, apply=args.apply)
