"""
scripts/cleanup_html.py — Remove low-value files from offline HTML mirrors.

Identifies and removes non-content files from dokumentacja/ directories:
  1. href.html duplicates inside html-single guide directories
  2. .html stub/redirect files at the html-single/ level (not inside a guide dir)
  3. html/ multi-page directory (we use html-single/ for full single-page content)
  4. Non-content assets: fonts, styles, _nuxt, robots.txt, SVG, JS, CSS, WOFF, WebP
  5. Landing/index pages outside html-single/

Also normalizes structure:
  - If a guide has index/index.html but no top-level index.html, moves it up

Usage:
    python scripts/cleanup_html.py D:/cloude/dokumentacja/rhel-docs-rhel8
    python scripts/cleanup_html.py D:/cloude/dokumentacja/rhel-docs-rhel10
    python scripts/cleanup_html.py D:/cloude/dokumentacja/rhel-docs-rhel8 --dry-run
    python scripts/cleanup_html.py --all D:/cloude/dokumentacja
"""

import argparse
import shutil
import sys
from pathlib import Path


# ── File categories to remove ────────────────────────────────────────────────

NON_CONTENT_EXTENSIONS = {
    ".woff", ".woff2", ".css", ".js", ".svg", ".webp", ".png", ".jpg",
    ".gif", ".ico", ".ttf", ".eot", ".map",
}

NON_CONTENT_DIRS = {"fonts", "styles", "_nuxt"}

NON_CONTENT_FILES = {"robots.txt"}


def find_html_single_root(base: Path) -> Path | None:
    """
    Find the html-single/ directory inside a docs mirror.

    Handles both direct structure and 'raw/' prefix (RHEL 9 variant).
    """
    # Pattern: base/docs.redhat.com/en/documentation/red_hat_enterprise_linux/{ver}/html-single/
    # Or:      base/raw/docs.redhat.com/en/documentation/red_hat_enterprise_linux/{ver}/html-single/
    for candidate in base.rglob("html-single"):
        if candidate.is_dir() and "red_hat_enterprise_linux" in str(candidate):
            return candidate
    return None


def find_html_multipage_root(base: Path) -> Path | None:
    """Find the html/ (multi-page) directory."""
    for candidate in base.rglob("html"):
        if (
            candidate.is_dir()
            and candidate.name == "html"
            and "red_hat_enterprise_linux" in str(candidate)
            and "html-single" not in str(candidate)
        ):
            return candidate
    return None


def detect_version(base: Path) -> str:
    """Detect RHEL version from directory name."""
    name = base.name.lower()
    for v in ("8", "9", "10"):
        if f"rhel{v}" in name or f"rhel-{v}" in name or f"rhel{v}" in name.replace("-", ""):
            return v
    return "?"


def cleanup_version(base: Path, dry_run: bool = False) -> dict:
    """
    Clean up a single version's dokumentacja directory.

    Returns dict with statistics:
        {
            "version": "8",
            "href_html_removed": N,
            "stub_html_removed": N,
            "multipage_removed": N,
            "assets_removed": N,
            "landing_removed": N,
            "normalized": N,
            "total_removed": N,
            "valid_guides": N,
            "valid_index_files": N,
        }
    """
    version = detect_version(base)
    stats = {
        "version": version,
        "base": str(base),
        "href_html_removed": 0,
        "stub_html_removed": 0,
        "multipage_removed": 0,
        "assets_removed": 0,
        "landing_removed": 0,
        "normalized": 0,
        "total_removed": 0,
        "valid_guides": 0,
        "valid_index_files": 0,
    }

    html_single = find_html_single_root(base)
    if not html_single:
        print(f"  WARNING: No html-single/ directory found in {base}")
        return stats

    print(f"  html-single root: {html_single}")

    # ── 1. Normalize: move index/index.html → index.html if missing ──────
    for guide_dir in sorted(html_single.iterdir()):
        if not guide_dir.is_dir():
            continue
        index_html = guide_dir / "index.html"
        nested_index = guide_dir / "index" / "index.html"
        if not index_html.exists() and nested_index.exists():
            print(f"  NORMALIZE: {guide_dir.name}/index/index.html -> index.html")
            if not dry_run:
                shutil.copy2(nested_index, index_html)
            stats["normalized"] += 1

    # ── 2. Remove href.html duplicates ───────────────────────────────────
    for href in html_single.rglob("href.html"):
        print(f"  REMOVE href.html: {href.relative_to(base)}")
        if not dry_run:
            href.unlink()
        stats["href_html_removed"] += 1

    # ── 3. Remove .html stubs at html-single/ level ─────────────────────
    for stub in sorted(html_single.glob("*.html")):
        if stub.is_file():
            print(f"  REMOVE stub: {stub.relative_to(base)}")
            if not dry_run:
                stub.unlink()
            stats["stub_html_removed"] += 1

    # ── 4. Remove html/ multi-page directory ─────────────────────────────
    html_multi = find_html_multipage_root(base)
    if html_multi and html_multi.exists():
        file_count = sum(1 for _ in html_multi.rglob("*") if _.is_file())
        print(f"  REMOVE html/ multi-page: {html_multi.relative_to(base)} ({file_count} files)")
        if not dry_run:
            # Use \\?\ prefix on Windows for long paths
            long_path = str(html_multi)
            if sys.platform == "win32" and not long_path.startswith("\\\\?\\"):
                long_path = "\\\\?\\" + str(html_multi.resolve())
            shutil.rmtree(long_path, ignore_errors=True)
        stats["multipage_removed"] = file_count

    # ── 5. Remove non-content assets ─────────────────────────────────────
    for f in sorted(base.rglob("*")):
        if not f.is_file():
            continue
        # Non-content by extension
        if f.suffix.lower() in NON_CONTENT_EXTENSIONS:
            if not dry_run:
                f.unlink()
            stats["assets_removed"] += 1
            continue
        # Non-content by filename
        if f.name in NON_CONTENT_FILES:
            if not dry_run:
                f.unlink()
            stats["assets_removed"] += 1
            continue
        # Non-content by parent directory
        if any(part in NON_CONTENT_DIRS for part in f.parts):
            if not dry_run:
                f.unlink()
            stats["assets_removed"] += 1

    # ── 6. Remove landing/index pages outside html-single ────────────────
    # Find all HTML files not under html-single/
    for f in sorted(base.rglob("*.html")):
        if not f.is_file():
            continue
        if "html-single" in str(f):
            continue
        # This is a landing page or html/ leftover
        print(f"  REMOVE landing: {f.relative_to(base)}")
        if not dry_run:
            f.unlink()
        stats["landing_removed"] += 1

    # ── 7. Remove empty directories ──────────────────────────────────────
    if not dry_run:
        _remove_empty_dirs(base)

    # ── 8. Count valid remaining files ───────────────────────────────────
    valid_guides = 0
    valid_index = 0
    for guide_dir in sorted(html_single.iterdir()):
        if not guide_dir.is_dir():
            continue
        index_html = guide_dir / "index.html"
        if index_html.exists():
            valid_guides += 1
            valid_index += 1

    stats["valid_guides"] = valid_guides
    stats["valid_index_files"] = valid_index
    stats["total_removed"] = (
        stats["href_html_removed"]
        + stats["stub_html_removed"]
        + stats["multipage_removed"]
        + stats["assets_removed"]
        + stats["landing_removed"]
    )

    return stats


def _remove_empty_dirs(base: Path) -> int:
    """Remove empty directories bottom-up."""
    removed = 0
    for d in sorted(base.rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
            removed += 1
    return removed


def print_stats(stats: dict) -> None:
    """Print a formatted summary of cleanup statistics."""
    ver = stats["version"]
    print(f"\n  {'='*50}")
    print(f"  RHEL {ver} Cleanup Summary")
    print(f"  {'='*50}")
    print(f"  Base: {stats['base']}")
    print(f"  {'─'*50}")
    print(f"  Removed:")
    print(f"    href.html duplicates:    {stats['href_html_removed']:>5}")
    print(f"    .html stubs at root:     {stats['stub_html_removed']:>5}")
    print(f"    html/ multi-page files:  {stats['multipage_removed']:>5}")
    print(f"    Non-content assets:      {stats['assets_removed']:>5}")
    print(f"    Landing pages:           {stats['landing_removed']:>5}")
    if stats["normalized"]:
        print(f"    Structure normalized:    {stats['normalized']:>5}")
    print(f"  {'─'*50}")
    print(f"  TOTAL REMOVED:             {stats['total_removed']:>5}")
    print(f"  {'─'*50}")
    print(f"  Valid guides remaining:    {stats['valid_guides']:>5}")
    print(f"  Valid index.html files:    {stats['valid_index_files']:>5}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Remove low-value files from RHEL docs mirrors"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Path(s) to dokumentacja version directories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Treat path as parent dir containing rhel-docs-rhel*/ subdirectories",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN — no files will be deleted\n")

    all_stats = []

    for path_str in args.paths:
        base = Path(path_str)
        if not base.exists():
            print(f"ERROR: {base} does not exist")
            sys.exit(1)

        if args.all:
            # Process all rhel-docs-rhel* subdirectories
            version_dirs = sorted(base.glob("rhel-docs-rhel*"))
            if not version_dirs:
                print(f"ERROR: No rhel-docs-rhel* directories found in {base}")
                sys.exit(1)
            for vdir in version_dirs:
                print(f"\nProcessing {vdir.name}...")
                stats = cleanup_version(vdir, dry_run=args.dry_run)
                all_stats.append(stats)
                print_stats(stats)
        else:
            print(f"\nProcessing {base.name}...")
            stats = cleanup_version(base, dry_run=args.dry_run)
            all_stats.append(stats)
            print_stats(stats)

    # Grand total
    if len(all_stats) > 1:
        total_removed = sum(s["total_removed"] for s in all_stats)
        total_valid = sum(s["valid_guides"] for s in all_stats)
        print(f"{'='*52}")
        print(f"  GRAND TOTAL: {total_removed} files removed, {total_valid} valid guides remain")
        print(f"{'='*52}")


if __name__ == "__main__":
    main()
