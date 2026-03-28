# -*- coding: utf-8 -*-
"""
Unified Keyword-Item Mapping (replaces keywordmapping70.py … keywordmapping95.py)

Configurable via CLI arguments or direct function calls.
Supports running a single threshold or sweeping all thresholds at once.

Usage:
  # Single threshold
  python keywordmapping.py --threshold 90

  # All thresholds (sweep mode)
  python keywordmapping.py --sweep

  # Custom threshold list
  python keywordmapping.py --thresholds 70 80 90

Inputs (same folder as this script):
 - classified_nonstopwords_gemini_output.csv  (column: 'Words')
 - all_item_130868.csv                         (columns: 'ชื่อชุดการแสดง', 'คำอธิบายชุดการแสดง')

Outputs (per threshold T):
 - mapped_words_to_itemsT.csv   : ลำดับ | ชื่อชุดการแสดง | words
 - unmapped_wordsT.csv          : คำที่ยังไม่ถูกแมป
 - unmapped_itemsT.csv          : ชื่อชุดการแสดงที่ยังไม่ถูกแมป
"""

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

# ═══════════════════════════════════════════════════════════════
# DEFAULT CONFIG (override via CLI or function arguments)
# ═══════════════════════════════════════════════════════════════
DEFAULT_THRESHOLDS = [70, 75, 80, 85, 90, 95]

WORDS_FILE = "classified_nonstopwords_gemini_output.csv"
ITEMS_FILE = "all_item_130868.csv"

OUT_MAIN_PATTERN = "mapped_words_to_items{threshold}.csv"
OUT_UNMAPPED_WORDS_PATTERN = "unmapped_words{threshold}.csv"
OUT_UNMAPPED_ITEMS_PATTERN = "unmapped_items{threshold}.csv"

# ═══════════════════════════════════════════════════════════════
# LOCKED COLUMNS
# ═══════════════════════════════════════════════════════════════
COL_WORDS = "Words"
COL_ITEM_NAME = "ชื่อชุดการแสดง"
COL_DESC = "คำอธิบายชุดการแสดง"

USE_FUZZY = True
RE_KEEP_CHARS = re.compile(r"[^0-9A-Za-zก-๙\s]", re.UNICODE)

# ═══════════════════════════════════════════════════════════════
# SYNONYMS (shared across all thresholds — single source of truth)
# ═══════════════════════════════════════════════════════════════
SYNONYMS: Dict[str, List[str]] = {
    "มโนราห์": ["โนรา", "มโนรา"],
    "ไทยทรงดำ": ["ไทดำ", "ไทยดำ", "ลาวโซ่ง", "โซ่ง"],
    "พระอิศวร": ["พระศิวะ", "ศิวะ", "อิศวร", "อีศวร"],
    "พระศิวะ": ["พระอิศวร", "ศิวะ", "อิศวร", "อีศวร"],
    "สีน้ำเงิน": ["น้ำเงิน", "คราม", "สีคราม"],
    "ฤดูฝน": ["หน้าฝน", "มรสุม", "ฤดูมรสุม"],
    "ความมีน้ำใจ": ["น้ำใจ", "เอื้อเฟื้อเผื่อแผ่"],
    "หล่อสำริด": ["หล่อบรอนซ์", "บรอนซ์", "ทองสัมฤทธิ์", "สำริด"],
    "เพลงอัตราจังหวะสองชั้น": ["สองชั้น", "อัตราสองชั้น", "ชั้นสอง"],
    "แม่น้ำโขง": ["โขง"],
    "รำลึก": ["ระลึก", "น้อมรำลึก", "น้อมระลึก"],
    "กิเนสส์บุ๊ค": ["กินเนสส์เวิลด์เรคคอร์ดส์", "กินเนสเวิลด์เรคคอร์ดส์", "กินเนสส์บุ๊ก", "กินเนสบุ๊ค"],
    "วิ่งหน้า": ["วิ่งไปข้างหน้า", "วิ่งไปด้านหน้า", "วิ่งนำ", "วิ่งหน้า-ถอยหลัง"],
    "สัตว์น้ำ": ["สัตว์ทะเล", "ปลา", "สัตว์น้ำจืด", "สัตว์น้ำเค็ม", "สิ่งมีชีวิตน้ำ"],
    "แหล่งน้ำ": ["แม่น้ำ", "ลำธาร", "คลอง", "ทะเลสาบ", "หนอง", "บึง", "แอ่งน้ำ", "แหล่งน้ำธรรมชาติ"],
}


# ═══════════════════════════════════════════════════════════════
# TEXT UTILITIES
# ═══════════════════════════════════════════════════════════════

def normalize_text(s: str) -> str:
    """Normalize: keep bracket contents, remove only bracket characters;
    strip symbols; lowercase; collapse spaces."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    s = re.sub(r"[()\[\]{}＜＞《》]", " ", s)
    s = s.replace("\n", " ").replace("\t", " ")
    s = RE_KEEP_CHARS.sub(" ", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _thai_affix_variants(w: str) -> List[str]:
    """Simple Thai morphological heuristics."""
    variants = set()
    if w.startswith("ความ") and len(w) > 4:
        variants.add(w[4:])
        if w.startswith("ความมี") and len(w) > 7:
            variants.add(w[7:])
    if w.startswith("การ") and len(w) > 3:
        variants.add(w[3:])
    if w.startswith("สี") and len(w) > 2:
        variants.add(w[1:])
    return [v for v in variants if v]


def generate_variants(word: str) -> List[str]:
    """Generate all matching variants for a word."""
    w_norm = normalize_text(word)
    vs = {w_norm}
    if " " in w_norm:
        vs.add(w_norm.replace(" ", ""))

    for hx in _thai_affix_variants(w_norm):
        vs.add(hx)
        if " " in hx:
            vs.add(hx.replace(" ", ""))

    for syn in SYNONYMS.get(word, []):
        syn_norm = normalize_text(syn)
        vs.add(syn_norm)
        if " " in syn_norm:
            vs.add(syn_norm.replace(" ", ""))
        for hx in _thai_affix_variants(syn_norm):
            vs.add(hx)
            if " " in hx:
                vs.add(hx.replace(" ", ""))

    return [x for x in vs if x]


def try_import_rapidfuzz():
    if not USE_FUZZY:
        return None
    try:
        from rapidfuzz import fuzz
        return fuzz
    except Exception:
        return None


def text_contains_any(target_text: str, patterns: List[str], fuzz_mod, threshold: int) -> bool:
    """Check if any pattern appears in target_text."""
    if not target_text:
        return False
    if fuzz_mod:
        for p in patterns:
            if not p:
                continue
            if fuzz_mod.partial_ratio(p, target_text) >= threshold:
                return True
        return False
    return any(p and (p in target_text) for p in patterns)


# ═══════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════

def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    return header, rows


def load_words(words_csv: Path) -> List[str]:
    header, rows = read_csv(words_csv)
    if COL_WORDS not in header:
        raise KeyError(f"ไม่พบคอลัมน์ '{COL_WORDS}' ในไฟล์ {words_csv.name} (พบ: {header})")
    words: List[str] = []
    for r in rows:
        raw = (r.get(COL_WORDS) or "").strip()
        if not raw:
            continue
        if any(sep in raw for sep in [",", ";", "|", "/"]):
            parts = re.split(r"[,\|;/]+", raw)
        else:
            parts = [raw]
        for p in parts:
            p = p.strip()
            if p:
                words.append(p)
    seen, uniq = set(), []
    for w in words:
        if w not in seen:
            uniq.append(w); seen.add(w)
    return uniq


def load_items(items_csv: Path) -> Tuple[List[Dict[str, str]], str, str]:
    header, rows = read_csv(items_csv)
    if COL_ITEM_NAME not in header:
        raise KeyError(f"ไม่พบคอลัมน์ '{COL_ITEM_NAME}' ในไฟล์ {items_csv.name} (พบ: {header})")
    if COL_DESC not in header:
        print(f"[warn] ไม่พบคอลัมน์ '{COL_DESC}' ในไฟล์ {items_csv.name} จะใช้เฉพาะชื่อชุดการแสดง",
              file=sys.stderr)
    return rows, COL_ITEM_NAME, (COL_DESC if COL_DESC in header else "")


# ═══════════════════════════════════════════════════════════════
# CORE MAPPER — parameterised by threshold
# ═══════════════════════════════════════════════════════════════

def map_words_to_items(
    words: List[str],
    items: List[Dict[str, str]],
    name_col: str,
    desc_col: Optional[str],
    fuzzy_threshold: int = 90,
) -> Tuple[List[Tuple[int, str, List[str]]], List[str], List[str]]:
    """Run keyword–item mapping with a given fuzzy_threshold.

    Returns:
        results:        list of (index, item_name, [matched_words])
        unmapped_words: words that didn't match any item
        unmapped_items: items that weren't matched by any word
    """
    fuzz_mod = try_import_rapidfuzz()

    prepared_items = []
    for r in items:
        original_name = (r.get(name_col) or "").strip()
        name_norm = normalize_text(original_name)
        desc_norm = normalize_text(r.get(desc_col, "")) if desc_col else ""
        combined = f"{name_norm} {desc_norm}".strip()
        prepared_items.append((original_name, combined))

    word_to_variants = {w: generate_variants(w) for w in words}

    name_to_words = defaultdict(set)
    for original_name, combined_text in prepared_items:
        if not original_name:
            continue
        for original_word, variants in word_to_variants.items():
            if text_contains_any(combined_text, variants, fuzz_mod, fuzzy_threshold):
                name_to_words[original_name].add(original_word)

    results = []
    idx = 1
    for name, wset in name_to_words.items():
        wsorted = sorted(wset, key=lambda x: normalize_text(x))
        results.append((idx, name, wsorted))
        idx += 1

    mapped_words = set().union(*name_to_words.values()) if name_to_words else set()
    unmapped_words = [w for w in words if w not in mapped_words]

    all_item_names = [(r.get(name_col) or "").strip() for r in items]
    unmapped_items = [n for n in all_item_names if n and n not in name_to_words]

    return results, unmapped_words, unmapped_items


# ═══════════════════════════════════════════════════════════════
# OUTPUT WRITERS
# ═══════════════════════════════════════════════════════════════

def write_outputs(
    results: List[Tuple[int, str, List[str]]],
    unmapped_words: List[str],
    unmapped_items: List[str],
    out_main: Path,
    out_unmapped_words: Path,
    out_unmapped_items: Path,
):
    with open(out_main, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ลำดับ", "ชื่อชุดการแสดง", "words"])
        for idx, name, words in results:
            w.writerow([idx, name, ", ".join(words)])

    with open(out_unmapped_words, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unmapped_word"])
        for t in unmapped_words:
            w.writerow([t])

    with open(out_unmapped_items, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unmapped_item_name"])
        for n in unmapped_items:
            w.writerow([n])


# ═══════════════════════════════════════════════════════════════
# PUBLIC API: run_mapping()  — call from other scripts or notebooks
# ═══════════════════════════════════════════════════════════════

def run_mapping(
    threshold: int = 90,
    base_dir: Optional[Path] = None,
    words_file: str = WORDS_FILE,
    items_file: str = ITEMS_FILE,
    write_files: bool = True,
) -> Dict:
    """Run keyword mapping for a single threshold.

    Args:
        threshold:   fuzzy threshold (0–100)
        base_dir:    directory containing input files (default: script dir)
        words_file:  CSV with words column
        items_file:  CSV with items
        write_files: whether to write output CSVs

    Returns:
        dict with keys: threshold, results, unmapped_words, unmapped_items,
                        n_words, n_items, n_mapped_items
    """
    base = base_dir or (Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd())
    words_path = base / words_file
    items_path = base / items_file

    if not words_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ {words_path}")
    if not items_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ {items_path}")

    words = load_words(words_path)
    items, name_col, desc_col = load_items(items_path)

    results, unmapped_words, unmapped_items = map_words_to_items(
        words, items, name_col, desc_col or None, fuzzy_threshold=threshold
    )

    if write_files:
        out_main = base / OUT_MAIN_PATTERN.format(threshold=threshold)
        out_uw = base / OUT_UNMAPPED_WORDS_PATTERN.format(threshold=threshold)
        out_ui = base / OUT_UNMAPPED_ITEMS_PATTERN.format(threshold=threshold)
        write_outputs(results, unmapped_words, unmapped_items, out_main, out_uw, out_ui)

    return {
        "threshold": threshold,
        "results": results,
        "unmapped_words": unmapped_words,
        "unmapped_items": unmapped_items,
        "n_words": len(words),
        "n_items": len(items),
        "n_mapped_items": len(results),
        "n_mapped_words": len(words) - len(unmapped_words),
    }


def run_sweep(
    thresholds: Optional[List[int]] = None,
    base_dir: Optional[Path] = None,
    write_files: bool = True,
) -> List[Dict]:
    """Run mapping for multiple thresholds (sweep).

    Returns:
        list of dicts (one per threshold) from run_mapping()
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    base = base_dir or (Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd())

    all_results = []
    for t in thresholds:
        print(f"\n{'='*50}")
        print(f"  Running threshold = {t}")
        print(f"{'='*50}")
        res = run_mapping(threshold=t, base_dir=base, write_files=write_files)
        all_results.append(res)

        print(f"  Mapped items : {res['n_mapped_items']}/{res['n_items']}")
        print(f"  Mapped words : {res['n_mapped_words']}/{res['n_words']}")

    return all_results


# ═══════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unified Keyword-Item Mapping (configurable threshold)"
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--threshold", "-t", type=int, default=None,
                       help="Single fuzzy threshold (0-100)")
    group.add_argument("--sweep", action="store_true",
                       help="Run all default thresholds (70,75,80,85,90,95)")
    parser.add_argument("--thresholds", nargs="+", type=int, default=None,
                        help="Custom list of thresholds to sweep")
    parser.add_argument("--dir", type=str, default=None,
                        help="Base directory for input/output files")
    args = parser.parse_args()

    base = Path(args.dir) if args.dir else None

    if args.thresholds:
        results = run_sweep(thresholds=args.thresholds, base_dir=base)
    elif args.sweep:
        results = run_sweep(base_dir=base)
    elif args.threshold is not None:
        res = run_mapping(threshold=args.threshold, base_dir=base)
        results = [res]
    else:
        # Default: sweep all thresholds
        results = run_sweep(base_dir=base)

    # Print summary table
    print("\n" + "=" * 70)
    print("  SWEEP SUMMARY")
    print("=" * 70)
    print(f"  {'Threshold':>10}  {'Mapped Items':>14}  {'Mapped Words':>14}  {'Unmapped Words':>16}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*16}")
    for r in results:
        print(f"  {r['threshold']:>10}  {r['n_mapped_items']:>14}  "
              f"{r['n_mapped_words']:>14}  {len(r['unmapped_words']):>16}")
    print("=" * 70)


if __name__ == "__main__":
    main()
