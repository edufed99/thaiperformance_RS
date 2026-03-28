# -*- coding: utf-8 -*-
"""
Generate  C) Fuzzy Keyword–Item Mapping  summary CSV
=====================================================
Sections produced
  C1  Threshold-sweep results  (τ = 70 → 95)
  C2  Matching-method note     (no β / embedding — pure fuzzy string)
  C3  Mapping statistics per threshold
  C4  Sample correct & incorrect mappings (τ = 90)

Output → mapping_summary.csv  (UTF-8 BOM)
"""

import csv, os
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent

# ── file patterns ──
EVAL_SUMMARY  = BASE / "mapping_evaluation_filteredgold_summary.csv"
MAPPED_PAT    = "mapped_words_to_items{}.csv"
UNMAP_W_PAT   = "unmapped_words{}.csv"
UNMAP_I_PAT   = "unmapped_items{}.csv"
ERROR_RPT     = BASE / "error_report_threshold_90.csv"
NONSTOP_FILE  = BASE / "classified_nonstopwords_gemini_output.csv"

THRESHOLDS = [70, 75, 80, 85, 90, 95]
OUT_FILE   = BASE / "mapping_summary.csv"

# ─────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────

def read_csv_rows(path):
    """Return (header, rows) with utf-8-sig."""
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def count_data_rows(path):
    """Count non-header rows in a CSV (returns 0 if file empty / header only)."""
    if not path.exists():
        return -1  # missing
    _, rows = read_csv_rows(path)
    return len(rows)


def load_nonstop_count():
    """Total non-stopwords (K) from Gemini classified output."""
    _, rows = read_csv_rows(NONSTOP_FILE)
    words = set()
    for r in rows:
        # 'Words' column (index varies — search header)
        pass
    # Re-read with header awareness
    with open(NONSTOP_FILE, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("Words", "")
            if raw:
                for w in raw.split(","):
                    w = w.strip()
                    if w:
                        words.add(w)
    return len(words)


# ─────────────────────────────────────────────
# C1  Threshold sweep
# ─────────────────────────────────────────────

def section_c1(writer):
    writer.writerow(["=== C1: Threshold Sweep Results (τ = 70 → 95) ==="])
    writer.writerow([])

    header, rows = read_csv_rows(EVAL_SUMMARY)
    # header: threshold,TP,FP,FN,Precision,Recall,F1  (possibly more cols)
    writer.writerow(header + ["best?"])

    # Find best F1
    f1_idx = header.index("F1")
    best_f1 = max(float(r[f1_idx]) for r in rows)

    for r in rows:
        is_best = "★" if float(r[f1_idx]) == best_f1 else ""
        writer.writerow(r + [is_best])

    writer.writerow([])
    writer.writerow(["Note: best threshold by F1 is τ = 90 "
                     "(Precision 0.8284, Recall 0.9701, F1 0.8937)"])
    writer.writerow([])


# ─────────────────────────────────────────────
# C2  Matching method note
# ─────────────────────────────────────────────

def section_c2(writer):
    writer.writerow(["=== C2: Matching Method Description ==="])
    writer.writerow([])
    writer.writerow(["field", "value"])
    info = [
        ("method", "Fuzzy string matching (rapidfuzz.fuzz.partial_ratio)"),
        ("library", "rapidfuzz  (Python)"),
        ("score_range", "0 – 100  (threshold τ applied as lower bound)"),
        ("synonym_expansion", "Yes — manual SYNONYMS dict (≥ 15 entries)"),
        ("thai_affix_variants", "Yes — ความ / การ / สี prefixes auto-generated"),
        ("text_normalisation", "lowercase + whitespace collapse"),
        ("β parameter", "NOT applicable — no embedding / hybrid weighting in this pipeline"),
        ("embedding_similarity", "NOT used — mapping is string-only"),
        ("note", "The paper's Eq 4–5 β comparison (string vs embedding vs hybrid) "
                 "was NOT implemented in this codebase; only string-based fuzzy "
                 "matching is present."),
    ]
    for k, v in info:
        writer.writerow([k, v])
    writer.writerow([])


# ─────────────────────────────────────────────
# C3  Mapping statistics per threshold
# ─────────────────────────────────────────────

def section_c3(writer):
    writer.writerow(["=== C3: Mapping Statistics per Threshold ==="])
    writer.writerow([])

    total_kw = load_nonstop_count()
    writer.writerow(["total_keywords_(K)", total_kw])

    header_out = [
        "threshold",
        "mapped_items",
        "total_links",
        "avg_keywords_per_item",
        "avg_items_per_keyword",
        "unmapped_words",
        "unmapped_items",
        "pct_unmapped_words",
        "pct_unmapped_items",
    ]
    writer.writerow(header_out)

    for t in THRESHOLDS:
        mapped_path = BASE / MAPPED_PAT.format(t)
        uw_path     = BASE / UNMAP_W_PAT.format(t)
        ui_path     = BASE / UNMAP_I_PAT.format(t)

        # -- mapped file --
        _, rows = read_csv_rows(mapped_path)
        n_items = len(rows)

        total_links = 0
        kw_to_items = defaultdict(int)
        for r in rows:
            words_str = r[2] if len(r) > 2 else ""
            wlist = [w.strip() for w in words_str.split(",") if w.strip()]
            total_links += len(wlist)
            for w in wlist:
                kw_to_items[w] += 1

        avg_kw_per_item = round(total_links / n_items, 2) if n_items else 0
        n_mapped_kw = len(kw_to_items)
        avg_items_per_kw = round(total_links / n_mapped_kw, 2) if n_mapped_kw else 0

        # -- unmapped counts --
        n_uw = count_data_rows(uw_path)
        n_ui = count_data_rows(ui_path)

        pct_uw = round(100 * n_uw / total_kw, 2) if total_kw else ""
        # total items (use mapped + unmapped)
        total_items_approx = n_items + (n_ui if n_ui >= 0 else 0)
        pct_ui = round(100 * n_ui / total_items_approx, 2) if total_items_approx else ""

        writer.writerow([
            t, n_items, total_links,
            avg_kw_per_item, avg_items_per_kw,
            n_uw, n_ui, pct_uw, pct_ui,
        ])

    writer.writerow([])


# ─────────────────────────────────────────────
# C4  Sample correct / incorrect mappings (τ=90)
# ─────────────────────────────────────────────

def section_c4(writer):
    writer.writerow(["=== C4: Sample Correct & Incorrect Mappings (τ = 90) ==="])
    writer.writerow([])

    _, rows = read_csv_rows(ERROR_RPT)
    # columns: item, gold_tokens, predicted_tokens, TP, FP, FN, FP_words, FN_words

    correct  = []
    incorrect = []
    for r in rows:
        item = r[0]
        tp   = int(r[3])
        fp   = int(r[4])
        fn   = int(r[5])
        fp_w = r[6] if len(r) > 6 else ""
        fn_w = r[7] if len(r) > 7 else ""
        if fp == 0 and fn == 0 and tp > 0:
            correct.append((item, tp, fp, fn, fp_w, fn_w))
        elif fp > 0 or fn > 0:
            incorrect.append((item, tp, fp, fn, fp_w, fn_w))

    # ── C4a  Correct (exact match) ──
    writer.writerow(["--- C4a: Correct mappings (FP = 0, FN = 0) — sample 15 ---"])
    writer.writerow(["item", "TP", "FP", "FN", "FP_words", "FN_words"])
    for row in correct[:15]:
        writer.writerow(list(row))
    writer.writerow([f"(total exact-match items: {len(correct)} / {len(rows)})"])
    writer.writerow([])

    # ── C4b  Incorrect (has FP or FN) ──
    # Sort by descending (FP+FN) for most informative examples
    incorrect.sort(key=lambda x: x[2] + x[3], reverse=True)
    writer.writerow(["--- C4b: Incorrect mappings (FP > 0 or FN > 0) — sample 15 ---"])
    writer.writerow(["item", "TP", "FP", "FN", "FP_words", "FN_words"])
    for row in incorrect[:15]:
        writer.writerow(list(row))
    writer.writerow([f"(total items with errors: {len(incorrect)} / {len(rows)})"])
    writer.writerow([])

    # ── C4c  Error-type frequency ──
    writer.writerow(["--- C4c: Most frequent FP words (over-mapped) ---"])
    writer.writerow(["FP_word", "count"])
    fp_freq = defaultdict(int)
    for r in rows:
        fp_w = r[6] if len(r) > 6 else ""
        for w in fp_w.split(","):
            w = w.strip()
            if w:
                fp_freq[w] += 1
    for w, c in sorted(fp_freq.items(), key=lambda x: -x[1])[:15]:
        writer.writerow([w, c])
    writer.writerow([])

    writer.writerow(["--- C4d: Most frequent FN words (missed) ---"])
    writer.writerow(["FN_word", "count"])
    fn_freq = defaultdict(int)
    for r in rows:
        fn_w = r[7] if len(r) > 7 else ""
        for w in fn_w.split(","):
            w = w.strip()
            if w:
                fn_freq[w] += 1
    for w, c in sorted(fn_freq.items(), key=lambda x: -x[1])[:15]:
        writer.writerow([w, c])
    writer.writerow([])


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main():
    with open(OUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["C) Fuzzy Keyword-Item Mapping Summary"])
        w.writerow(["Generated by generate_mapping_summary.py"])
        w.writerow([])

        section_c1(w)
        section_c2(w)
        section_c3(w)
        section_c4(w)

    print(f"✅ Written → {OUT_FILE}")


if __name__ == "__main__":
    main()
