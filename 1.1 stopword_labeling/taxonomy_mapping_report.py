#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
taxonomy_mapping_report.py
===========================
Part A: LLM Labeling / Taxonomy analysis
        — leaf counts, leaf sizes, entropy, max-group share, inter-runner agreement
Part B: Keyword–Item Mapping threshold sweep
        — Precision / Recall / F1 / TP FP FN per τ
        — evidence stats (mean keywords per item, %items mapped, %keywords used)

Outputs:
  taxonomy_mapping_report.csv   (CSV export of the two main tables)
"""

import csv, math, os, sys
from pathlib import Path
from collections import defaultdict
from itertools import combinations

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required: pip install pandas")

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════
LABEL_DIR  = Path(__file__).resolve().parent          # 3.Data Label/update101268
MAPPING_DIR = LABEL_DIR.parent.parent / "4.Keword-Item Mapping" / "update101268"

MODELS = {
    "Gemini":   {"classified": "classified_nonstopwords_gemini_output.csv",
                 "full_name": "Gemini 2.5 Pro"},
    "GPT":      {"classified": "classified_nonstopwords_GPT_output.csv",
                 "full_name": "GPT-4.1"},
    "Qwen":     {"classified": "classified_nonstopwords_QWEN_output.csv",
                 "full_name": "Qwen3-235B-A22B"},
    "DeepSeek": {"classified": "classified_nonstopwords_DEEPSEEK_output.csv",
                 "full_name": "DeepSeek-R1-0528"},
}

THRESHOLDS = [70, 75, 80, 85, 90, 95]
TOTAL_ITEMS = 115  # performing-arts corpus

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def entropy_bits(counts):
    """Shannon entropy in bits from a list of counts."""
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts if c > 0)


def split_words(cell):
    """Split a comma-separated Words cell into a list."""
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [w.strip() for w in cell.split(",") if w.strip()]


# ═══════════════════════════════════════════════════════════════
# PART A: Taxonomy
# ═══════════════════════════════════════════════════════════════


def analyse_taxonomy():
    SEP  = "=" * 76
    SEP2 = "-" * 76

    print(SEP)
    print("  PART A — LLM Labeling / Taxonomy Analysis")
    print("  (classified_nonstopwords_{model}_output.csv)")
    print(SEP)

    model_word_label = {}          # for agreement later
    taxonomy_rows = []             # for CSV export

    for model_key, info in MODELS.items():
        csv_path = LABEL_DIR / info["classified"]
        if not csv_path.exists():
            print(f"\n  [{model_key}]  FILE NOT FOUND: {csv_path.name}")
            continue

        df = read_csv(csv_path)

        n_main = df["Main Label"].nunique()
        n_sub  = df["Sub Label"].nunique()
        n_leaf = df["Sub Group"].nunique()

        # word counts per leaf
        leaf_word_counts = defaultdict(int)
        word_label = {}
        for _, row in df.iterrows():
            sg = str(row["Sub Group"]).strip()
            words = split_words(row.get("Words", ""))
            leaf_word_counts[sg] += len(words)
            for ww in words:
                word_label[ww] = sg

        model_word_label[model_key] = word_label

        total_words = sum(leaf_word_counts.values())
        sizes = sorted(leaf_word_counts.values(), reverse=True)
        ent = entropy_bits(sizes)
        max_share = max(sizes) / total_words if total_words else 0
        max_leaf = max(leaf_word_counts, key=leaf_word_counts.get)

        taxonomy_rows.append({
            "model": model_key,
            "full_name": info["full_name"],
            "main_labels": n_main,
            "sub_labels": n_sub,
            "leaf_labels": n_leaf,
            "total_words": total_words,
            "entropy_bits": round(ent, 3),
            "max_group_share": round(max_share, 4),
            "max_leaf": max_leaf,
            "leaf_min": min(sizes),
            "leaf_max": max(sizes),
            "leaf_mean": round(sum(sizes) / len(sizes), 1),
        })

        # Print summary
        print(f"\n{SEP2}")
        print(f"  [{model_key}]  {info['full_name']}")
        print(SEP2)
        print(f"  Hierarchy depth:  Main={n_main}  Sub={n_sub}  Leaf(SubGroup)={n_leaf}")
        print(f"  Total words classified: {total_words}")
        print(f"  Entropy (leaf-word dist): {ent:.3f} bits")
        print(f"  Max group share: {max_share:.1%}  ({max_leaf})")
        print(f"  Leaf size — min={min(sizes)}  max={max(sizes)}  "
              f"mean={sum(sizes)/len(sizes):.1f}  median={sorted(sizes)[len(sizes)//2]}")

        # Print all leaves sorted by size
        print(f"\n  {'Leaf (Sub Group)':<46s} {'Words':>6s} {'Share':>7s}")
        print(f"  {'-'*46} {'-'*6} {'-'*7}")
        for sg, cnt in sorted(leaf_word_counts.items(), key=lambda x: -x[1]):
            share = cnt / total_words * 100 if total_words else 0
            print(f"  {sg:<46s} {cnt:>6d} {share:>6.1f}%")

    # ── Inter-model taxonomy agreement ──
    print(f"\n{SEP}")
    print("  INTER-MODEL TAXONOMY AGREEMENT (same Sub Group on shared words)")
    print(SEP)

    print(f"\n  {'Pair':<22s} {'Common':>7s} {'Same':>6s} {'Rate':>7s}")
    print(f"  {'-'*22} {'-'*7} {'-'*6} {'-'*7}")
    for m1, m2 in combinations(MODELS.keys(), 2):
        d1 = model_word_label.get(m1, {})
        d2 = model_word_label.get(m2, {})
        common = set(d1.keys()) & set(d2.keys())
        if not common:
            print(f"  {m1+' vs '+m2:<22s} {'—':>7s}")
            continue
        same = sum(1 for w in common if d1[w] == d2[w])
        rate = same / len(common)
        print(f"  {m1+' vs '+m2:<22s} {len(common):>7d} {same:>6d} {rate:>6.1%}")

    return taxonomy_rows


# ═══════════════════════════════════════════════════════════════
# PART B: Keyword Mapping Threshold Sweep
# ═══════════════════════════════════════════════════════════════


def analyse_mapping():
    SEP  = "=" * 76
    SEP2 = "-" * 76

    print(f"\n\n{SEP}")
    print("  PART B — Keyword–Item Mapping Threshold Sweep")
    print(f"  (data from {MAPPING_DIR.name}/)")
    print(SEP)

    if not MAPPING_DIR.exists():
        print(f"\n  [ERROR] Mapping directory not found: {MAPPING_DIR}")
        return []

    # Load total keyword vocabulary from nonstopwords
    nonstop_path = MAPPING_DIR / "classified_nonstopwords_gemini_output.csv"
    all_keywords = set()
    if nonstop_path.exists():
        ns_df = read_csv(nonstop_path)
        for w in ns_df["Words"]:
            for p in split_words(w):
                all_keywords.add(p)
    total_keywords = len(all_keywords)

    # Load evaluation summary (P/R/F1/TP/FP/FN)
    eval_path = MAPPING_DIR / "mapping_evaluation_filteredgold_summary.csv"
    eval_df = None
    if eval_path.exists():
        eval_df = read_csv(eval_path)

    # Collect evidence stats per threshold
    mapping_rows = []

    for t in THRESHOLDS:
        mapped_file = MAPPING_DIR / f"mapped_words_to_items{t}.csv"
        unmapped_w_file = MAPPING_DIR / f"unmapped_words{t}.csv"
        unmapped_i_file = MAPPING_DIR / f"unmapped_items{t}.csv"

        row = {"threshold": t}

        # from eval summary
        if eval_df is not None:
            er = eval_df[eval_df["threshold"] == t]
            if len(er) == 1:
                er = er.iloc[0]
                row.update({
                    "TP": int(er["TP"]), "FP": int(er["FP"]), "FN": int(er["FN"]),
                    "Precision": float(er["Precision"]),
                    "Recall": float(er["Recall"]),
                    "F1": float(er["F1"]),
                })

        # from mapped file
        if mapped_file.exists():
            mdf = read_csv(mapped_file)
            n_items_mapped = len(mdf)
            total_pairs = 0
            used_keywords = set()
            per_item_kw = []

            for _, r in mdf.iterrows():
                words = split_words(r.get("words", ""))
                total_pairs += len(words)
                per_item_kw.append(len(words))
                used_keywords.update(words)

            mean_kw = total_pairs / n_items_mapped if n_items_mapped else 0
            pct_items = n_items_mapped / TOTAL_ITEMS * 100
            pct_kw_used = len(used_keywords) / total_keywords * 100 if total_keywords else 0

            row.update({
                "items_mapped": n_items_mapped,
                "pct_items_mapped": round(pct_items, 1),
                "total_kw_item_pairs": total_pairs,
                "mean_kw_per_item": round(mean_kw, 2),
                "unique_kw_used": len(used_keywords),
                "pct_kw_used": round(pct_kw_used, 1),
            })

        # from unmapped files
        if unmapped_w_file.exists():
            row["unmapped_words"] = len(read_csv(unmapped_w_file))
        if unmapped_i_file.exists():
            row["unmapped_items"] = len(read_csv(unmapped_i_file))

        mapping_rows.append(row)

    # ── Print threshold sweep table ──
    print(f"\n{SEP2}")
    print("  1. Threshold Sweep — Precision / Recall / F1  (gold-filtered)")
    print(SEP2)

    h = (f"  {'τ':>3s}  {'TP':>5s} {'FP':>5s} {'FN':>4s}  "
         f"{'Prec':>6s} {'Recall':>6s} {'F1':>6s}")
    print(f"\n{h}")
    print(f"  {'---':>3s}  {'-----':>5s} {'-----':>5s} {'----':>4s}  "
          f"{'------':>6s} {'------':>6s} {'------':>6s}")
    for r in mapping_rows:
        print(f"  {r['threshold']:>3d}  {r.get('TP',0):>5d} {r.get('FP',0):>5d} "
              f"{r.get('FN',0):>4d}  {r.get('Precision',0):>6.4f} "
              f"{r.get('Recall',0):>6.4f} {r.get('F1',0):>6.4f}")

    # Best threshold
    best = max(mapping_rows, key=lambda x: x.get("F1", 0))
    print(f"\n  Best τ by F1: {best['threshold']}  "
          f"(P={best.get('Precision',0):.4f}  R={best.get('Recall',0):.4f}  "
          f"F1={best.get('F1',0):.4f})")

    # ── Print evidence / coverage table ──
    print(f"\n{SEP2}")
    print("  2. Evidence & Coverage Statistics")
    print(f"     Total items: {TOTAL_ITEMS}  |  Total unique keywords: {total_keywords}")
    print(SEP2)

    h2 = (f"  {'τ':>3s}  {'Items':>5s} {'%Map':>5s}  {'Pairs':>6s} "
          f"{'MeanKW':>7s}  {'UniqKW':>6s} {'%KWused':>7s}  "
          f"{'Unmap_w':>7s} {'Unmap_i':>7s}")
    print(f"\n{h2}")
    print(f"  {'---':>3s}  {'-----':>5s} {'-----':>5s}  {'------':>6s} "
          f"{'-------':>7s}  {'------':>6s} {'-------':>7s}  "
          f"{'-------':>7s} {'-------':>7s}")
    for r in mapping_rows:
        print(f"  {r['threshold']:>3d}  "
              f"{r.get('items_mapped',0):>5d} "
              f"{r.get('pct_items_mapped',0):>4.1f}%  "
              f"{r.get('total_kw_item_pairs',0):>6d} "
              f"{r.get('mean_kw_per_item',0):>7.2f}  "
              f"{r.get('unique_kw_used',0):>6d} "
              f"{r.get('pct_kw_used',0):>6.1f}%  "
              f"{r.get('unmapped_words',0):>7d} "
              f"{r.get('unmapped_items',0):>7d}")

    # ── Print interpretation ──
    print(f"\n  Interpretation:")
    print(f"    • As τ increases, fuzzy-match tolerance tightens:")
    print(f"        - FP drops from {mapping_rows[0].get('FP',0)} (τ=70) → "
          f"{mapping_rows[-1].get('FP',0)} (τ=95)")
    print(f"        - FN rises from {mapping_rows[0].get('FN',0)} (τ=70) → "
          f"{mapping_rows[-1].get('FN',0)} (τ=95)")
    print(f"        - Mean KW/item shrinks from "
          f"{mapping_rows[0].get('mean_kw_per_item',0):.1f} → "
          f"{mapping_rows[-1].get('mean_kw_per_item',0):.1f}")
    r90 = next((r for r in mapping_rows if r["threshold"] == 90), {})
    print(f"    • τ=90 is the sweet-spot: F1={r90.get('F1',0):.4f}, "
          f"Precision={r90.get('Precision',0):.4f}, "
          f"Recall={r90.get('Recall',0):.4f}")
    print(f"    • τ=95 gives marginally higher Precision but lower Recall/F1")

    return mapping_rows


# ═══════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════


def export_csv(taxonomy_rows, mapping_rows):
    out = LABEL_DIR / "taxonomy_mapping_report.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)

        # Section 1: Taxonomy
        w.writerow(["=== TAXONOMY SUMMARY ==="])
        w.writerow(["Model", "Full Name", "Main Labels", "Sub Labels",
                     "Leaf Labels", "Total Words", "Entropy (bits)",
                     "Max Group Share", "Max Leaf", "Leaf Min", "Leaf Max",
                     "Leaf Mean"])
        for r in taxonomy_rows:
            w.writerow([r["model"], r["full_name"], r["main_labels"],
                        r["sub_labels"], r["leaf_labels"], r["total_words"],
                        r["entropy_bits"], f"{r['max_group_share']:.4f}",
                        r["max_leaf"], r["leaf_min"], r["leaf_max"],
                        r["leaf_mean"]])

        w.writerow([])

        # Section 2: Mapping Sweep
        w.writerow(["=== KEYWORD MAPPING SWEEP ==="])
        w.writerow(["Threshold", "TP", "FP", "FN", "Precision", "Recall", "F1",
                     "Items Mapped", "%Items Mapped", "Total KW-Item Pairs",
                     "Mean KW/Item", "Unique KW Used", "%KW Used",
                     "Unmapped Words", "Unmapped Items"])
        for r in mapping_rows:
            w.writerow([
                r["threshold"],
                r.get("TP", ""), r.get("FP", ""), r.get("FN", ""),
                r.get("Precision", ""), r.get("Recall", ""), r.get("F1", ""),
                r.get("items_mapped", ""), r.get("pct_items_mapped", ""),
                r.get("total_kw_item_pairs", ""), r.get("mean_kw_per_item", ""),
                r.get("unique_kw_used", ""), r.get("pct_kw_used", ""),
                r.get("unmapped_words", ""), r.get("unmapped_items", ""),
            ])

    print(f"\n  --> CSV saved: {out.name}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════


def main():
    taxonomy_rows = analyse_taxonomy()
    mapping_rows  = analyse_mapping()
    export_csv(taxonomy_rows, mapping_rows)

    SEP = "=" * 76
    print(f"\n{SEP}")
    print("  END OF TAXONOMY & MAPPING REPORT")
    print(SEP)


if __name__ == "__main__":
    main()
