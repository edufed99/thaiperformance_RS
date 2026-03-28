#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
llm_screening_report.py
========================
Generates a comprehensive LLM Screening Report for the Thai NLP pipeline.

Analyses the 4-model stopword-screening results:
  Gemini 2.5 Pro, GPT-4.1, Qwen3-235B, DeepSeek-R1-0528

Reports:
  1) Retain/Remove counts per runner (+ seed pre-filter)
  2) Audit log: removal reasons by rubric category + schema-valid rate
  3) Inter-model agreement (Cohen's κ)
  4) Downstream evaluation (coherence, entropy, ModelScore)
"""

import os, csv, json
from pathlib import Path
from collections import Counter, defaultdict

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

BASE = Path(__file__).resolve().parent

# ── Files ──
CLUSTER_CSV      = BASE / "cluster_results.csv"
RUNLOG_CSV       = BASE / "stopwords_runlog.csv"
RUNMETA_JSON     = BASE / "stopwords_runmeta.json"
EVAL_SUMMARY     = BASE / "model_eval_summary_wcb.csv"
EVAL_RANKED      = BASE / "model_eval_ranked_wcb.csv"
PAIRWISE_KAPPA   = BASE / "model_pairwise_kappa.csv"
COHERENCE_CSV    = BASE / "coherence_per_group_wcb.csv"

MODELS = {
    "Gemini":   {"stopwords": "stopwords_analysis_gemini.csv",
                 "nonstop":   "non_stopwords_gemini.csv",
                 "classified": "classified_nonstopwords_gemini_output.csv",
                 "full_name": "Gemini 2.5 Pro"},
    "GPT":      {"stopwords": "stopwords_analysis_GPT.csv",
                 "nonstop":   "non_stopwords_GPT.csv",
                 "classified": "classified_nonstopwords_GPT_output.csv",
                 "full_name": "GPT-4.1"},
    "Qwen":     {"stopwords": "stopwords_analysis_QWEN.csv",
                 "nonstop":   "non_stopwords_QWEN.csv",
                 "classified": "classified_nonstopwords_QWEN_output.csv",
                 "full_name": "Qwen3-235B-A22B"},
    "DeepSeek": {"stopwords": "stopwords_analysis_DEEPSEEK.csv",
                 "nonstop":   "non_stopwords_DEEPSEEK.csv",
                 "classified": "classified_nonstopwords_DEEPSEEK_output.csv",
                 "full_name": "DeepSeek-R1-0528"},
}

# The same SEED_STOPWORDS used across all 4 scripts (count from code)
SEED_SW_COUNT = 196  # counted from the shared set in the scripts


def read_csv_safe(path):
    p = BASE / path if not isinstance(path, Path) else path
    if not p.exists():
        return None
    return pd.read_csv(p, encoding="utf-8-sig")


def main():
    sep  = "=" * 72
    sep2 = "-" * 72

    print(sep)
    print("  LLM SCREENING REPORT — Process A / Step 3: Data Label")
    print("  Domain-specific stopword identification via 4 LLM runners")
    print(sep)

    # ── 0. Corpus snapshot ──
    cluster_df = read_csv_safe(CLUSTER_CSV)
    total_words = len(cluster_df) if cluster_df is not None else 0
    n_clusters  = cluster_df["cluster_label"].nunique() if cluster_df is not None else 0
    print(f"\n[Corpus] {total_words} words (from K-Means clustering, K={n_clusters})")
    print(f"[Seed pre-filter] {SEED_SW_COUNT} manually curated stopwords removed before LLM call")
    words_sent_to_llm = total_words - SEED_SW_COUNT
    print(f"[Words sent to LLM] {words_sent_to_llm} words per runner")

    # ── 1. Retain / Remove per runner ──
    print(f"\n{sep2}")
    print("  1. RETAIN vs REMOVE per Runner")
    print(sep2)

    rubric_names = [
        "คำที่มีความหมายกว้าง (Umbrella Terms)",
        "คำเชิงนามธรรม หรือ คำเชิงประเมินค่า (Abstract or Evaluative Terms)",
        "คำเฉพาะโดเมนที่พบบ่อยเกินไป (Domain-Specific Overuse)",
        "คำซ้ำซ้อน หรือ คำพ้องความหมาย (Redundant or Synonym)",
    ]
    rubric_short = ["Umbrella", "Abstract/Eval", "Domain-Overuse", "Redundant/Syn"]

    summary_rows = []
    all_stopword_dfs = {}

    for model_key, info in MODELS.items():
        sw_df  = read_csv_safe(info["stopwords"])
        ns_df  = read_csv_safe(info["nonstop"])

        llm_removed = len(sw_df) if sw_df is not None else 0
        retained    = len(ns_df) if ns_df is not None else 0
        total_removed = SEED_SW_COUNT + llm_removed

        # Schema-valid rate: rows with non-empty word + stopword_type + reason
        schema_valid = 0
        if sw_df is not None and len(sw_df) > 0:
            valid_mask = (
                sw_df["word"].notna() & (sw_df["word"].astype(str).str.strip() != "") &
                sw_df["stopword_type"].notna() & (sw_df["stopword_type"].astype(str).str.strip() != "") &
                sw_df["reason"].notna() & (sw_df["reason"].astype(str).str.strip() != "")
            )
            schema_valid = valid_mask.sum()
            all_stopword_dfs[model_key] = sw_df

        schema_rate = schema_valid / llm_removed if llm_removed > 0 else 0.0

        summary_rows.append({
            "model": model_key,
            "full_name": info["full_name"],
            "seed_removed": SEED_SW_COUNT,
            "llm_removed": llm_removed,
            "total_removed": total_removed,
            "retained": retained,
            "schema_valid": schema_valid,
            "schema_rate": schema_rate,
        })

    # Print table
    print(f"\n  {'Model':<12s} {'Full Name':<22s} {'Seed':>5s} {'LLM↓':>6s} "
          f"{'Total↓':>7s} {'Retain':>7s} {'Valid%':>7s}")
    print(f"  {'-'*12} {'-'*22} {'-'*5} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")
    for r in summary_rows:
        print(f"  {r['model']:<12s} {r['full_name']:<22s} {r['seed_removed']:>5d} "
              f"{r['llm_removed']:>6d} {r['total_removed']:>7d} {r['retained']:>7d} "
              f"{r['schema_rate']:>6.1%}")

    print(f"\n  Interpretation:")
    most_aggressive = max(summary_rows, key=lambda x: x["llm_removed"])
    most_conservative = min(summary_rows, key=lambda x: x["llm_removed"])
    print(f"    Most aggressive:   {most_aggressive['model']} "
          f"(removed {most_aggressive['llm_removed']} by LLM → retained {most_aggressive['retained']})")
    print(f"    Most conservative: {most_conservative['model']} "
          f"(removed {most_conservative['llm_removed']} by LLM → retained {most_conservative['retained']})")

    # ── 2. Audit Log: Removal reasons by rubric ──
    print(f"\n{sep2}")
    print("  2. AUDIT LOG — Removal Reasons by Rubric Category")
    print(sep2)

    for model_key in MODELS:
        sw_df = all_stopword_dfs.get(model_key)
        if sw_df is None or len(sw_df) == 0:
            print(f"\n  [{model_key}] No data")
            continue

        total = len(sw_df)
        print(f"\n  [{model_key}]  {total} words flagged as stopword")

        # Normalize stopword_type to match rubric
        type_counts = defaultdict(int)
        type_examples = defaultdict(list)
        unmatched_types = defaultdict(int)

        for _, row in sw_df.iterrows():
            st = str(row.get("stopword_type", "")).strip()
            word = str(row.get("word", "")).strip()
            reason = str(row.get("reason", "")).strip()
            matched = False
            for i, rname in enumerate(rubric_names):
                # Fuzzy match: check if key part of rubric name appears
                if any(key in st for key in [
                    "ความหมายกว้าง", "Umbrella",
                    "นามธรรม", "ประเมินค่า", "Abstract", "Evaluative",
                    "โดเมน", "Domain", "บ่อยเกิน", "Overuse",
                    "ซ้ำซ้อน", "พ้องความหมาย", "Redundant", "Synonym",
                ][i*2:(i+1)*2]):
                    type_counts[rubric_short[i]] += 1
                    if len(type_examples[rubric_short[i]]) < 3:
                        type_examples[rubric_short[i]].append(f"{word}: {reason[:40]}")
                    matched = True
                    break
            if not matched:
                # Try broader matching
                if "กว้าง" in st or "Umbrella" in st:
                    type_counts[rubric_short[0]] += 1
                    matched = True
                elif "นามธรรม" in st or "ประเมิน" in st or "Abstract" in st or "Evaluat" in st:
                    type_counts[rubric_short[1]] += 1
                    matched = True
                elif "โดเมน" in st or "Domain" in st or "บ่อย" in st:
                    type_counts[rubric_short[2]] += 1
                    matched = True
                elif "ซ้ำ" in st or "พ้อง" in st or "Redundant" in st or "Synonym" in st:
                    type_counts[rubric_short[3]] += 1
                    matched = True
            if not matched:
                unmatched_types[st] += 1

        for i, short in enumerate(rubric_short):
            cnt = type_counts.get(short, 0)
            pct = cnt / total * 100 if total > 0 else 0
            exs = type_examples.get(short, [])
            ex_str = " | ".join(exs) if exs else "-"
            print(f"    {short:<18s}  {cnt:>4d} ({pct:5.1f}%)  e.g. {ex_str}")

        if unmatched_types:
            unk = sum(unmatched_types.values())
            print(f"    {'(unmatched)':<18s}  {unk:>4d} ({unk/total*100:5.1f}%)")

    # ── 3. Schema-valid rate detail ──
    print(f"\n{sep2}")
    print("  3. SCHEMA VALIDATION (per runner)")
    print(sep2)

    print(f"\n  Expected schema: word | cluster_label | stopword_type | reason")
    print(f"  A row is 'valid' if all 4 fields are non-empty and parseable.\n")

    for r in summary_rows:
        sw_df = all_stopword_dfs.get(r["model"])
        n_rows = len(sw_df) if sw_df is not None else 0
        # Check for duplicate words
        n_dup = 0
        if sw_df is not None and len(sw_df) > 0:
            n_dup = sw_df["word"].duplicated().sum()

        print(f"  {r['model']:<12s}  rows={n_rows:>4d}  valid={r['schema_valid']:>4d}  "
              f"rate={r['schema_rate']:.1%}  duplicates={n_dup}")

    # ── 4. Inter-model agreement (pairwise κ) ──
    print(f"\n{sep2}")
    print("  4. INTER-MODEL AGREEMENT (Cohen's κ)")
    print(sep2)

    kappa_df = read_csv_safe(PAIRWISE_KAPPA)
    if kappa_df is not None:
        # Clean up index
        idx_col = [c for c in kappa_df.columns if "Unnamed" in c]
        if idx_col:
            kappa_df = kappa_df.set_index(idx_col[0])
        print(f"\n  {'':>12s}", end="")
        for col in kappa_df.columns:
            print(f"  {col:>10s}", end="")
        print()
        for idx, row in kappa_df.iterrows():
            print(f"  {str(idx):>12s}", end="")
            for col in kappa_df.columns:
                val = row[col]
                if str(idx) == col:
                    print(f"  {'---':>10s}", end="")
                else:
                    print(f"  {val:>10.3f}", end="")
            print()

        # Average κ
        vals = []
        for i, r1 in enumerate(kappa_df.index):
            for j, c1 in enumerate(kappa_df.columns):
                if i < j:
                    vals.append(kappa_df.iloc[i][c1])
        avg_kappa = sum(vals) / len(vals) if vals else 0
        print(f"\n  Average pairwise κ = {avg_kappa:.3f}", end="")
        if avg_kappa >= 0.61:
            print("  (substantial agreement)")
        elif avg_kappa >= 0.41:
            print("  (moderate agreement)")
        else:
            print("  (fair agreement)")

    # ── 5. Downstream evaluation (ModelScore) ──
    print(f"\n{sep2}")
    print("  5. DOWNSTREAM EVALUATION (WangchanBERTa coherence)")
    print(sep2)

    eval_df = read_csv_safe(EVAL_RANKED)
    if eval_df is not None:
        idx_col = [c for c in eval_df.columns if "Unnamed" in c]
        if idx_col:
            eval_df = eval_df.rename(columns={idx_col[0]: "model"})

        print(f"\n  {'Rank':>4s}  {'Model':<10s} {'Words':>6s} {'SubGrp':>7s} "
              f"{'Entropy':>8s} {'Coherence':>10s} {'MajMatch':>9s} {'Score':>8s} {'ParseRate':>10s}")
        print(f"  {'-'*4}  {'-'*10} {'-'*6} {'-'*7} {'-'*8} {'-'*10} {'-'*9} {'-'*8} {'-'*10}")
        for rank, (_, row) in enumerate(eval_df.iterrows(), 1):
            model = row.get("model", row.iloc[0]) if "model" in eval_df.columns else row.iloc[0]
            print(f"  {rank:>4d}  {str(model):<10s} "
                  f"{int(row.get('n_words', 0)):>6d} "
                  f"{int(row.get('n_subgroups_used', 0)):>7d} "
                  f"{row.get('entropy_bits', 0):>8.3f} "
                  f"{row.get('coherence_wcb', 0):>10.4f} "
                  f"{row.get('majority_match_rate', 0):>9.1%} "
                  f"{row.get('ModelScore', 0):>8.4f} "
                  f"{row.get('coverage_parse_rate', 0):>10.1%}")

        print(f"\n  Metrics explanation:")
        print(f"    coverage_parse_rate: fraction of input tokens parseable (schema compliance)")
        print(f"    entropy_bits:        label distribution entropy (higher = more diverse labeling)")
        print(f"    coherence_wcb:       avg WangchanBERTa cosine similarity within sub-groups")
        print(f"    majority_match_rate: fraction of sub-groups where majority matches reference")
        print(f"    ModelScore:          composite score (weighted combination of above)")

    # ── 6. Coherence per semantic group (best model) ──
    print(f"\n{sep2}")
    print("  6. SEMANTIC COHERENCE per Sub-Group (Top Model: Gemini)")
    print(sep2)

    coh_df = read_csv_safe(COHERENCE_CSV)
    if coh_df is not None:
        gemini_coh = coh_df[coh_df["model"] == "Gemini"].sort_values("coherence", ascending=False)
        if len(gemini_coh) > 0:
            print(f"\n  {'Sub-Group':<40s} {'Coherence':>10s} {'Size':>5s}")
            print(f"  {'-'*40} {'-'*10} {'-'*5}")
            for _, row in gemini_coh.iterrows():
                print(f"  {row['group']:<40s} {row['coherence']:>10.4f} {int(row['size']):>5d}")
            avg_coh = gemini_coh["coherence"].mean()
            print(f"\n  Average coherence (Gemini): {avg_coh:.4f}")

    # ── 7. Run metadata ──
    print(f"\n{sep2}")
    print("  7. RUN METADATA")
    print(sep2)

    # Runlog
    runlog_df = read_csv_safe(RUNLOG_CSV)
    if runlog_df is not None and len(runlog_df) > 0:
        n_batches = len(runlog_df)
        n_success = (runlog_df["status"] == "success").sum()
        n_failed  = (runlog_df["status"] != "success").sum()
        max_attempts = runlog_df["attempts"].max()
        print(f"\n  Batch runlog (DeepSeek/Qwen chunked mode):")
        print(f"    Total batches: {n_batches}  success: {n_success}  failed: {n_failed}")
        print(f"    Max attempts per batch: {int(max_attempts)}")
        print(f"    Chunk size: {int(runlog_df['size'].mode().iloc[0])}")

    # Runmeta
    if RUNMETA_JSON.exists():
        with open(RUNMETA_JSON, encoding="utf-8") as f:
            meta = json.load(f)
        print(f"\n  Run metadata (last chunked run):")
        for k, v in meta.items():
            print(f"    {k}: {v}")

    print(f"\n  Model configurations (shared across all runners):")
    print(f"    API endpoint:   ModelHarbor v1/chat/completions")
    print(f"    Temperature:    0.1")
    print(f"    Prompt:         4-rubric classification (Umbrella / Abstract / Domain-Overuse / Redundant)")
    print(f"    Seed stopwords: {SEED_SW_COUNT} pre-filtered before LLM call")

    # ── 8. Export ──
    csv_path = BASE / "llm_screening_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Full Name", "Seed Removed", "LLM Removed",
                         "Total Removed", "Retained", "Schema Valid", "Schema Rate"])
        for r in summary_rows:
            writer.writerow([r["model"], r["full_name"], r["seed_removed"],
                             r["llm_removed"], r["total_removed"], r["retained"],
                             r["schema_valid"], f"{r['schema_rate']:.4f}"])
    print(f"\n  --> Report CSV saved: {csv_path.name}")

    print(f"\n{sep}")
    print("  END OF REPORT")
    print(sep)


if __name__ == "__main__":
    main()
