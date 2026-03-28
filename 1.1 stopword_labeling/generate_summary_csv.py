"""
Generate comprehensive summary CSV for LLM keyword screening process.
Outputs:
  1) W0 / K per cluster per LLM  (|Ck|, |Ck'|, %removed)
  2) Removal-reason taxonomy with example words
  3) Inter-LLM agreement (pairwise Cohen's κ, majority-vote rate)
"""

import pathlib, re
import pandas as pd
import numpy as np

DIR = pathlib.Path(__file__).resolve().parent

# ── helpers ──────────────────────────────────────────────────────────
MODELS = ["Gemini", "GPT", "DeepSeek", "Qwen"]
STOP_FILES = {
    "Gemini":  "stopwords_analysis_gemini.csv",
    "GPT":     "stopwords_analysis_GPT.csv",
    "DeepSeek":"stopwords_analysis_DEEPSEEK.csv",
    "Qwen":    "stopwords_analysis_QWEN.csv",
}
NON_STOP_FILES = {
    "Gemini":  "non_stopwords_gemini.csv",
    "GPT":     "non_stopwords_GPT.csv",
    "DeepSeek":"non_stopwords_DEEPSEEK.csv",
    "Qwen":    "non_stopwords_QWEN.csv",
}

# Canonical reason categories
CANON = {
    "umbrella":          "Umbrella Terms",
    "evaluative":        "Abstract / Evaluative",
    "domain-overuse":    "Domain-Specific Overuse",
    "redundant":         "Redundant / Synonym",
}

def _canon(raw: str) -> str:
    """Map a raw stopword_type string to one of 4 canonical categories."""
    s = str(raw).lower()
    if "umbrella" in s or "ความหมายกว้าง" in s:
        return "umbrella"
    if "evaluative" in s or "นามธรรม" in s or "ประเมินค่า" in s:
        return "evaluative"
    if "domain" in s or "โดเมน" in s or "overuse" in s:
        return "domain-overuse"
    if "redundant" in s or "synonym" in s or "ซ้ำ" in s or "พ้อง" in s:
        return "redundant"
    return "other"


# ── 1. Load cluster_results  (= W0 universe) ────────────────────────
cluster_df = pd.read_csv(DIR / "cluster_results.csv")
W0_total = len(cluster_df)
W0_per_cluster = cluster_df.groupby("cluster_label").size().to_dict()
num_clusters = len(W0_per_cluster)

# ── 2. Per-LLM: retained / removed counts per cluster ───────────────
rows_cluster = []
for model in MODELS:
    # Non-stopwords = retained keywords
    ns = pd.read_csv(DIR / NON_STOP_FILES[model])
    K_total = len(ns)
    K_per_cluster = ns.groupby("cluster_label").size().to_dict()

    for ck in sorted(W0_per_cluster):
        w0 = W0_per_cluster[ck]
        k  = K_per_cluster.get(ck, 0)
        removed = w0 - k
        pct = removed / w0 * 100 if w0 else 0
        rows_cluster.append({
            "LLM":           model,
            "cluster":       ck,
            "|Ck| (W0)":     w0,
            "|Ck'| (K)":     k,
            "removed":       removed,
            "%removed":      round(pct, 1),
        })
    # totals
    removed_total = W0_total - K_total
    rows_cluster.append({
        "LLM":           model,
        "cluster":       "TOTAL",
        "|Ck| (W0)":     W0_total,
        "|Ck'| (K)":     K_total,
        "removed":       removed_total,
        "%removed":      round(removed_total / W0_total * 100, 1),
    })

df_cluster = pd.DataFrame(rows_cluster)

# ── 3. Removal-reason taxonomy with example words (5-10 per type) ────
rows_reason = []
for model in MODELS:
    sw = pd.read_csv(DIR / STOP_FILES[model])
    # Filter out metadata / note rows
    sw = sw[sw["word"].notna() & ~sw["word"].str.startswith("*") &
            ~sw["word"].str.startswith("Example") &
            ~sw["word"].str.startswith("Umbrella") &
            ~sw["word"].str.startswith("Abstract") &
            ~sw["word"].str.startswith("Domain") &
            ~sw["word"].str.startswith("Redundant") &
            ~sw["word"].str.startswith("Definitions")].copy()
    sw["canon"] = sw["stopword_type"].apply(_canon)
    sw = sw[sw["canon"] != "other"]

    for canon_key, canon_label in CANON.items():
        subset = sw[sw["canon"] == canon_key]
        n = len(subset)
        examples = subset["word"].drop_duplicates().head(10).tolist()
        rows_reason.append({
            "LLM":             model,
            "reason_category": canon_label,
            "count":           n,
            "example_words":   " | ".join(examples),
        })

df_reason = pd.DataFrame(rows_reason)

# ── 4. Inter-LLM agreement ──────────────────────────────────────────
# Build binary decision matrix  (word × model)  1 = keep, 0 = remove
all_words = sorted(cluster_df["word"].unique())
decisions = pd.DataFrame(index=all_words)

for model in MODELS:
    ns = pd.read_csv(DIR / NON_STOP_FILES[model])
    kept = set(ns["word"].dropna())
    decisions[model] = [1 if w in kept else 0 for w in all_words]

# majority vote
decisions["majority_keep"] = (decisions[MODELS].sum(axis=1) >= 3).astype(int)

# majority-match rate per model
rows_agree = []
for model in MODELS:
    match_rate = (decisions[model] == decisions["majority_keep"]).mean()
    rows_agree.append({"LLM": model, "majority_match_rate": round(match_rate, 4)})

# pairwise Cohen's κ
kappa_df = pd.read_csv(DIR / "model_pairwise_kappa.csv", index_col=0)
rows_kappa = []
done = set()
for m1 in MODELS:
    for m2 in MODELS:
        if m1 >= m2:
            continue
        pair = f"{m1}–{m2}"
        if pair in done:
            continue
        done.add(pair)
        k = kappa_df.loc[m1, m2] if m1 in kappa_df.index and m2 in kappa_df.columns else np.nan
        rows_kappa.append({"pair": pair, "Cohen_kappa": round(k, 4)})

df_agree = pd.DataFrame(rows_agree)
df_kappa = pd.DataFrame(rows_kappa)

# overall majority-keep count
majority_keep_count = int(decisions["majority_keep"].sum())
majority_remove_count = len(all_words) - majority_keep_count

# ── 5. Write a single CSV with multiple sections ────────────────────
out = DIR / "data_label_summary.csv"
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    f.write("=" * 80 + "\n")
    f.write("SECTION 1: Word counts before/after screening per cluster per LLM\n")
    f.write(f"Total words before screening (W0): {W0_total}\n")
    f.write(f"Number of clusters: {num_clusters}\n")
    f.write("=" * 80 + "\n")
    df_cluster.to_csv(f, index=False)

    f.write("\n")
    f.write("=" * 80 + "\n")
    f.write("SECTION 2: Removal reason categories with example words\n")
    f.write("=" * 80 + "\n")
    df_reason.to_csv(f, index=False)

    f.write("\n")
    f.write("=" * 80 + "\n")
    f.write("SECTION 3: Inter-LLM agreement\n")
    f.write("=" * 80 + "\n")

    f.write("\n--- 3a. Majority-vote match rate per model ---\n")
    f.write(f"Majority-vote rule: word kept if >= 3 of 4 LLMs keep it\n")
    f.write(f"Majority-keep words: {majority_keep_count},  Majority-remove words: {majority_remove_count}\n")
    df_agree.to_csv(f, index=False)

    f.write("\n--- 3b. Pairwise Cohen's kappa ---\n")
    df_kappa.to_csv(f, index=False)

    f.write("\n--- 3c. Model evaluation summary (from model_eval_ranked_wcb.csv) ---\n")
    eval_df = pd.read_csv(DIR / "model_eval_ranked_wcb.csv", index_col=0)
    eval_df.to_csv(f)

print(f"✅  Summary written to {out}")
print(f"   W0 = {W0_total}  |  clusters = {num_clusters}")
for m in MODELS:
    ns = pd.read_csv(DIR / NON_STOP_FILES[m])
    print(f"   {m:10s}  K = {len(ns):>4d}  removed = {W0_total - len(ns):>4d}  ({(W0_total - len(ns)) / W0_total * 100:.1f}%)")
