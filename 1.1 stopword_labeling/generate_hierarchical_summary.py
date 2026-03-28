"""
Generate Section B – Hierarchical Labeling under Blueprint
===========================================================
Outputs:
  B1) Total paths |P| in blueprint (Main → Sub → SubGroup)
  B2) Word distribution per Main / Sub / Sub Group  (per LLM, before–after view)
  B3) Quality metrics per leaf group: coherence, entropy, imbalance, top groups + examples
  B4) Example output table (first ~40 leaf-groups with words)
"""

import pathlib, csv, io, math
import pandas as pd
import numpy as np

DIR = pathlib.Path(__file__).resolve().parent

# ── Model config ─────────────────────────────────────────────────────
MODELS = ["Gemini", "GPT", "DeepSeek", "Qwen"]
CLASSIFIED_FILES = {
    "Gemini":  "classified_nonstopwords_gemini_output.csv",
    "GPT":     "classified_nonstopwords_GPT_output.csv",
    "DeepSeek":"classified_nonstopwords_DEEPSEEK_output.csv",
    "Qwen":    "classified_nonstopwords_QWEN_output.csv",
}

# ── Load taxonomy_mapping_report (has summary row) ───────────────────
tax_raw = DIR / "taxonomy_mapping_report.csv"
tax_lines = tax_raw.read_text(encoding="utf-8-sig").splitlines()

# Parse the TAXONOMY SUMMARY section
tax_rows = []
in_tax = False
for line in tax_lines:
    if line.startswith("Model,"):
        in_tax = True
        header = line.split(",")
        continue
    if in_tax:
        if line.strip() == "" or line.startswith("==="):
            in_tax = False
            continue
        tax_rows.append(line.split(",", len(header)-1))

tax_df = pd.DataFrame(tax_rows, columns=header)
for c in ["Main Labels", "Sub Labels", "Leaf Labels", "Total Words"]:
    tax_df[c] = pd.to_numeric(tax_df[c], errors="coerce")

# ── Helper: parse classified CSV → per-word rows ─────────────────────
def load_classified(model: str) -> pd.DataFrame:
    """Return DataFrame with columns: Main Label, Sub Label, Sub Group, word"""
    fp = DIR / CLASSIFIED_FILES[model]
    df = pd.read_csv(fp)
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    # Remove clearly invalid rows
    df = df[df["Sub Group"].notna()].copy()
    # Explode words
    rows = []
    for _, r in df.iterrows():
        main = str(r["Main Label"]).strip()
        sub  = str(r["Sub Label"]).strip()
        grp  = str(r["Sub Group"]).strip()
        words_str = str(r["Words"]).strip()
        if main == "..." or main == "nan":
            continue
        for w in words_str.split(","):
            w = w.strip().strip("()")
            if w and w != "nan":
                rows.append({"Main Label": main, "Sub Label": sub, "Sub Group": grp, "word": w})
    return pd.DataFrame(rows)


# ── Load coherence_per_group_wcb ─────────────────────────────────────
coh_df = pd.read_csv(DIR / "coherence_per_group_wcb.csv")

# ── Load cluster_results (= W0) ─────────────────────────────────────
cluster_df = pd.read_csv(DIR / "cluster_results.csv")
W0 = len(cluster_df)

# =====================================================================
# B1: Total paths |P|
# =====================================================================
all_classified = {}
for model in MODELS:
    all_classified[model] = load_classified(model)

# Use Gemini (highest-scoring model) as the blueprint reference
ref = all_classified["Gemini"]
paths = ref[["Main Label", "Sub Label", "Sub Group"]].drop_duplicates()
P_total = len(paths)
main_labels = paths["Main Label"].nunique()
sub_labels  = paths["Sub Label"].nunique()
leaf_labels = paths["Sub Group"].nunique()

# =====================================================================
# B2: Word distribution per Main / Sub / SubGroup  (per LLM)
# =====================================================================

# 2a: Per-model distribution at Main Label level
rows_main = []
for model in MODELS:
    df = all_classified[model]
    K = len(df)
    dist = df.groupby("Main Label").size().reset_index(name="word_count")
    for _, r in dist.iterrows():
        rows_main.append({
            "LLM":        model,
            "Main Label": r["Main Label"],
            "word_count":  int(r["word_count"]),
            "pct":         round(r["word_count"] / K * 100, 1),
        })
df_main = pd.DataFrame(rows_main)

# 2b: Per-model distribution at Sub Label level
rows_sub = []
for model in MODELS:
    df = all_classified[model]
    K = len(df)
    dist = df.groupby(["Main Label", "Sub Label"]).size().reset_index(name="word_count")
    for _, r in dist.iterrows():
        rows_sub.append({
            "LLM":        model,
            "Main Label": r["Main Label"],
            "Sub Label":  r["Sub Label"],
            "word_count":  int(r["word_count"]),
            "pct":         round(r["word_count"] / K * 100, 1),
        })
df_sub = pd.DataFrame(rows_sub)

# 2c: Per-model distribution at Sub Group (leaf) level
rows_leaf = []
for model in MODELS:
    df = all_classified[model]
    K = len(df)
    dist = df.groupby(["Main Label", "Sub Label", "Sub Group"]).size().reset_index(name="word_count")
    for _, r in dist.iterrows():
        rows_leaf.append({
            "LLM":        model,
            "Main Label": r["Main Label"],
            "Sub Label":  r["Sub Label"],
            "Sub Group":  r["Sub Group"],
            "word_count":  int(r["word_count"]),
            "pct":         round(r["word_count"] / K * 100, 1),
        })
df_leaf = pd.DataFrame(rows_leaf)

# =====================================================================
# B3: Quality metrics – coherence, entropy, imbalance per model
# =====================================================================
rows_quality = []
for model in MODELS:
    df = all_classified[model]
    K = len(df)
    # number of unique paths
    p = df[["Main Label", "Sub Label", "Sub Group"]].drop_duplicates()
    n_main = p["Main Label"].nunique()
    n_sub  = p["Sub Label"].nunique()
    n_leaf = len(p)

    # leaf-level size distribution
    leaf_sizes = df.groupby("Sub Group").size()
    probs = leaf_sizes / leaf_sizes.sum()
    entropy = -sum(p_i * math.log2(p_i) for p_i in probs if p_i > 0)
    max_share = leaf_sizes.max() / leaf_sizes.sum()
    gini = 1 - sum(p_i**2 for p_i in probs)

    # coherence from coherence_per_group_wcb.csv
    coh_model = coh_df[coh_df["model"] == model]
    mean_coh = coh_model["coherence"].mean() if len(coh_model) > 0 else np.nan
    min_coh  = coh_model["coherence"].min() if len(coh_model) > 0 else np.nan
    max_coh  = coh_model["coherence"].max() if len(coh_model) > 0 else np.nan

    rows_quality.append({
        "LLM":              model,
        "W0":               W0,
        "K (retained)":     K,
        "%retained":        round(K / W0 * 100, 1),
        "|P| paths":        n_leaf,
        "Main labels":      n_main,
        "Sub labels":       n_sub,
        "Leaf labels":      n_leaf,
        "Entropy (bits)":   round(entropy, 3),
        "Max group share":  round(max_share, 4),
        "Gini impurity":    round(gini, 4),
        "Leaf min size":    int(leaf_sizes.min()),
        "Leaf max size":    int(leaf_sizes.max()),
        "Leaf mean size":   round(leaf_sizes.mean(), 1),
        "Leaf std":         round(leaf_sizes.std(), 1),
        "Coherence mean":   round(mean_coh, 4) if not np.isnan(mean_coh) else "",
        "Coherence min":    round(min_coh, 4) if not np.isnan(min_coh) else "",
        "Coherence max":    round(max_coh, 4) if not np.isnan(max_coh) else "",
    })

df_quality = pd.DataFrame(rows_quality)

# =====================================================================
# B3b: Top/Bottom 5 leaf groups (by coherence) for Gemini (best model)
# =====================================================================
coh_gemini = coh_df[coh_df["model"] == "Gemini"].copy()
coh_gemini = coh_gemini.sort_values("coherence", ascending=False)
top5 = coh_gemini.head(5)
bot5 = coh_gemini.tail(5)

# =====================================================================
# B4: Example output table – full taxonomy with example words (Gemini)
# =====================================================================
gemini_cls = all_classified["Gemini"]
rows_example = []
for _, r in paths.iterrows():
    main = r["Main Label"]
    sub  = r["Sub Label"]
    grp  = r["Sub Group"]
    subset = gemini_cls[
        (gemini_cls["Main Label"] == main) &
        (gemini_cls["Sub Label"]  == sub) &
        (gemini_cls["Sub Group"]  == grp)
    ]
    words = subset["word"].tolist()
    n = len(words)
    examples = ", ".join(words[:8])
    if n > 8:
        examples += f", ... (+{n-8} more)"
    rows_example.append({
        "Main Label": main,
        "Sub Label":  sub,
        "Sub Group":  grp,
        "n_words":    n,
        "example_words": examples,
    })

df_example = pd.DataFrame(rows_example)

# =====================================================================
# WRITE OUTPUT
# =====================================================================
out = DIR / "hierarchical_labeling_summary.csv"
with open(out, "w", encoding="utf-8-sig", newline="") as f:

    f.write("=" * 100 + "\n")
    f.write("SECTION B: Hierarchical Labeling under Blueprint\n")
    f.write("=" * 100 + "\n\n")

    # ── B1 ──
    f.write("-" * 100 + "\n")
    f.write("B1) Blueprint Structure (reference: Gemini – best ModelScore)\n")
    f.write("-" * 100 + "\n")
    f.write(f"Total leaf paths |P|:  {P_total}\n")
    f.write(f"Main labels:           {main_labels}\n")
    f.write(f"Sub labels:            {sub_labels}\n")
    f.write(f"Leaf (Sub Group) labels: {leaf_labels}\n")
    f.write(f"Total words before screening (W0): {W0}\n\n")

    f.write("Per-model taxonomy size comparison:\n")
    tax_df.to_csv(f, index=False)
    f.write("\n")

    # ── B2a: Main Label distribution ──
    f.write("-" * 100 + "\n")
    f.write("B2a) Word Distribution per Main Label (after screening, per LLM)\n")
    f.write("-" * 100 + "\n")
    pivot_main = df_main.pivot_table(index="Main Label", columns="LLM",
                                      values="word_count", fill_value=0)
    pivot_main = pivot_main[MODELS]
    pivot_main.to_csv(f)
    f.write("\n")

    # ── B2b: Sub Label distribution ──
    f.write("-" * 100 + "\n")
    f.write("B2b) Word Distribution per Sub Label (after screening, per LLM)\n")
    f.write("-" * 100 + "\n")
    pivot_sub = df_sub.pivot_table(index=["Main Label", "Sub Label"], columns="LLM",
                                    values="word_count", fill_value=0)
    pivot_sub = pivot_sub[MODELS]
    pivot_sub.to_csv(f)
    f.write("\n")

    # ── B2c: Leaf distribution (Gemini only for brevity) ──
    f.write("-" * 100 + "\n")
    f.write("B2c) Word Distribution per Leaf Group (all LLMs)\n")
    f.write("-" * 100 + "\n")
    pivot_leaf = df_leaf.pivot_table(index=["Main Label", "Sub Label", "Sub Group"],
                                     columns="LLM", values="word_count", fill_value=0)
    pivot_leaf = pivot_leaf[MODELS]
    pivot_leaf.to_csv(f)
    f.write("\n")

    # ── B3: Quality metrics ──
    f.write("-" * 100 + "\n")
    f.write("B3) Quality Metrics per LLM\n")
    f.write("-" * 100 + "\n")
    df_quality.to_csv(f, index=False)
    f.write("\n")

    # ── B3b: Top/Bottom coherence groups ──
    f.write("-" * 100 + "\n")
    f.write("B3b) Top-5 & Bottom-5 Leaf Groups by Coherence (Gemini)\n")
    f.write("-" * 100 + "\n")
    f.write("\nTop-5 (highest coherence):\n")
    top5[["group", "coherence", "size"]].to_csv(f, index=False)
    f.write("\nBottom-5 (lowest coherence):\n")
    bot5[["group", "coherence", "size"]].to_csv(f, index=False)
    f.write("\n")

    # ── B3c: Coherence per leaf group (all models) ──
    f.write("-" * 100 + "\n")
    f.write("B3c) Coherence per Leaf Group (all models)\n")
    f.write("-" * 100 + "\n")
    coh_pivot = coh_df.pivot_table(index="group", columns="model",
                                    values="coherence", aggfunc="first")
    coh_pivot["mean"] = coh_pivot.mean(axis=1)
    coh_pivot = coh_pivot.sort_values("mean", ascending=False)
    coh_pivot.to_csv(f, float_format="%.4f")
    f.write("\n")

    # ── B4: Example output table ──
    f.write("=" * 100 + "\n")
    f.write("B4) Example Output: Full Taxonomy with Example Words (Gemini – best model)\n")
    f.write("=" * 100 + "\n")
    df_example.to_csv(f, index=False)

print(f"✅  Written to {out}")
print(f"   |P| = {P_total}  ({main_labels} Main → {sub_labels} Sub → {leaf_labels} Leaf)")
for model in MODELS:
    df = all_classified[model]
    n_paths = df[["Main Label", "Sub Label", "Sub Group"]].drop_duplicates().shape[0]
    print(f"   {model:10s}  K={len(df):>4d}  paths={n_paths}")
