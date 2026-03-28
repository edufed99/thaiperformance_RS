# Thai Performance Recommendation System Dataset

This repository contains the datasets and artifacts used in the study:

"Eligibility-Gated Model for Hybrid Recommendation"

## Contents

- all_item_130868.csv  
  Curated Thai performing arts item database containing 1,022 records corresponding to 114 unique items.  
  This dataset provides item metadata used for semantic grounding and context-aware candidate construction.

- user_log_with_keywords_only_list.csv  
  User interaction log containing 2,574 interaction records from 156 users.  
  This dataset supports user profiling, preference modeling, and offline recommendation evaluation.

- cluster_results.csv  
  Clustered Thai keyword artifact derived from prior keyword clustering experiments.  
  This dataset represents the semantic input used for constructing item-linked semantic artifacts.

- golddata.csv  
  Manually curated keyword–item mapping reference used for calibration and validation of semantic linking.

## Purpose

These datasets support the full pipeline of the proposed recommendation framework, including:
- Semantic artifact construction from Thai keyword clusters
- Keyword-to-item mapping and calibration
- Context-aware candidate selection
- Hybrid recommendation evaluation

## Reproducibility

All datasets are used consistently across experiments with fixed random seeds and shared evaluation splits to ensure reproducibility and fair comparison across models.

## Note

These datasets were constructed and curated by the authors for research purposes in Thai performing arts recommendation.

## Citation

If you use this dataset, please cite:

Pichaya Dumnil, "Thai Performance Recommendation Dataset," GitHub repository, 2026.  
[Online]. Available: https://github.com/edufed99/thaiperformance_RS
