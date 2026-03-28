# Thai Performance Recommendation Dataset and Code

A reproducible research repository for **semantic artifact construction** and **eligibility-gated hybrid recommendation** in Thai performing arts.

This repository provides the datasets, semantic processing pipeline, and recommendation system implementation used in the study:

**“Eligibility-Gated Model for Hybrid Recommendation”**

---

## Overview

Thai performing arts recommendations differ from conventional recommendation systems because recommendation quality depends not only on relevance but also on **contextual appropriateness**.

This repository supports a full end-to-end workflow:

**clustered Thai keywords → semantic artifact construction → keyword–item semantic linking → context-valid candidate generation → hybrid recommendation evaluation**

---

## Core Datasets

### `all_item_130868.csv`
Curated Thai performing arts item database containing **1,022 records corresponding to 114 unique items**.  
Used for:
- item representation
- semantic grounding
- candidate generation

### `user_log_with_keywords_only_list.csv`
User interaction log containing **2,574 records from 156 users**.  
Used for:
- user profiling
- collaborative modeling
- hybrid recommendation
- offline evaluation

### `cluster_results.csv`
Clustered Thai keyword artifact derived from prior experiments.  
Serves as the **semantic input** for artifact construction.

### `golddata.csv`
Manually curated keyword–item mapping reference.  
Used for:
- calibration
- threshold tuning
- validation

---

## Repository Structure

### `1.semantic-artifact-code/`
Implements the **semantic artifact construction layer**, including:
- LLM-based stopword screening
- blueprint-constrained hierarchical labeling
- semantic evaluation across models
- keyword–item mapping
- threshold calibration

### `2.recommendation-code/`
Implements the **eligibility-gated recommendation layer**, including:
- context-valid candidate construction
- hybrid recommendation experiments
- evaluation outputs
- paper figures and tables

---

## Research Contribution

This repository supports the proposed framework where:

> **context validity is enforced before ranking**

Unlike conventional recommendation pipelines, this design:
- restricts candidate items before ranking
- reduces context policy violations
- enables fair comparison across models under shared constraints

---

## Reproducibility

The repository is designed for reproducible research:
- shared datasets across all experiments
- consistent evaluation settings
- fixed random seeds
- comparable candidate constraints

---

## Intended Use

This repository is intended for:
- research reproducibility
- semantic artifact inspection
- recommendation system experimentation
- cultural heritage recommendation studies
- extension to context-aware recommendation domains

---

## Note

All datasets and code were developed and curated for research on Thai performing arts recommendations.

---

## Citation

If you use this repository, please cite:

**Pichaya Dumnil**,  
“Thai Performance Recommendation Dataset and Code,”  
GitHub repository, 2026.  
[Online]. Available: https://github.com/edufed99/thaiperformance_RS
