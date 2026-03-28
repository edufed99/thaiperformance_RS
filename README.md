# Recommendation Code

This folder contains the implementation of the **eligibility-gated hybrid recommendation module** used in the study:

**“Eligibility-Gated Model for Hybrid Recommendation”**

## Overview

This module generates **context-valid recommendations** by combining:
- semantic signals from keyword–item mappings
- user interaction evidence
- contextual constraints applied before ranking

The recommendation workflow includes:
1. candidate construction
2. context-aware eligibility gating
3. hybrid ranking and evaluation

## Structure

- `input/`  
  Input data for recommendation, including item metadata, semantic mappings, and user interaction logs

- `output/`  
  Experimental outputs, including comparison tables, evaluation summaries, and paper figures

- `recommender.ipynb`  
  Main notebook for recommendation experiments and evaluation

- `experiment_config.json`  
  Configuration file for experimental settings

## Input Files

- `all_item_130868.csv`  
  Item metadata used for candidate generation

- `mapped_words_to_items90.csv`  
  Keyword–item semantic mapping under threshold 0.90

- `user_log_with_keywords_only_list.csv`  
  User interaction log in keyword-based representation

## Role in the Framework

This module corresponds to the **recommendation layer** of the proposed framework.  
Its main design principle is:

> **context validity is enforced before ranking**

This allows the system to evaluate semantic, collaborative, and hybrid recommenders under the same context-constrained candidate space.

## Note

This module is designed to be used together with the semantic artifact outputs generated in:

`1.semantic-artifact-code/`
