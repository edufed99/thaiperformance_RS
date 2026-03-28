# Semantic Artifact Code

This folder contains the implementation of the **semantic artifact construction stage** used in the study:

**"Eligibility-Gated Model for Hybrid Recommendation"**

This stage transforms clustered Thai keywords into structured and reusable semantic artifacts for downstream recommendation.

## Main Components

### `1.1 stopword_labeling/`
This module performs:
- LLM-based stopword screening
- blueprint-constrained hierarchical labeling
- evaluation and summary reporting

Main scripts include:
- stopword filtering with Gemini, GPT, Qwen, and DeepSeek
- hierarchical labeling with Gemini, GPT, Qwen, and DeepSeek
- evaluation and reporting scripts

Supporting folders:
- `input/`
- `result/`

### `1.2 keyword-item mapping/`
This module performs:
- keyword-to-item semantic linking
- threshold-based mapping experiments
- mapping evaluation and summary generation

Main scripts include:
- `keywordmapping.py`
- `generate_mapping_summary.py`

Supporting folders:
- `input/`
- `result/`

## Role in the Framework

This folder implements the **semantic artifact layer** of the proposed framework.  
Its outputs are used to:
- refine clustered Thai keyword artifacts
- construct structured semantic representations
- link semantic units to actual items
- support context-valid candidate construction in the recommendation layer

## Inputs
Typical inputs include:
- clustered Thai keyword artifacts
- taxonomy definitions
- item metadata
- gold mapping references

## Outputs
Typical outputs include:
- screened non-stopword keyword sets
- hierarchical labeling outputs
- evaluation summaries across LLMs
- keyword-item mapping outputs
- threshold comparison reports

## Note
This folder focuses only on **semantic artifact preparation**.  
The downstream recommendation models are implemented separately in the recommendation module.
