# Business Entity Resolution Pipeline

## Overview
This package contains the complete, self-contained end-to-end entity resolution pipeline for the Amazon ML Challenge 2026.
It takes noisy business records from 3 independent sources (Source 1 reference, Source 2 and Source 3 targets) across US, India, and France (open-set country), and identifies all matching records.

## Architecture
1. **Normalization**: Vectorized cleaning, abbreviation expansion (Corp/Corporation, Rd/Road, etc.), legal suffix extraction, country-agnostic tokenization.
2. **Blocking / Candidate Generation**: Multi-key inverted index (sorted-token n-grams, longest-token prefix, rare name tokens, numeric address tokens, address pairs, domain stripping).
3. **Candidate Screening**: High-throughput two-stage scoring using `rapidfuzz` string similarity to prioritize top candidates.
4. **Feature Engineering**: 20 generic, country-agnostic lexical, character n-gram, address overlap, token sort, and legal metadata features.
5. **Machine Learning Model**: Native LightGBM GBDT booster trained with positive class weighting and tuned on held-out grouped validation entities.
6. **Evaluation Metric**: Macro-averaged entity-level $F_{0.5}$ (weights precision 2x over recall, penalizing false merges).
7. **Streaming Inference**: Batch-streamed candidate scoring and writing directly to `candidate_pairs.tsv` and `matching_results.tsv` to ensure stable memory usage under 3 GB.

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Reproduce End-to-End
Run from the `student_resource/` directory:

1. **Full Pipeline Execution**:
```bash
python3 code/business_entity_resolution/src/run_inference.py
```

2. **Validate Submission Files**:
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Output Artefacts
- `output/matching_results.tsv`: Final predictions (`source1_entity_id`, `matched_entity_ids`).
- `output/candidate_pairs.tsv`: Final candidate set fed to the classifier (`source1_entity_id`, `candidate_entity_ids`).
