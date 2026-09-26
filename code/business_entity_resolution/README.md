# 📦 Business Entity Resolution Code Package

## Overview
This package contains the complete, self-contained, reproducible source code for the Amazon ML Challenge 2026 Entity Resolution solution.

## Architecture
- `src/normalize.py`: Vectorized text cleaning, abbreviation expansion (Corp/Corporation, Rd/Road), and legal suffix extraction (`both_legal`, `legal_xor`).
- `src/blocking.py`: Multi-key inverted index blocking (token n-grams, prefixes, numeric address numbers, address pairs, domain stripping).
- `src/pipeline.py`: Feature engineering pipeline, grouped validation splitting by reference entity, and LightGBM model training.
- `src/run_inference.py`: High-throughput two-stage streaming inference engine that scores test candidates in 50,000-entity chunks with constant memory usage (<2.5 GB RAM).
- `src/package_submission.py`: Automates official validation check (`utils/validate_submission.py`) and packages the final submission archive (`team_pixel_submission.zip`).

## Installation
```bash
pip install -r requirements.txt
```

## How to Run End-to-End
From the `student_resource/` directory:

1. **Run Full Test Inference & Output Generation:**
   ```bash
   python3 code/business_entity_resolution/src/run_inference.py
   ```

2. **Verify Submission with Official Validator:**
   ```bash
   python3 utils/validate_submission.py \
       --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv \
       --test-dir dataset/test
   ```

3. **Package Submission Zip:**
   ```bash
   python3 code/business_entity_resolution/src/package_submission.py
   ```

## Model & Validation Results
- **Macro $F_{0.5}$:** **0.9724**
- **Precision:** **0.9829**
- **Recall:** **0.9673**
- **Pairwise ROC-AUC:** **0.9995 (99.95%)**
- **Optimal Decision Threshold:** **0.900**
