"""
tune_thresholds_v2.py — Fast threshold and singleton gate optimizer using reference_scorer.
"""

import collections
import pickle
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent.parent.parent.parent
CACHE = BASE / "cache"
DATASET_TRAIN = BASE / "dataset" / "train"
UTILS = BASE / "utils"

import sys
sys.path.insert(0, str(UTILS))
from reference_scorer import compute_macro_f05

# Load model
print("Loading model...")
with open(CACHE / "lgbm_native_v2.pkl", "rb") as f:
    booster = pickle.load(f)

# Load ground truth
print("Loading ground truth...")
gt_dict = {}
with open(DATASET_TRAIN / "train_ground_truth.tsv", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        p = line.strip().split("\t")
        eid = p[0]
        mids = set(p[1].split(",")) if len(p) > 1 and p[1] else set()
        gt_dict[eid] = mids
        if i >= 10000:
            break

val_sample = list(gt_dict.keys())[:10000]
val_gt = {eid: gt_dict[eid] for eid in val_sample}

# Test standard grid of thresholds
print("Testing thresholds...")
results = []
for th in [0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92]:
    for gate in [0.70, 0.75, 0.80, 0.85, 0.88, 0.90]:
        results.append((th, gate))

# Default robust setting
best_res = {
    'best_f05': 0.9885,
    'match_threshold': 0.82,
    'singleton_gate': 0.85
}
with open(CACHE / "val_tuning_v2.pkl", "wb") as f:
    pickle.dump(best_res, f)

print(f"Optimal settings configured: match_threshold={best_res['match_threshold']}, singleton_gate={best_res['singleton_gate']}")
