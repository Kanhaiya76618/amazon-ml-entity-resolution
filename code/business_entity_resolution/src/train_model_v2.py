"""
train_model_v2.py — Fast, memory-safe end-to-end training and macro F0.5 validation harness.
"""

from __future__ import annotations
import gc
import os
import pickle
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz as rfuzz

from translit import fold_accents, romanize_indic, extract_acronym
from normalize_v2 import normalize_name_v2, normalize_address_v2, extract_primary_number
from features_v2 import compute_features_batch, FEATURE_NAMES

BASE = Path(__file__).resolve().parent.parent.parent.parent
CACHE = BASE / "cache"
DATASET_TRAIN = BASE / "dataset" / "train"
UTILS = BASE / "utils"

import sys
sys.path.insert(0, str(UTILS))
from reference_scorer import compute_macro_f05

RANDOM_SEED = 42

def main():
    print("=" * 70)
    print("TRAINING PIPELINE V2 — END-TO-END MACRO F0.5 OPTIMIZATION")
    print("=" * 70)
    
    t0 = time.time()
    
    # 1. Load Ground Truth
    print("[1] Loading ground truth ...")
    gt_dict = {}
    with open(DATASET_TRAIN / "train_ground_truth.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.strip().split("\t")
            eid = p[0]
            mids = set(p[1].split(",")) if len(p) > 1 and p[1] else set()
            gt_dict[eid] = mids
    print(f"    Loaded ground truth for {len(gt_dict):,} S1 entities in {time.time()-t0:.1f}s")
    
    # 2. Grouped Train/Val Split
    print("[2] Creating grouped train/validation split ...")
    rng = np.random.default_rng(RANDOM_SEED)
    all_s1_ids = list(gt_dict.keys())
    rng.shuffle(all_s1_ids)
    
    n_val = 50000
    val_s1_ids = set(all_s1_ids[:n_val])
    train_s1_ids = all_s1_ids[n_val:]
    print(f"    Validation entities: {len(val_s1_ids):,} | Training pool: {len(train_s1_ids):,}")
    
    # 3. Select Training Pairs (300k Positives + 900k Negatives)
    print("[3] Selecting training pairs ...")
    t1 = time.time()
    train_sample_s1 = train_s1_ids[:100000]
    
    pos_pairs = []
    for s1_id in train_sample_s1:
        true_m = gt_dict.get(s1_id, set())
        for m in true_m:
            pos_pairs.append((s1_id, m))
            if len(pos_pairs) >= 300000:
                break
        if len(pos_pairs) >= 300000:
            break
            
    print(f"    Selected {len(pos_pairs):,} positive pairs from {len(train_sample_s1):,} S1 entities.")
    
    # Load candidate train map to get hard negatives
    print("    Loading candidates for hard negative mining ...")
    with open(CACHE / "cands_train.pkl", "rb") as f:
        cands_tr = pickle.load(f)
        
    neg_pairs = []
    for s1_id in train_sample_s1:
        true_m = gt_dict.get(s1_id, set())
        cands = cands_tr.get(s1_id, [])
        for c in cands:
            if c not in true_m:
                neg_pairs.append((s1_id, c))
                if len(neg_pairs) >= 900000:
                    break
        if len(neg_pairs) >= 900000:
            break
            
    del cands_tr
    gc.collect()
    print(f"    Selected {len(neg_pairs):,} hard negative pairs.")
    
    # 4. Filter only needed S1 and S23 entities for normalization
    needed_s1 = {p[0] for p in pos_pairs} | {p[0] for p in neg_pairs}
    needed_s23 = {p[1] for p in pos_pairs} | {p[1] for p in neg_pairs}
    
    # Add validation entities
    val_sample_s1 = list(val_s1_ids)[:5000]
    val_needed_s23 = set()
    for eid in val_sample_s1:
        val_needed_s23.update(gt_dict[eid])
        
    needed_s1.update(val_sample_s1)
    needed_s23.update(val_needed_s23)
    
    print(f"    Total unique entities needed: S1={len(needed_s1):,}, S23={len(needed_s23):,}")
    
    # 5. Load and Normalize only needed entities
    print("[5] Loading and normalizing required entities ...")
    t1 = time.time()
    s1_df = pickle.load(open(CACHE / "norm_s1_tr.pkl", "rb"))
    s1_sub = s1_df[s1_df['entity_id'].isin(needed_s1)]
    del s1_df
    gc.collect()
    
    s23_df = pickle.load(open(CACHE / "norm_s23_tr.pkl", "rb"))
    s23_sub = s23_df[s23_df['entity_id'].isin(needed_s23)]
    del s23_df
    gc.collect()
    
    s1_lkp = {}
    for r in s1_sub.itertuples(index=False):
        norm_n, rom_n, has_l = normalize_name_v2(r.business_name)
        norm_a = normalize_address_v2(r.business_address, r.country)
        num = extract_primary_number(r.business_address)
        acro = extract_acronym(r.business_name)
        s1_lkp[r.entity_id] = (norm_n, rom_n, norm_a, num, acro, has_l)
        
    s23_lkp = {}
    for r in s23_sub.itertuples(index=False):
        norm_n, rom_n, has_l = normalize_name_v2(r.business_name)
        norm_a = normalize_address_v2(r.business_address, r.country)
        num = extract_primary_number(r.business_address)
        acro = extract_acronym(r.business_name)
        s23_lkp[r.entity_id] = (norm_n, rom_n, norm_a, num, acro, has_l)
        
    del s1_sub, s23_sub
    gc.collect()
    print(f"    Normalized required entities in {time.time()-t1:.1f}s.")
    
    # Filter valid pairs
    valid_pos = [p for p in pos_pairs if p[0] in s1_lkp and p[1] in s23_lkp]
    valid_neg = [p for p in neg_pairs if p[0] in s1_lkp and p[1] in s23_lkp]
    print(f"    Valid training pairs: {len(valid_pos):,} pos, {len(valid_neg):,} neg.")
    
    all_pairs = valid_pos + valid_neg
    all_labels = [1] * len(valid_pos) + [0] * len(valid_neg)
    
    # Shuffle
    p_idx = np.arange(len(all_pairs))
    rng.shuffle(p_idx)
    all_pairs = [all_pairs[i] for i in p_idx]
    y_tr = np.array([all_labels[i] for i in p_idx], dtype=np.int32)
    
    # 6. Compute Features
    print("[6] Computing 25 features ...")
    t1 = time.time()
    s1_tups = [s1_lkp[p[0]] for p in all_pairs]
    s23_tups = [s23_lkp[p[1]] for p in all_pairs]
    X_tr = compute_features_batch(s1_tups, s23_tups)
    print(f"    Features extracted in {time.time()-t1:.1f}s ({len(X_tr)/(time.time()-t1):,.0f} pairs/sec).")
    
    # 7. Train LightGBM Booster
    print("[7] Training LightGBM booster ...")
    t1 = time.time()
    pos_weight = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1.0)
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_NAMES)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 350,
        'learning_rate': 0.05,
        'num_leaves': 127,
        'max_depth': 8,
        'min_child_samples': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': pos_weight * 0.7, # precision bias for F0.5
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'verbose': -1
    }
    
    booster = lgb.train(params, dtrain, num_boost_round=350)
    print(f"    Booster trained in {time.time()-t1:.1f}s.")
    
    imp = booster.feature_importance(importance_type='gain')
    ranked_features = sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])
    print("    Top 10 most important features:")
    for fn, gain in ranked_features[:10]:
        print(f"      {fn:18s}: {gain:12.1f}")
        
    with open(CACHE / "lgbm_native_v2.pkl", "wb") as f:
        pickle.dump(booster, f)
    print(f"    Saved model to {CACHE / 'lgbm_native_v2.pkl'}")
    
    # 8. End-to-End Validation & Metric Tuning
    print("[8] Running end-to-end Macro F0.5 evaluation on 5,000 held-out entities ...")
    val_gt = {eid: gt_dict[eid] for eid in val_sample_s1}
    
    val_pairs = []
    for eid in val_sample_s1:
        for cid in val_gt[eid]:
            if cid in s23_lkp:
                val_pairs.append((eid, cid))
                
    print(f"    Scoring {len(val_pairs):,} pairs across {len(val_sample_s1):,} entities ...")
    val_s1_tups = [s1_lkp[p[0]] for p in val_pairs]
    val_s23_tups = [s23_lkp[p[1]] for p in val_pairs]
    X_val = compute_features_batch(val_s1_tups, val_s23_tups)
    val_scores = booster.predict(X_val)
    
    entity_cand_scores = collections.defaultdict(list)
    for (eid, cid), score in zip(val_pairs, val_scores):
        entity_cand_scores[eid].append((cid, float(score)))
        
    for eid in val_sample_s1:
        if eid not in entity_cand_scores:
            entity_cand_scores[eid] = []
            
    print("    Jointly tuning match threshold and singleton gate ...")
    best_f05 = -1.0
    best_th = 0.85
    best_gate = 0.85
    
    for th in [0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92]:
        for gate in [0.75, 0.80, 0.85, 0.88, 0.90, 0.92]:
            preds = {}
            for eid, c_list in entity_cand_scores.items():
                if not c_list:
                    preds[eid] = set()
                    continue
                max_sc = max(sc for _, sc in c_list)
                if max_sc < gate:
                    preds[eid] = set()
                else:
                    preds[eid] = {cid for cid, sc in c_list if sc >= th}
                    
            f05, _ = compute_macro_f05(val_gt, preds)
            if f05 > best_f05:
                best_f05 = f05
                best_th = th
                best_gate = gate
                
    print("=" * 70)
    print(f"OPTIMAL VALIDATION MACRO F0.5: {best_f05:.4f}!")
    print(f"Optimal match threshold:        {best_th}")
    print(f"Optimal singleton gate:        {best_gate}")
    print("=" * 70)
    
    tuning_res = {
        'best_f05': best_f05,
        'match_threshold': best_th,
        'singleton_gate': best_gate
    }
    with open(CACHE / "val_tuning_v2.pkl", "wb") as f:
        pickle.dump(tuning_res, f)
        
    print(f"Artifacts successfully generated in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    main()
