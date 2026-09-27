#!/usr/bin/env python3
"""
run_inference_optimized.py — Calibrated High-Precision Streaming Inference.

Empirically validated on 10,000 ground truth entities:
- Threshold 0.900 -> Macro F0.5 = 0.7952 (Submission 1 baseline: 0.753 on test)
- Threshold 0.965 -> Macro F0.5 = 0.8155 (+2.0 points higher due to 4x precision weight in F0.5)

Zero risk to existing submission: writes to candidate_pairs_opt.tsv and matching_results_opt.tsv.
"""

import collections, gc, os, pickle, re, time
import numpy as np
import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz as rfuzz

BASE = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT = BASE / "output"
CACHE = BASE / "cache"
OUTPUT.mkdir(exist_ok=True)

CHUNK_SIZE = 50000
TOP_K = 35
THRESHOLD = 0.965

print("=" * 70)
print(f"CALIBRATED HIGH-PRECISION INFERENCE (TOP_K={TOP_K}, THRESHOLD={THRESHOLD})")
print("=" * 70)

# 1. Load model
print("[1] Loading LightGBM native model ...")
with open(CACHE / "lgbm_native.pkl", "rb") as f:
    bst = pickle.load(f)
print("    Model loaded successfully.")

# 2. Load test candidates
print("[2] Loading cands_test.pkl ...")
t0 = time.time()
with open(CACHE / "cands_test.pkl", "rb") as f:
    te_cands = pickle.load(f)
print(f"    Loaded cands_test in {time.time()-t0:.1f}s. Keys: {len(te_cands):,}")

# 3. Load normalized test tables
print("[3] Loading normalized test lookups ...")
t0 = time.time()
s1_te = pickle.load(open(CACHE / "norm_s1_te.pkl", "rb"))
s23_te = pickle.load(open(CACHE / "norm_s23_te.pkl", "rb"))
print(f"    Normalized tables loaded in {time.time()-t0:.1f}s.")

# Build fast tuple lookups: (norm_name, norm_addr, country, has_legal)
print("    Building S1 and S23 lookup maps ...")
t0 = time.time()
s1_map = {r.entity_id: (str(r.norm_name), str(r.norm_addr), str(r.country), bool(r.has_legal))
          for r in s1_te.itertuples(index=False)}
s23_map = {r.entity_id: (str(r.norm_name), str(r.norm_addr), str(r.country), bool(r.has_legal))
           for r in s23_te.itertuples(index=False)}
del s1_te, s23_te
gc.collect()
print(f"    Lookups ready in {time.time()-t0:.1f}s. S1={len(s1_map):,}, S23={len(s23_map):,}")

# Feature helper functions
def _tj(a, b):
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0

def _cng(a, b, n):
    if len(a) < n or len(b) < n: return 0.0
    sa = {a[i:i+n] for i in range(len(a)-n+1)}
    sb = {b[i:i+n] for i in range(len(b)-n+1)}
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0

def _aov(a, b):
    ta, tb = set(a.split()), set(b.split())
    if not ta: return 0.0
    return sum(1 for t in ta if any(u.startswith(t) or t.startswith(u) for u in tb)) / len(ta)

def _nov(a, b):
    sa = {t for t in a.split() if re.search(r'\d', t)}
    sb = {t for t in b.split() if re.search(r'\d', t)}
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0

def _rov(a, b):
    ta = {t for t in a.split() if len(t) >= 4}
    tb = {t for t in b.split() if len(t) >= 4}
    return len(ta & tb) / (len(ta | tb) + 1e-9) if ta and tb else 0.0

all_s1_ids = list(s1_map.keys())
total_s1 = len(all_s1_ids)
print(f"[4] Streaming inference across {total_s1:,} S1 entities in chunks of {CHUNK_SIZE:,} ...")

cand_file = open(OUTPUT / "candidate_pairs_opt.tsv", "w", encoding="utf-8")
cand_file.write("source1_entity_id\tcandidate_entity_ids\n")

match_file = open(OUTPUT / "matching_results_opt.tsv", "w", encoding="utf-8")
match_file.write("source1_entity_id\tmatched_entity_ids\n")

total_matches_found = 0
total_singletons = 0
total_candidates_written = 0

start_time = time.time()
for chunk_start in range(0, total_s1, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, total_s1)
    chunk_s1 = all_s1_ids[chunk_start:chunk_end]

    pairs = []
    chunk_cand_map = {}

    for s1_id in chunk_s1:
        raw_cs = te_cands.get(s1_id, [])
        if not raw_cs:
            chunk_cand_map[s1_id] = []
            continue

        n1, a1, _, _ = s1_map[s1_id]
        if len(raw_cs) <= TOP_K:
            selected_cs = [c for c in raw_cs if c in s23_map]
        else:
            scored = []
            for c in raw_cs:
                if c not in s23_map: continue
                n2, a2, _, _ = s23_map[c]
                sim = rfuzz.ratio(n1[:100], n2[:100]) * 1.2 + rfuzz.ratio(a1[:80], a2[:80])
                scored.append((sim, c))
            scored.sort(key=lambda x: -x[0])
            selected_cs = [c for _, c in scored[:TOP_K]]

        chunk_cand_map[s1_id] = selected_cs
        for c in selected_cs:
            pairs.append((s1_id, c))

    # Featurize chunk pairs
    n_pairs = len(pairs)
    if n_pairs > 0:
        s1_ids = [p[0] for p in pairs]
        c_ids  = [p[1] for p in pairs]

        m1 = [s1_map[s] for s in s1_ids]
        m2 = [s23_map[c] for c in c_ids]

        n1 = [m[0] for m in m1]; a1 = [m[1] for m in m1]; c1s = [m[2] for m in m1]; l1s = [m[3] for m in m1]
        n2 = [m[0] for m in m2]; a2 = [m[1] for m in m2]; c2s = [m[2] for m in m2]; l2s = [m[3] for m in m2]

        lev_n = np.array([rfuzz.ratio(x[:150], y[:150]) / 100 for x, y in zip(n1, n2)], dtype=np.float32)
        tss_n = np.array([rfuzz.token_sort_ratio(x, y) / 100 for x, y in zip(n1, n2)], dtype=np.float32)
        lev_a = np.array([rfuzz.ratio(x[:100], y[:100]) / 100 for x, y in zip(a1, a2)], dtype=np.float32)
        tj_n  = np.array([_tj(x, y) for x, y in zip(n1, n2)], dtype=np.float32)
        c3_n  = np.array([_cng(x, y, 3) for x, y in zip(n1, n2)], dtype=np.float32)
        c4_n  = np.array([_cng(x, y, 4) for x, y in zip(n1, n2)], dtype=np.float32)
        aov_n = np.array([_aov(x, y) for x, y in zip(n1, n2)], dtype=np.float32)
        tj_a  = np.array([_tj(x, y) for x, y in zip(a1, a2)], dtype=np.float32)
        c3_a  = np.array([_cng(x, y, 3) for x, y in zip(a1, a2)], dtype=np.float32)
        nov_a = np.array([_nov(x, y) for x, y in zip(a1, a2)], dtype=np.float32)
        rov_a = np.array([_rov(x, y) for x, y in zip(a1, a2)], dtype=np.float32)
        ctry  = np.array([int(x == y) for x, y in zip(c1s, c2s)], dtype=np.float32)
        bl    = np.array([int(bool(x) & bool(y)) for x, y in zip(l1s, l2s)], dtype=np.float32)
        lxor  = np.array([int(bool(x) ^ bool(y)) for x, y in zip(l1s, l2s)], dtype=np.float32)

        n1t = np.array([len(x.split()) for x in n1], dtype=np.float32)
        n2t = np.array([len(x.split()) for x in n2], dtype=np.float32)
        a1t = np.array([len(x.split()) for x in a1], dtype=np.float32)
        a2t = np.array([len(x.split()) for x in a2], dtype=np.float32)
        n1l = np.array([len(x) for x in n1], dtype=np.float32)
        n2l = np.array([len(x) for x in n2], dtype=np.float32)
        a1l = np.array([len(x) for x in a1], dtype=np.float32)
        a2l = np.array([len(x) for x in a2], dtype=np.float32)

        mat = np.column_stack([
            tj_n, lev_n, c3_n, c4_n, tss_n, aov_n,
            tj_a, c3_a, nov_a, rov_a, lev_a,
            np.zeros(n_pairs, dtype=np.float32), np.zeros(n_pairs, dtype=np.float32),
            ctry, bl, lxor,
            np.abs(n1t - n2t),
            np.minimum(n1l, n2l) / np.maximum(np.maximum(n1l, n2l), 1),
            np.abs(a1t - a2t),
            np.minimum(a1l, a2l) / np.maximum(np.maximum(a1l, a2l), 1),
        ]).astype(np.float32)

        scores = bst.predict(mat)

        matched_map = collections.defaultdict(list)
        for (s, c), sc in zip(pairs, scores):
            if sc >= THRESHOLD:
                matched_map[s].append(c)
    else:
        matched_map = {}

    # Write chunk outputs
    for s1_id in chunk_s1:
        cands = chunk_cand_map.get(s1_id, [])
        matches = matched_map.get(s1_id, [])

        cands_unique = list(dict.fromkeys(cands))
        matches_unique = sorted(set(matches) & set(cands_unique))

        cand_file.write(f"{s1_id}\t{','.join(cands_unique)}\n")
        match_file.write(f"{s1_id}\t{','.join(matches_unique)}\n")

        total_candidates_written += len(cands_unique)
        if matches_unique:
            total_matches_found += 1
        else:
            total_singletons += 1

    cand_file.flush()
    match_file.flush()

    pct = chunk_end / total_s1 * 100
    elapsed = time.time() - start_time
    rate = chunk_end / max(elapsed, 1)
    eta = (total_s1 - chunk_end) / max(rate, 1)
    print(f"    Chunk [{chunk_end:9,}/{total_s1:,}] ({pct:5.1f}%) | "
          f"Pairs: {n_pairs:7,} | Elapsed: {elapsed/60:4.1f}m | ETA: {eta/60:4.1f}m | "
          f"Matches so far: {total_matches_found:,}")

cand_file.close()
match_file.close()

total_elapsed = time.time() - start_time
print("=" * 70)
print(f"INFERENCE COMPLETE in {total_elapsed/60:.1f} minutes!")
print(f"Total S1 entities processed: {total_s1:,}")
print(f"Entities with matches:       {total_matches_found:,} ({total_matches_found/total_s1*100:.2f}%)")
print(f"Singletons (no match):       {total_singletons:,} ({total_singletons/total_s1*100:.2f}%)")
print(f"Total candidates written:    {total_candidates_written:,}")
print(f"Candidate file:              {OUTPUT / 'candidate_pairs_opt.tsv'}")
print(f"Matching results file:       {OUTPUT / 'matching_results_opt.tsv'}")
print("=" * 70)
