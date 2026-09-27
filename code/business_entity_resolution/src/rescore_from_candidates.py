#!/usr/bin/env python3
"""
rescore_from_candidates.py — Streaming Rescorer for Final Leaderboard Submission.

Why this script exists:
1. output/candidate_pairs.tsv is already built with 82,035,054 high-recall candidates
   across all 1,732,544 test S1 entities using multi-channel blocking (Indic translit,
   acronyms, numeric tokens, France St->saint, etc.).
2. The previous 0.450 drop was caused by SINGLETON_GATE = 0.85 wiping out 450,835 entities (26.02%)
   when true singletons are only 5.58%.
3. This script streams the existing candidate pairs directly, extracts 25 C++ Rapidfuzz features,
   predicts with native LightGBM booster, and applies calibrated thresholding:
   - Base match threshold: 0.88
   - Safe top-1 fallback: if no candidate >= 0.88, but best candidate >= 0.70, emit top candidate.
   - True singleton: if best candidate < 0.70, emit empty [] (matches 5.0%-5.5% singleton rate).
4. Memory footprint: Keeps RAM < 1.8 GB by loading S23 country-by-country (France -> US -> India).
"""

from __future__ import annotations
import collections
import gc
import os
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd

from translit import romanize_indic, extract_acronym
from normalize_v2 import extract_primary_number
from features_v2 import compute_features_batch

BASE = Path(__file__).resolve().parent.parent.parent.parent
CACHE = BASE / "cache"
OUTPUT = BASE / "output"

MATCH_THRESHOLD = 0.88
FALLBACK_THRESHOLD = 0.70
CHUNK_SIZE = 25000

def main():
    print("=" * 70)
    print("STREAMING RESCORER FROM CANDIDATE PAIRS (OPTIMAL F0.5)")
    print("=" * 70)
    t_start = time.time()
    
    # 1. Load model
    print("[1] Loading LightGBM model ...")
    with open(CACHE / "lgbm_native_v2.pkl", "rb") as f:
        booster = pickle.load(f)
    print("    Model loaded.")
    
    # 2. Load normalized test tables
    print("[2] Loading normalized test tables from cache ...")
    t0 = time.time()
    s1_te = pickle.load(open(CACHE / "norm_s1_te.pkl", "rb"))
    s23_te = pickle.load(open(CACHE / "norm_s23_te.pkl", "rb"))
    print(f"    Normalized tables loaded in {time.time()-t0:.1f}s. S1={len(s1_te):,}, S23={len(s23_te):,}")
    
    total_test_s1 = len(s1_te)
    
    cand_path = OUTPUT / "candidate_pairs.tsv"
    match_path = OUTPUT / "matching_results.tsv"
    
    if not cand_path.exists():
        raise FileNotFoundError(f"Missing {cand_path}!")
        
    match_file = open(match_path, "w", encoding="utf-8")
    match_file.write("source1_entity_id\tmatched_entity_ids\n")
    
    cand_file = open(cand_path, "r", encoding="utf-8")
    header = next(cand_file)
    assert header.strip().split("\t") == ["source1_entity_id", "candidate_entity_ids"]
    
    total_matches_found = 0
    total_singletons = 0
    total_fallback_emitted = 0
    total_multi_emitted = 0
    total_processed = 0
    
    # Countries in order in candidate_pairs.tsv: France, US, India
    country_specs = [
        ("France", 259452),
        ("US", 663106),
        ("India", 809986)
    ]
    
    for c_idx, (country, expected_count) in enumerate(country_specs, start=1):
        print("\n" + "-" * 70)
        print(f"[{c_idx}/3] Processing country: {country} ({expected_count:,} entities) ...")
        print("-" * 70)
        t_c = time.time()
        
        c_s1_df = s1_te[s1_te['country'] == country]
        c_s23_df = s23_te[s23_te['country'] == country]
        is_india = (country == 'India')
        
        # Build fast S23 attribute lookup for this country
        print(f"    Building S23 lookup for {len(c_s23_df):,} {country} entities ...")
        t_l = time.time()
        s23_lkp = {}
        for r in c_s23_df.itertuples(index=False):
            name = str(r.business_name) if pd.notna(r.business_name) else ''
            addr = str(r.business_address) if pd.notna(r.business_address) else ''
            norm_n = str(r.norm_name) if pd.notna(r.norm_name) else ''
            norm_a = str(r.norm_addr) if pd.notna(r.norm_addr) else ''
            rom_n = romanize_indic(name) if is_india else norm_n
            num = extract_primary_number(addr)
            acro = extract_acronym(name)
            s23_lkp[r.entity_id] = (norm_n, rom_n, norm_a, num, acro, bool(r.has_legal))
        print(f"    S23 lookup built in {time.time()-t_l:.1f}s.")
        
        # Build fast S1 attribute lookup for this country
        print(f"    Building S1 lookup for {len(c_s1_df):,} {country} entities ...")
        s1_lkp = {}
        for r in c_s1_df.itertuples(index=False):
            name = str(r.business_name) if pd.notna(r.business_name) else ''
            addr = str(r.business_address) if pd.notna(r.business_address) else ''
            norm_n = str(r.norm_name) if pd.notna(r.norm_name) else ''
            norm_a = str(r.norm_addr) if pd.notna(r.norm_addr) else ''
            rom_n = romanize_indic(name) if is_india else norm_n
            num = extract_primary_number(addr)
            acro = extract_acronym(name)
            s1_lkp[r.entity_id] = (norm_n, rom_n, norm_a, num, acro, bool(r.has_legal))
            
        # Stream lines from candidate_pairs.tsv in chunks
        chunk_lines = []
        c_processed = 0
        
        while c_processed < expected_count:
            to_read = min(CHUNK_SIZE, expected_count - c_processed)
            chunk_s1_ids = []
            chunk_cand_map = {}
            chunk_pairs = []
            
            for _ in range(to_read):
                line = cand_file.readline()
                if not line:
                    break
                p = line.rstrip("\r\n").split("\t")
                s1_id = p[0]
                cands = p[1].split(",") if len(p) > 1 and p[1] else []
                chunk_s1_ids.append(s1_id)
                chunk_cand_map[s1_id] = cands
                for cid in cands:
                    if cid in s23_lkp:
                        chunk_pairs.append((s1_id, cid))
                        
            n_chunk_pairs = len(chunk_pairs)
            
            # Featurize and predict
            entity_scores = collections.defaultdict(list)
            if n_chunk_pairs > 0:
                s1_tups = [s1_lkp[p[0]] for p in chunk_pairs]
                s23_tups = [s23_lkp[p[1]] for p in chunk_pairs]
                feat_mat = compute_features_batch(s1_tups, s23_tups)
                scores = booster.predict(feat_mat)
                for (s, c), sc in zip(chunk_pairs, scores):
                    entity_scores[s].append((c, float(sc)))
                    
            # Apply calibrated threshold + fallback rule
            for s1_id in chunk_s1_ids:
                cands = chunk_cand_map.get(s1_id, [])
                c_scores = entity_scores.get(s1_id, [])
                
                if not c_scores:
                    matched_final = []
                else:
                    # Candidates meeting MATCH_THRESHOLD
                    high_cands = [c for c, sc in c_scores if sc >= MATCH_THRESHOLD]
                    if high_cands:
                        matched_final = sorted(set(high_cands))
                        total_multi_emitted += 1
                    else:
                        # Fallback check
                        best_c, best_sc = max(c_scores, key=lambda x: x[1])
                        if best_sc >= FALLBACK_THRESHOLD:
                            matched_final = [best_c]
                            total_fallback_emitted += 1
                        else:
                            matched_final = []
                            
                # Verify subset guarantee
                matched_final = [c for c in matched_final if c in cands]
                
                match_file.write(f"{s1_id}\t{','.join(matched_final)}\n")
                if matched_final:
                    total_matches_found += 1
                else:
                    total_singletons += 1
                    
            c_processed += len(chunk_s1_ids)
            total_processed += len(chunk_s1_ids)
            match_file.flush()
            
            pct = total_processed / total_test_s1 * 100
            elapsed = time.time() - t_start
            rate = total_processed / max(elapsed, 1.0)
            eta = (total_test_s1 - total_processed) / max(rate, 1.0)
            print(f"    [{country}] Chunk [{c_processed:7,}/{expected_count:,}] | "
                  f"Overall: {total_processed:9,}/{total_test_s1:,} ({pct:5.1f}%) | "
                  f"Pairs: {n_chunk_pairs:7,} | Elapsed: {elapsed/60:4.1f}m | ETA: {eta/60:4.1f}m | "
                  f"Matches: {total_matches_found:,} | Singletons: {total_singletons:,}")
                  
        # Free memory before next country
        del s23_lkp, s1_lkp, c_s1_df, c_s23_df
        gc.collect()
        print(f"    Finished {country} in {(time.time()-t_c)/60:.1f}m. Memory cleared.")
        
    cand_file.close()
    match_file.close()
    
    total_elapsed = time.time() - t_start
    print("=" * 70)
    print(f"RESCORING COMPLETE in {total_elapsed/60:.1f} minutes!")
    print(f"Total S1 entities processed: {total_test_s1:,}")
    print(f"Entities with matches:       {total_matches_found:,} ({total_matches_found/total_test_s1*100:.2f}%)")
    print(f"Singletons (empty match):    {total_singletons:,} ({total_singletons/total_test_s1*100:.2f}%)")
    print(f"Multi/High-thresh matches:   {total_multi_emitted:,}")
    print(f"Fallback single matches:     {total_fallback_emitted:,}")
    print(f"Matching results file:       {match_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
