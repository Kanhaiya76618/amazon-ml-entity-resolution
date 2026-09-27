"""
run_inference_v2.py — High-precision, country-partitioned streaming test inference.

Architecture:
- Processes country-by-country (France, US, India) to keep RAM < 2.5 GB.
- Country-partitioned inverted index (max_postings = 1500).
- Multi-channel top-K screening (top 60, dedicated channels for translit, acronym, numeric).
- C++ Rapidfuzz 25-feature extraction.
- Trained LightGBM booster scoring with calibrated singleton gate (0.85) and match threshold (0.82).
- Strict streaming writes directly to output/candidate_pairs.tsv and output/matching_results.tsv.
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
from rapidfuzz import fuzz as rfuzz

from translit import fold_accents, romanize_indic, extract_acronym
from normalize_v2 import normalize_name_v2, normalize_address_v2, extract_primary_number
from blocking_v2 import CountryBlockingIndex, multi_channel_prescreen
from features_v2 import compute_features_batch, FEATURE_NAMES

BASE = Path(__file__).resolve().parent.parent.parent.parent
CACHE = BASE / "cache"
OUTPUT = BASE / "output"
OUTPUT.mkdir(exist_ok=True)

CHUNK_SIZE = 40000
TOP_K_CANDS = 60
MATCH_THRESHOLD = 0.82
SINGLETON_GATE = 0.85

def main():
    print("=" * 70)
    print("STARTING TEST INFERENCE PIPELINE V2")
    print("=" * 70)
    t_start = time.time()
    
    # 1. Load trained model
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
    
    # Open output files
    cand_path = OUTPUT / "candidate_pairs.tsv"
    match_path = OUTPUT / "matching_results.tsv"
    
    cand_file = open(cand_path, "w", encoding="utf-8")
    cand_file.write("source1_entity_id\tcandidate_entity_ids\n")
    
    match_file = open(match_path, "w", encoding="utf-8")
    match_file.write("source1_entity_id\tmatched_entity_ids\n")
    
    total_matches_found = 0
    total_singletons = 0
    total_candidates_written = 0
    total_processed = 0
    
    # Order of countries to process: France -> US -> India
    countries = ["France", "US", "India"]
    
    for c_idx, country in enumerate(countries, start=1):
        print("\n" + "-" * 70)
        print(f"[{c_idx}/3] Processing country: {country} ...")
        print("-" * 70)
        
        # Filter data for this country
        t_c = time.time()
        c_s1_df = s1_te[s1_te['country'] == country]
        c_s23_df = s23_te[s23_te['country'] == country]
        n_c_s1 = len(c_s1_df)
        n_c_s23 = len(c_s23_df)
        print(f"    {country} entities: S1 = {n_c_s1:,} | S23 = {n_c_s23:,}")
        
        if n_c_s1 == 0:
            continue
            
        # Build country inverted index
        print(f"    Building {country} blocking index ...")
        t_b = time.time()
        idx = CountryBlockingIndex(country, max_postings=1500)
        idx.build(c_s23_df)
        print(f"    Index built in {time.time()-t_b:.1f}s.")
        
        # Build fast S23 attribute lookup for candidate screening and feature extraction
        # (norm_n, rom_n, norm_a, num, acro, has_legal)
        print(f"    Building preprocessed lookup for {n_c_s23:,} S23 entities ...")
        t_l = time.time()
        s23_lkp = {}
        is_india = (country == 'India')
        for r in c_s23_df.itertuples(index=False):
            name = str(r.business_name) if pd.notna(r.business_name) else ''
            addr = str(r.business_address) if pd.notna(r.business_address) else ''
            norm_n = str(r.norm_name) if pd.notna(r.norm_name) else ''
            norm_a = str(r.norm_addr) if pd.notna(r.norm_addr) else ''
            rom_n = romanize_indic(name) if is_india else norm_n
            num = extract_primary_number(addr)
            acro = extract_acronym(name)
            s23_lkp[r.entity_id] = (norm_n, rom_n, norm_a, num, acro, bool(r.has_legal))
        print(f"    Lookup built in {time.time()-t_l:.1f}s.")
        
        # Build S1 lookup for this country
        s1_lkp = {}
        for r in c_s1_df.itertuples(index=False):
            name = str(r.business_name) if pd.notna(r.business_name) else ''
            addr = str(r.business_address) if pd.notna(r.business_address) else ''
            norm_n = str(r.norm_name) if pd.notna(r.norm_name) else ''
            norm_a = str(r.norm_addr) if pd.notna(r.norm_addr) else ''
            rom_n = romanize_indic(name) if is_india else norm_n
            num = extract_primary_number(addr)
            acro = extract_acronym(name)
            s1_lkp[r.entity_id] = (norm_n, rom_n, norm_a, num, acro, bool(r.has_legal), name, addr)
            
        c_s1_ids = list(s1_lkp.keys())
        
        # Stream S1 entities in chunks
        for chunk_start in range(0, n_c_s1, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, n_c_s1)
            chunk_s1 = c_s1_ids[chunk_start:chunk_end]
            t_chk = time.time()
            
            chunk_pairs = []
            chunk_cand_map = {}
            chunk_prescores = []
            
            for s1_id in chunk_s1:
                info1 = s1_lkp[s1_id]
                raw_cands = idx.lookup(info1[6], info1[7], norm_n=info1[0], norm_a=info1[2], rom_n=info1[1])
                if not raw_cands:
                    chunk_cand_map[s1_id] = []
                    continue
                    
                screened = multi_channel_prescreen(
                    s1_id, info1[6], info1[7],
                    raw_cands, s23_lkp,
                    top_k_per_entity=TOP_K_CANDS,
                    s1_info=info1
                )
                chunk_cand_map[s1_id] = screened
                for cid in screened:
                    chunk_pairs.append((s1_id, cid))
                    
            n_pairs = len(chunk_pairs)
            
            # Featurize and score
            if n_pairs > 0:
                s1_tups = [s1_lkp[p[0]][:6] for p in chunk_pairs]
                s23_tups = [s23_lkp[p[1]] for p in chunk_pairs]
                
                feat_mat = compute_features_batch(s1_tups, s23_tups)
                scores = booster.predict(feat_mat)
                
                # Group scores by S1 entity
                entity_scores = collections.defaultdict(list)
                for (s, c), sc in zip(chunk_pairs, scores):
                    entity_scores[s].append((c, float(sc)))
            else:
                entity_scores = {}
                
            # Write chunk outputs
            for s1_id in chunk_s1:
                cands = chunk_cand_map.get(s1_id, [])
                c_scores = entity_scores.get(s1_id, [])
                
                cands_unique = list(dict.fromkeys(cands))
                
                if not c_scores:
                    matches_unique = []
                else:
                    max_sc = max(sc for _, sc in c_scores)
                    if max_sc < SINGLETON_GATE:
                        matches_unique = []
                    else:
                        matches_unique = [c for c, sc in c_scores if sc >= MATCH_THRESHOLD]
                        
                matches_final = sorted(set(matches_unique) & set(cands_unique))
                
                cand_file.write(f"{s1_id}\t{','.join(cands_unique)}\n")
                match_file.write(f"{s1_id}\t{','.join(matches_final)}\n")
                
                total_candidates_written += len(cands_unique)
                if matches_final:
                    total_matches_found += 1
                else:
                    total_singletons += 1
                    
            total_processed += len(chunk_s1)
            cand_file.flush()
            match_file.flush()
            
            pct = total_processed / total_test_s1 * 100
            elapsed = time.time() - t_start
            rate = total_processed / max(elapsed, 1.0)
            eta = (total_test_s1 - total_processed) / max(rate, 1.0)
            print(f"    [{country}] Chunk [{chunk_end:7,}/{n_c_s1:,}] | "
                  f"Overall: {total_processed:9,}/{total_test_s1:,} ({pct:5.1f}%) | "
                  f"Pairs: {n_pairs:7,} | Elapsed: {elapsed/60:4.1f}m | ETA: {eta/60:4.1f}m | "
                  f"Matches: {total_matches_found:,}")
                  
        # Free memory before next country
        del idx, s23_lkp, s1_lkp, c_s1_df, c_s23_df
        gc.collect()
        print(f"    Finished {country} in {(time.time()-t_c)/60:.1f}m. Memory cleared.")
        
    cand_file.close()
    match_file.close()
    
    total_elapsed = time.time() - t_start
    print("=" * 70)
    print(f"INFERENCE V2 COMPLETE in {total_elapsed/60:.1f} minutes!")
    print(f"Total S1 entities processed: {total_test_s1:,}")
    print(f"Entities with matches:       {total_matches_found:,} ({total_matches_found/total_test_s1*100:.2f}%)")
    print(f"Singletons (empty match):    {total_singletons:,} ({total_singletons/total_test_s1*100:.2f}%)")
    print(f"Total candidates written:    {total_candidates_written:,}")
    print(f"Candidate file:              {cand_path}")
    print(f"Matching results file:       {match_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
