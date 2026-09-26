#!/usr/bin/env python3
"""
pipeline.py — End-to-end entity resolution pipeline.

Optimisations for scale (2.2M S1, 10M S2/S3):
  - Vectorised pandas normalisation (1-2 min for 10M rows)
  - Blocking first → embed only S23 entities in candidate sets
  - rapidfuzz for all string similarity (900K-3M pairs/sec)
  - MPS (Apple Silicon) acceleration for embeddings
  - Full caching at every stage; each stage is skipped if cache exists

Run from student_resource/ directory:
  python3 code/business_entity_resolution/src/pipeline.py
"""

from __future__ import annotations
import collections, os, pickle, random, re, sys, time, unicodedata
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from rapidfuzz import fuzz as rfuzz

BASE      = Path(__file__).resolve().parent.parent.parent.parent
TRAIN_DIR = BASE / "dataset" / "train"
TEST_DIR  = BASE / "dataset" / "test"
OUTPUT    = BASE / "output"
CACHE     = BASE / "cache"
OUTPUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Vectorised normalisation
# ─────────────────────────────────────────────────────────────────────────────
_LEGAL_PAT = (r'(?i)\b(?:corporation|corp|incorporated|inc|limited|ltd|llc|llp|pvt|private|'
              r'company|co|associates|assoc|partnership|sarl|sas|sasu|sa|eurl|sci|sei|'
              r'pty|plc|gmbh|bv|nv|ag|ab|oy|as|kk|ou)\b')
_NAME_SUBS = [
    (r'(?i)\bcorp\b','corporation'),(r'(?i)\bpvt\b','private'),
    (r'(?i)\bltd\b','limited'),(r'(?i)\binc\b','incorporated'),
    (r'(?i)\bco\b','company'),(r'(?i)\bassoc\b','associates'),
    (r'(?i)\bintl\b','international'),(r'(?i)\bmgmt\b','management'),
    (r'(?i)\bsvcs\b','services'),(r'(?i)\bsvc\b','service'),
    (r'(?i)\bgrp\b','group'),(r'(?i)\bnatl\b','national'),
    (r'&',' and '),
]
_ADDR_SUBS = [
    (r'(?i)\brd\b','road'),(r'(?i)\bst\b','street'),(r'(?i)\bstr\b','street'),
    (r'(?i)\bave\b','avenue'),(r'(?i)\bav\b','avenue'),
    (r'(?i)\bblvd\b','boulevard'),(r'(?i)\bdr\b','drive'),
    (r'(?i)\bln\b','lane'),(r'(?i)\bct\b','court'),
    (r'(?i)\bpl\b','place'),(r'(?i)\bpkwy\b','parkway'),
    (r'(?i)\bhwy\b','highway'),(r'(?i)\bapt\b','apartment'),
    (r'(?i)\bste\b','suite'),(r'(?i)\bflr\b','floor'),
    (r'(?i)\bbldg\b','building'),(r'&',' and '),
    (r'(?i)\bopp\b','opposite'),
]

def _vec_norm_name(s: pd.Series):
    s = s.fillna('').astype(str)
    hl = s.str.contains(_LEGAL_PAT, regex=True, na=False)
    s = s.str.lower()
    s = s.str.replace(r'https?://\S+|www\.\S+', ' ', regex=True)
    for p, r in _NAME_SUBS: s = s.str.replace(p, r, regex=True)
    s = s.str.replace(r"[^\w\s\-']", ' ', regex=True)
    s = s.str.replace(r'(?<!\w)-|-(?!\w)', ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s, hl

def _vec_norm_addr(s: pd.Series):
    s = s.fillna('').astype(str).str.lower()
    for p, r in _ADDR_SUBS: s = s.str.replace(p, r, regex=True)
    s = s.str.replace(r'[,;|/\\]', ' ', regex=True)
    s = s.str.replace(r"[^\w\s\-]", ' ', regex=True)
    s = s.str.replace(r'(?<!\w)-|-(?!\w)', ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s

def add_norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    nn, hl = _vec_norm_name(df['business_name'])
    df['norm_name'] = nn; df['has_legal'] = hl
    df['norm_addr'] = _vec_norm_addr(df['business_address'])
    return df

def _cached_norm(df, name):
    cp = CACHE/f"norm_{name}.pkl"
    if cp.exists():
        print(f"    Cache: {name}"); return pickle.load(open(cp,'rb'))
    print(f"    Normalising {name} ({len(df):,} rows) ...", flush=True)
    out = add_norm_cols(df)
    pickle.dump(out, open(cp,'wb'))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Blocking
# ─────────────────────────────────────────────────────────────────────────────
def _toks(t: str) -> list:
    return [x for x in str(t).split() if len(x) >= 2]

def _sorted_pairs(toks: list) -> list:
    st = sorted(set(toks))
    keys = [f"{a}|{b}" for a, b in combinations(st, 2)]
    if st: keys.append('|'.join(st[:6]))
    return keys

def _pfx(toks: list, n=4):
    if not toks: return None
    lg = max(toks, key=len)
    return f"pfx:{lg[:n]}" if len(lg) >= n else None

def _numkey(addr: str):
    for t in str(addr).split():
        if re.search(r'\d', t): return f"num:{t}"
    return None

def _addrkey(addr: str):
    ts = sorted([t for t in str(addr).split() if len(t) >= 4], key=len, reverse=True)[:2]
    if len(ts) >= 2: return 'adr:' + '|'.join(sorted(ts[:2]))
    if ts: return f"adr1:{ts[0]}"
    return None

def _domstrip(t: str) -> str:
    t = re.sub(r'^#+', '', str(t))
    t = re.sub(r'\.(com|org|net|co|in|fr|biz|info|io|us|uk)(\b|$)', '', t, flags=re.I)
    t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
    return t.lower().strip()

def _bkeys(nn: str, na: str, raw='') -> list:
    nt = _toks(nn)
    ks = _sorted_pairs(nt)
    p = _pfx(nt)
    if p: ks.append(p)
    ks += [f"tok:{t}" for t in nt]
    n = _numkey(na)
    if n: ks.append(n)
    a = _addrkey(na)
    if a: ks.append(a)
    if raw:
        st = _toks(_domstrip(str(raw)))
        if st and st != nt:
            ks += _sorted_pairs(st); ks += [f"tok:{t}" for t in st]
    return list(set(ks))

def _build_idx(s23: pd.DataFrame, max_post=400):
    print(f"    Building index over {len(s23):,} ...", flush=True)
    t0 = time.time()
    idx, freq = collections.defaultdict(list), collections.Counter()
    for row in s23.itertuples(index=False):
        raw = row.business_name if pd.notna(row.business_name) else ''
        for k in _bkeys(row.norm_name, row.norm_addr, raw):
            freq[k] += 1; idx[k].append(row.entity_id)
    pruned = {k for k, v in freq.items() if v > max_post}
    for k in pruned: del idx[k]
    print(f"    Index: {len(idx):,} keys, pruned {len(pruned):,}  ({time.time()-t0:.0f}s)")
    return dict(idx)

def _lookup(idx, nn, na, raw='', cap=400):
    cs: set = set()
    for k in _bkeys(nn, na, raw):
        if k in idx: cs.update(idx[k])
    lst = list(cs)
    return lst[:cap] if len(lst) > cap else lst

def run_blocking(s1: pd.DataFrame, s23: pd.DataFrame, cp: Path, max_post=400, cap=400):
    if cp.exists():
        print(f"    Cache: {cp.name}"); return pickle.load(open(cp,'rb'))
    idx = _build_idx(s23, max_post)
    res = {}; t0 = time.time()
    for row in s1.itertuples(index=False):
        raw = row.business_name if pd.notna(row.business_name) else ''
        res[row.entity_id] = _lookup(idx, row.norm_name, row.norm_addr, raw, cap)
    del idx
    tc = sum(len(v) for v in res.values())
    print(f"    Lookup {time.time()-t0:.0f}s: {tc:,} pairs, avg {tc/max(len(res),1):.1f}/entity")
    pickle.dump(res, open(cp,'wb'))
    return res


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature extraction  (rapidfuzz-accelerated)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'name_tj','name_lev','name_c3j','name_c4j','name_tss','name_aov',
    'addr_tj','addr_c3j','addr_nov','addr_rov','addr_lev',
    'name_emb','addr_emb',
    'ctry','both_legal','legal_xor',
    'n_tlen','n_lrat','a_tlen','a_lrat',
]

def _tj(a, b):
    sa, sb = set(str(a).split()), set(str(b).split())
    return len(sa&sb)/len(sa|sb) if sa and sb else 0.0

def _cng(a, b, n):
    a, b = str(a), str(b)
    if len(a)<n or len(b)<n: return 0.0
    sa = {a[i:i+n] for i in range(len(a)-n+1)}
    sb = {b[i:i+n] for i in range(len(b)-n+1)}
    return len(sa&sb)/len(sa|sb) if sa and sb else 0.0

def _aov(a, b):
    ta, tb = set(str(a).split()), set(str(b).split())
    if not ta: return 0.0
    return sum(1 for t in ta if any(u.startswith(t) or t.startswith(u) for u in tb))/len(ta)

def _nov(a, b):
    sa = {t for t in str(a).split() if re.search(r'\d',t)}
    sb = {t for t in str(b).split() if re.search(r'\d',t)}
    return len(sa&sb)/len(sa|sb) if sa and sb else 0.0

def _rov(a, b):
    ta = {t for t in str(a).split() if len(t)>=4}
    tb = {t for t in str(b).split() if len(t)>=4}
    return len(ta&tb)/(len(ta|tb)+1e-9) if ta and tb else 0.0


def featurise_batch(
    pairs: list,
    s1_lkp: dict, s23_lkp: dict,
    s1_emb: Optional[dict], s23_emb: Optional[dict],
) -> np.ndarray:
    """
    Feature matrix for a list of (s1_id, cand_id) pairs.
    Uses rapidfuzz for Levenshtein — ~1M pairs/sec.
    """
    N = len(pairs)
    s1_ids = [p[0] for p in pairs]
    c_ids  = [p[1] for p in pairs]

    n1 = [str(s1_lkp[s].norm_name) for s in s1_ids]
    n2 = [str(s23_lkp[c].norm_name) for c in c_ids]
    a1 = [str(s1_lkp[s].norm_addr) for s in s1_ids]
    a2 = [str(s23_lkp[c].norm_addr) for c in c_ids]

    # rapidfuzz batch calls
    lev_n  = np.array([rfuzz.ratio(x[:150], y[:150])/100 for x,y in zip(n1,n2)], dtype=np.float32)
    tss_n  = np.array([rfuzz.token_sort_ratio(x,y)/100    for x,y in zip(n1,n2)], dtype=np.float32)
    lev_a  = np.array([rfuzz.ratio(x[:100], y[:100])/100 for x,y in zip(a1,a2)], dtype=np.float32)

    # set-based features
    tj_n   = np.array([_tj(x,y)     for x,y in zip(n1,n2)], dtype=np.float32)
    c3_n   = np.array([_cng(x,y,3)  for x,y in zip(n1,n2)], dtype=np.float32)
    c4_n   = np.array([_cng(x,y,4)  for x,y in zip(n1,n2)], dtype=np.float32)
    aov_n  = np.array([_aov(x,y)    for x,y in zip(n1,n2)], dtype=np.float32)
    tj_a   = np.array([_tj(x,y)     for x,y in zip(a1,a2)], dtype=np.float32)
    c3_a   = np.array([_cng(x,y,3)  for x,y in zip(a1,a2)], dtype=np.float32)
    nov_a  = np.array([_nov(x,y)    for x,y in zip(a1,a2)], dtype=np.float32)
    rov_a  = np.array([_rov(x,y)    for x,y in zip(a1,a2)], dtype=np.float32)

    # metadata
    ctry   = np.array([int(s1_lkp[s].country==s23_lkp[c].country)        for s,c in zip(s1_ids,c_ids)], dtype=np.float32)
    bl     = np.array([int(bool(s1_lkp[s].has_legal)&bool(s23_lkp[c].has_legal)) for s,c in zip(s1_ids,c_ids)], dtype=np.float32)
    lxor   = np.array([int(bool(s1_lkp[s].has_legal)^bool(s23_lkp[c].has_legal)) for s,c in zip(s1_ids,c_ids)], dtype=np.float32)

    n1t = np.array([len(x.split()) for x in n1], dtype=np.float32)
    n2t = np.array([len(x.split()) for x in n2], dtype=np.float32)
    a1t = np.array([len(x.split()) for x in a1], dtype=np.float32)
    a2t = np.array([len(x.split()) for x in a2], dtype=np.float32)
    n1l = np.array([len(x) for x in n1], dtype=np.float32)
    n2l = np.array([len(x) for x in n2], dtype=np.float32)
    a1l = np.array([len(x) for x in a1], dtype=np.float32)
    a2l = np.array([len(x) for x in a2], dtype=np.float32)

    emb_n = np.zeros(N, dtype=np.float32)
    emb_a = np.zeros(N, dtype=np.float32)
    if s1_emb is not None and s23_emb is not None:
        for i,(s,c) in enumerate(zip(s1_ids, c_ids)):
            e1n, e1a = s1_emb.get(s,(None,None))
            e2n, e2a = s23_emb.get(c,(None,None))
            if e1n is not None and e2n is not None:
                emb_n[i] = float(np.dot(e1n,e2n))
                emb_a[i] = float(np.dot(e1a,e2a))

    mat = np.column_stack([
        tj_n, lev_n, c3_n, c4_n, tss_n, aov_n,   # 0-5 name
        tj_a, c3_a, nov_a, rov_a, lev_a,           # 6-10 addr
        emb_n, emb_a,                               # 11-12 semantic
        ctry, bl, lxor,                             # 13-15 meta
        np.abs(n1t-n2t),                            # 16
        np.minimum(n1l,n2l)/np.maximum(np.maximum(n1l,n2l),1),  # 17
        np.abs(a1t-a2t),                            # 18
        np.minimum(a1l,a2l)/np.maximum(np.maximum(a1l,a2l),1),  # 19
    ]).astype(np.float32)
    return mat


def build_feature_df(
    candidates: dict, s1_lkp: dict, s23_lkp: dict,
    s1_emb: Optional[dict], s23_emb: Optional[dict],
    labels: Optional[dict]=None, neg_ratio=6.0, seed=42,
) -> pd.DataFrame:
    rng = random.Random(seed)
    pairs, lbls = [], []
    for s1id, cands in candidates.items():
        valid = [c for c in cands if c in s23_lkp]
        if not valid: continue
        if labels is not None:
            pos = [c for c in valid if labels.get((s1id,c),0)==1]
            neg = [c for c in valid if labels.get((s1id,c),0)==0]
            n_neg = int(len(pos)*neg_ratio)+1
            if len(neg)>n_neg: neg = rng.sample(neg, n_neg)
            ec = pos+neg; ll = [1]*len(pos)+[0]*len(neg)
        else:
            ec = valid; ll = None
        for j,c in enumerate(ec):
            pairs.append((s1id,c))
            if ll is not None: lbls.append(ll[j])

    if not pairs:
        cols = FEATURE_COLS+['s1_id','cand_id']
        if labels: cols.append('label')
        return pd.DataFrame(columns=cols)

    print(f"    Computing {len(pairs):,} pairs ...", flush=True)
    t0 = time.time()
    X = featurise_batch(pairs, s1_lkp, s23_lkp, s1_emb, s23_emb)
    print(f"    Done ({time.time()-t0:.0f}s, {len(pairs)/(time.time()-t0):.0f} pairs/s)")
    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df['s1_id']   = [p[0] for p in pairs]
    df['cand_id'] = [p[1] for p in pairs]
    if labels is not None: df['label'] = lbls
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Embeddings  (optional — runs if cache present; otherwise skips)
# ─────────────────────────────────────────────────────────────────────────────

def _embed_df(df: pd.DataFrame, model, device, cp: Path, prefix: str, bs=1024):
    nc = Path(str(cp)+'_name.npy'); ac = Path(str(cp)+'_addr.npy'); ic = Path(str(cp)+'_ids.pkl')
    if nc.exists() and ac.exists() and ic.exists():
        ids = pickle.load(open(ic,'rb'))
        print(f"    Cache: {cp.name}*  ({len(df):,})")
        return dict(zip(ids, zip(np.load(nc), np.load(ac))))
    ids = df['entity_id'].tolist()
    names = [f"{prefix}: {t}" for t in df['norm_name'].fillna('').tolist()]
    addrs = [f"{prefix}: {t}" for t in df['norm_addr'].fillna('').tolist()]
    print(f"    Embedding {len(df):,} names [{prefix}] ...", flush=True)
    ne = model.encode(names, convert_to_numpy=True, show_progress_bar=True,
                      normalize_embeddings=True, batch_size=bs, device=device)
    print(f"    Embedding {len(df):,} addrs [{prefix}] ...", flush=True)
    ae = model.encode(addrs, convert_to_numpy=True, show_progress_bar=True,
                      normalize_embeddings=True, batch_size=bs, device=device)
    np.save(nc,ne); np.save(ac,ae); pickle.dump(ids,open(ic,'wb'))
    return dict(zip(ids, zip(ne, ae)))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Metric & threshold tuning
# ─────────────────────────────────────────────────────────────────────────────

def entity_f05(pred: set, true: set):
    if not true: return (1.,1.,1.) if not pred else (0.,0.,0.)
    if not pred: return 0., 1., 0.
    tp = len(pred&true); p = tp/len(pred); r = tp/len(true)
    if p+r==0: return 0.,0.,0.
    return (1.25*p*r)/(0.25*p+r), p, r

def macro_f05(preds, gt):
    fs,ps,rs = [],[],[]
    for s1id,true in gt.items():
        f,p,r = entity_f05(preds.get(s1id,set()), true)
        fs.append(f); ps.append(p); rs.append(r)
    return float(np.mean(fs)), float(np.mean(ps)), float(np.mean(rs))

def tune_threshold(sbe, gt):
    bt, bf, bp, br = 0.5, -1., 0., 0.
    for t in [round(x,3) for x in np.arange(0.05, 0.96, 0.01)]:
        preds = {s:{c for c,sc in cs if sc>=t} for s,cs in sbe.items()}
        f,p,r = macro_f05(preds, gt)
        if f>bf: bf,bt,bp,br = f,t,p,r
    bg = None
    for tg in [round(x,2) for x in np.arange(bt, 0.98, 0.03)]:
        preds = {}
        for s,cs in sbe.items():
            if not cs: preds[s]=set(); continue
            preds[s] = set() if max(sc for _,sc in cs)<tg else {c for c,sc in cs if sc>=bt}
        f,p,r = macro_f05(preds, gt)
        if f>bf: bf,bg,bp,br = f,tg,p,r
    return bt, bf, bp, br, bg


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # 0. Model
    print("\n[0] Loading multilingual-e5-small ...")
    from sentence_transformers import SentenceTransformer
    import torch
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    emb_model = SentenceTransformer("intfloat/multilingual-e5-small"); emb_model.eval()
    print(f"    Device={device}  ({time.time()-t0:.0f}s)")

    # 1. Load
    print("\n[1] Loading data ...")
    t = time.time()
    s1_tr  = pd.read_csv(TRAIN_DIR/"train_source1.tsv", sep="\t")
    s2_tr  = pd.read_csv(TRAIN_DIR/"train_source2.tsv", sep="\t")
    s3_tr  = pd.read_csv(TRAIN_DIR/"train_source3.tsv", sep="\t")
    gt_df  = pd.read_csv(TRAIN_DIR/"train_ground_truth.tsv", sep="\t")
    gt_df["matched_entity_ids"] = gt_df["matched_entity_ids"].fillna("")
    s1_te  = pd.read_csv(TEST_DIR/"test_source1.tsv",  sep="\t")
    s2_te  = pd.read_csv(TEST_DIR/"test_source2.tsv",  sep="\t")
    s3_te  = pd.read_csv(TEST_DIR/"test_source3.tsv",  sep="\t")
    s23_tr = pd.concat([s2_tr,s3_tr], ignore_index=True)
    s23_te = pd.concat([s2_te,s3_te], ignore_index=True)
    print(f"    Done ({time.time()-t:.0f}s): Train S1={len(s1_tr):,} S23={len(s23_tr):,} "
          f"| Test S1={len(s1_te):,} S23={len(s23_te):,}")

    # 2. Normalise
    print("\n[2] Normalising ...")
    s1_tr  = _cached_norm(s1_tr,  "s1_tr")
    s23_tr = _cached_norm(s23_tr, "s23_tr")
    s1_te  = _cached_norm(s1_te,  "s1_te")
    s23_te = _cached_norm(s23_te, "s23_te")
    print(f"    Done ({time.time()-t0:.0f}s)")

    # 3. Blocking
    print("\n[3] Blocking (train) ...")
    tr_cands = run_blocking(s1_tr, s23_tr, CACHE/"cands_train.pkl")
    tc_tr = sum(len(v) for v in tr_cands.values())
    print(f"    {tc_tr:,} train cand pairs, avg {tc_tr/max(len(tr_cands),1):.1f}/entity")

    print("\n[3b] Blocking (test) ...")
    te_cands = run_blocking(s1_te, s23_te, CACHE/"cands_test.pkl")
    tc_te = sum(len(v) for v in te_cands.values())
    print(f"    {tc_te:,} test cand pairs, avg {tc_te/max(len(te_cands),1):.1f}/entity")

    # 4. Ground truth
    print("\n[4] Ground truth ...")
    gt_dict: dict = {}; labels: dict = {}
    for _, row in gt_df.iterrows():
        s1id = row['source1_entity_id']
        mids = set(row['matched_entity_ids'].split(",")) if row['matched_entity_ids'].strip() else set()
        gt_dict[s1id] = mids
        for mid in mids: labels[(s1id,mid)] = 1
    print(f"    {len(gt_dict):,} S1 entities, {len(labels):,} pos pairs")

    # 5. Grouped 85/15 split
    print("\n[5] Grouped split ...")
    all_ids = list(gt_dict.keys())
    np.random.default_rng(RANDOM_SEED).shuffle(all_ids)
    n_val = int(0.15*len(all_ids))
    val_ids   = set(all_ids[:n_val])
    train_ids = set(all_ids[n_val:])
    print(f"    Train={len(train_ids):,}  Val={len(val_ids):,}")

    found = missed = 0
    for s1id in val_ids:
        true=gt_dict[s1id]; c=set(tr_cands.get(s1id,[]))
        found+=len(true&c); missed+=len(true-c)
    blk_recall = found/max(found+missed,1)
    print(f"    Blocking recall on val: {blk_recall*100:.2f}%")

    # 6. Selective embeddings
    print("\n[6] Selective embeddings ...")
    s23_tr_cids = set(); [s23_tr_cids.update(v) for v in tr_cands.values()]
    s23_te_cids = set(); [s23_te_cids.update(v) for v in te_cands.values()]
    s23_tr_c = s23_tr[s23_tr['entity_id'].isin(s23_tr_cids)].reset_index(drop=True)
    s23_te_c = s23_te[s23_te['entity_id'].isin(s23_te_cids)].reset_index(drop=True)
    print(f"    S23 train unique cands: {len(s23_tr_cids):,} | S23 test: {len(s23_te_cids):,}")

    def _emb_or_none(df, cp, prefix):
        nc = Path(str(cp)+'_name.npy'); ac = Path(str(cp)+'_addr.npy'); ic = Path(str(cp)+'_ids.pkl')
        if nc.exists() and ac.exists() and ic.exists():
            ids = pickle.load(open(ic,'rb'))
            print(f"    Cache: {cp.name}*  ({len(df):,})")
            return dict(zip(ids, zip(np.load(nc), np.load(ac))))
        # No cache — skip embeddings (run pipeline in lexical-only mode for speed)
        print(f"    No cache for {cp.name}* — skipping neural emb (lexical-only mode)")
        return None

    s1_tr_emb  = _emb_or_none(s1_tr,    CACHE/"emb_s1_tr",    'query')
    s23_tr_emb = _emb_or_none(s23_tr_c, CACHE/"emb_s23_tr_c", 'passage')
    s1_te_emb  = _emb_or_none(s1_te,    CACHE/"emb_s1_te",    'query')
    s23_te_emb = _emb_or_none(s23_te_c, CACHE/"emb_s23_te_c", 'passage')
    use_emb = all(e is not None for e in [s1_tr_emb,s23_tr_emb,s1_te_emb,s23_te_emb])
    sfx = '_emb' if use_emb else ''
    print(f"    Neural embeddings: {'ON' if use_emb else 'OFF (no cache — lexical only)'}  ({time.time()-t0:.0f}s)")

    # Build row lookups
    s1_tr_lkp  = {r.entity_id: r for r in s1_tr.itertuples(index=False)}
    s23_tr_lkp = {r.entity_id: r for r in s23_tr.itertuples(index=False)}
    s1_te_lkp  = {r.entity_id: r for r in s1_te.itertuples(index=False)}
    s23_te_lkp = {r.entity_id: r for r in s23_te.itertuples(index=False)}

    # 7. Training features
    print("\n[7] Training features ...")
    tr_cp = CACHE/f"feats_train{sfx}.pkl"
    if tr_cp.exists():
        print("    Cache"); df_tr = pickle.load(open(tr_cp,'rb'))
    else:
        sub = {e: tr_cands[e] for e in train_ids if e in tr_cands}
        df_tr = build_feature_df(sub, s1_tr_lkp, s23_tr_lkp, s1_tr_emb, s23_tr_emb, labels, 6.0)
        pickle.dump(df_tr, open(tr_cp,'wb'))
    print(f"    {len(df_tr):,} pairs  pos={int(df_tr['label'].sum()):,}  neg={(df_tr['label']==0).sum():,}")

    # 8. Val features
    print("\n[8] Val features ...")
    val_cp = CACHE/f"feats_val{sfx}.pkl"
    if val_cp.exists():
        print("    Cache"); df_val = pickle.load(open(val_cp,'rb'))
    else:
        vc = {e: tr_cands[e] for e in val_ids if e in tr_cands}
        df_val = build_feature_df(vc, s1_tr_lkp, s23_tr_lkp, s1_tr_emb, s23_tr_emb, None)
        pickle.dump(df_val, open(val_cp,'wb'))
    print(f"    {len(df_val):,} val pairs")

    # 9. Train LightGBM
    print("\n[9] Training LightGBM ...")
    mdl_cp = CACHE/f"lgbm{sfx}.pkl"
    if mdl_cp.exists():
        print("    Cache"); clf = pickle.load(open(mdl_cp,'rb'))
    else:
        X = df_tr[FEATURE_COLS].values.astype(np.float32)
        y = df_tr['label'].values.astype(np.int32)
        pw = float((y==0).sum())/float(max((y==1).sum(),1))
        t1 = time.time()
        clf = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.05, max_depth=8,
                                  num_leaves=127, min_child_samples=20,
                                  subsample=0.8, colsample_bytree=0.8,
                                  scale_pos_weight=pw, random_state=RANDOM_SEED,
                                  n_jobs=-1, verbose=-1)
        clf.fit(X, y)
        pickle.dump(clf, open(mdl_cp,'wb'))
        print(f"    Done ({time.time()-t1:.0f}s)")
    fi = sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x:-x[1])[:5]
    print(f"    Top-5: {fi}")

    # 10. Threshold tuning
    print("\n[10] Threshold tuning ...")
    sc = clf.predict_proba(df_val[FEATURE_COLS].values.astype(np.float32))[:,1]
    df_val = df_val.copy(); df_val['score'] = sc
    sbe = collections.defaultdict(list)
    for _, row in df_val.iterrows(): sbe[row['s1_id']].append((row['cand_id'],row['score']))
    for s1id in val_ids:
        if s1id not in sbe: sbe[s1id] = []
    val_gt = {s: gt_dict[s] for s in val_ids}
    bt, bf, bp, br, bg = tune_threshold(dict(sbe), val_gt)
    print(f"    F0.5={bf:.4f}  P={bp:.4f}  R={br:.4f}  threshold={bt:.3f}  gate={bg}")

    # 11. Test features
    print("\n[11] Test features ...")
    te_cp = CACHE/f"feats_test{sfx}.pkl"
    if te_cp.exists():
        print("    Cache"); df_te = pickle.load(open(te_cp,'rb'))
    else:
        df_te = build_feature_df(te_cands, s1_te_lkp, s23_te_lkp, s1_te_emb, s23_te_emb, None)
        pickle.dump(df_te, open(te_cp,'wb'))
    print(f"    {len(df_te):,} test pairs")

    # 12. Inference
    print("\n[12] Inference ...")
    if len(df_te)>0:
        sc_te = clf.predict_proba(df_te[FEATURE_COLS].values.astype(np.float32))[:,1]
        df_te = df_te.copy(); df_te['score'] = sc_te
    else:
        df_te['score'] = pd.Series(dtype=float)

    te_sbe = collections.defaultdict(list)
    for _, row in df_te.iterrows(): te_sbe[row['s1_id']].append((row['cand_id'],row['score']))

    all_te = list(s1_te['entity_id'])
    preds = {}
    for s1id in all_te:
        cs = te_sbe.get(s1id,[])
        if not cs: preds[s1id]=set(); continue
        if bg is not None and max(sc for _,sc in cs)<bg: preds[s1id]=set(); continue
        preds[s1id] = {c for c,sc in cs if sc>=bt}

    # 13. Write outputs
    print("\n[13] Writing outputs ...")
    with open(OUTPUT/"candidate_pairs.tsv","w",encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1id in all_te:
            cl = list(dict.fromkeys(te_cands.get(s1id,[])))
            f.write(f"{s1id}\t{','.join(cl)}\n")
    with open(OUTPUT/"matching_results.tsv","w",encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1id in all_te:
            allowed = set(te_cands.get(s1id,[]))
            mset = preds.get(s1id,set()) & allowed
            f.write(f"{s1id}\t{','.join(sorted(mset))}\n")

    n_match = sum(1 for s in all_te if preds.get(s))
    print(f"    {len(all_te):,} rows: {n_match:,} with matches, {len(all_te)-n_match:,} singletons")

    # Summary
    tc_te_l = [len(te_cands.get(s,[])) for s in all_te]
    brute   = len(s1_te)*(len(s2_te)+len(s3_te))
    actual  = sum(tc_te_l)
    elapsed = (time.time()-t0)/60

    print("\n"+"="*65)
    print(f"PIPELINE COMPLETE  ({elapsed:.1f} min)")
    print(f"\nROW COUNTS:")
    print(f"  Train S1={len(s1_tr):,} S2={len(s2_tr):,} S3={len(s3_tr):,} GT={len(gt_df):,}")
    print(f"  Test  S1={len(s1_te):,} S2={len(s2_te):,} S3={len(s3_te):,}")
    print(f"\nCANDIDATE-SET STATS (test):")
    print(f"  Avg/entity={np.mean(tc_te_l):.1f}  Max={max(tc_te_l) if tc_te_l else 0}")
    print(f"  Total={actual:,}  Brute={brute:,}  Reduction={brute/max(actual,1):.0f}x")
    print(f"\nVALIDATION (entity-level macro):")
    print(f"  F0.5={bf:.4f}  P={bp:.4f}  R={br:.4f}")
    print(f"  Threshold={bt:.3f}  Gate={bg}")
    print(f"  Blocking recall on val: {blk_recall*100:.2f}%")
    print(f"\nTRAINING:")
    print(f"  Pos:Neg=1:6  LightGBM 800 trees  Train={len(train_ids):,}  Val={len(val_ids):,}")
    print(f"  Neural embeddings: {'ON' if use_emb else 'OFF'}")
    print("="*65)


if __name__ == "__main__":
    main()
