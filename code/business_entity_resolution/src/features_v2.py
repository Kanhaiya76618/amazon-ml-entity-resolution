"""
features_v2.py — High-precision, high-throughput C++ feature extractor.

Computes 25 discriminative features across lexical, phonetic, transliterated,
structural, and numeric properties for candidate pairs.
"""

from __future__ import annotations
import re
import numpy as np
from rapidfuzz import fuzz as rfuzz
from rapidfuzz.distance import JaroWinkler

FEATURE_NAMES = [
    'name_tj', 'name_lev', 'name_jw', 'name_tsr', 'name_c3', 'name_c4',
    'name_tr_lev', 'name_tr_jw', 'name_tr_tsr',
    'addr_tj', 'addr_lev', 'addr_jw', 'addr_tsr', 'addr_c3', 'addr_nov',
    'num_exact', 'acro_match', 'both_legal', 'legal_xor',
    'n_tlen_diff', 'n_lrat', 'a_tlen_diff', 'a_lrat',
    'first_tok_match', 'prescore'
]

def _token_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def _char_ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    if len(a) < n or len(b) < n:
        return 0.0
    sa = {a[i:i+n] for i in range(len(a) - n + 1)}
    sb = {b[i:i+n] for i in range(len(b) - n + 1)}
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0

def _numeric_overlap(a: str, b: str) -> float:
    sa = {t for t in a.split() if re.search(r'\d', t)}
    sb = {t for t in b.split() if re.search(r'\d', t)}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def compute_features_batch(
    s1_tuples: list[tuple[str, str, str, str, str, bool]], # (norm_n, rom_n, norm_a, num, acro, has_legal)
    s23_tuples: list[tuple[str, str, str, str, str, bool]],
    prescores: list[float] | np.ndarray = None
) -> np.ndarray:
    """
    Vectorized batch feature extractor using C++ rapidfuzz.
    Input: parallel lists of S1 and S23 attribute tuples.
    Returns: (N, 25) float32 numpy feature matrix.
    """
    N = len(s1_tuples)
    if N == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        
    n1 = [t[0] for t in s1_tuples]
    r1 = [t[1] for t in s1_tuples]
    a1 = [t[2] for t in s1_tuples]
    u1 = [t[3] for t in s1_tuples]
    c1 = [t[4] for t in s1_tuples]
    l1 = [t[5] for t in s1_tuples]
    
    n2 = [t[0] for t in s23_tuples]
    r2 = [t[1] for t in s23_tuples]
    a2 = [t[2] for t in s23_tuples]
    u2 = [t[3] for t in s23_tuples]
    c2 = [t[4] for t in s23_tuples]
    l2 = [t[5] for t in s23_tuples]
    
    # 1. Name features
    name_tj  = np.array([_token_jaccard(x, y) for x, y in zip(n1, n2)], dtype=np.float32)
    name_lev = np.array([rfuzz.ratio(x[:120], y[:120]) / 100.0 for x, y in zip(n1, n2)], dtype=np.float32)
    name_jw  = np.array([JaroWinkler.similarity(x[:120], y[:120]) for x, y in zip(n1, n2)], dtype=np.float32)
    name_tsr = np.array([rfuzz.token_set_ratio(x[:120], y[:120]) / 100.0 for x, y in zip(n1, n2)], dtype=np.float32)
    name_c3  = np.array([_char_ngram_jaccard(x, y, 3) for x, y in zip(n1, n2)], dtype=np.float32)
    name_c4  = np.array([_char_ngram_jaccard(x, y, 4) for x, y in zip(n1, n2)], dtype=np.float32)
    
    # 2. Transliterated features
    name_tr_lev = np.array([rfuzz.ratio(x[:120], y[:120]) / 100.0 for x, y in zip(r1, r2)], dtype=np.float32)
    name_tr_jw  = np.array([JaroWinkler.similarity(x[:120], y[:120]) for x, y in zip(r1, r2)], dtype=np.float32)
    name_tr_tsr = np.array([rfuzz.token_set_ratio(x[:120], y[:120]) / 100.0 for x, y in zip(r1, r2)], dtype=np.float32)
    
    # 3. Address features
    addr_tj  = np.array([_token_jaccard(x, y) for x, y in zip(a1, a2)], dtype=np.float32)
    addr_lev = np.array([rfuzz.ratio(x[:100], y[:100]) / 100.0 for x, y in zip(a1, a2)], dtype=np.float32)
    addr_jw  = np.array([JaroWinkler.similarity(x[:100], y[:100]) for x, y in zip(a1, a2)], dtype=np.float32)
    addr_tsr = np.array([rfuzz.token_set_ratio(x[:100], y[:100]) / 100.0 for x, y in zip(a1, a2)], dtype=np.float32)
    addr_c3  = np.array([_char_ngram_jaccard(x, y, 3) for x, y in zip(a1, a2)], dtype=np.float32)
    addr_nov = np.array([_numeric_overlap(x, y) for x, y in zip(a1, a2)], dtype=np.float32)
    
    # 4. Identity & Numeric anchors
    num_exact  = np.array([float(bool(x and y and x == y)) for x, y in zip(u1, u2)], dtype=np.float32)
    acro_match = np.array([float(bool(x and y and x == y)) for x, y in zip(c1, c2)], dtype=np.float32)
    both_legal = np.array([float(bool(x and y)) for x, y in zip(l1, l2)], dtype=np.float32)
    legal_xor  = np.array([float(bool(x ^ y)) for x, y in zip(l1, l2)], dtype=np.float32)
    
    # 5. Length & Token counts
    n1_lens = np.array([len(x) for x in n1], dtype=np.float32)
    n2_lens = np.array([len(x) for x in n2], dtype=np.float32)
    a1_lens = np.array([len(x) for x in a1], dtype=np.float32)
    a2_lens = np.array([len(x) for x in a2], dtype=np.float32)
    
    n1_toks = np.array([len(x.split()) for x in n1], dtype=np.float32)
    n2_toks = np.array([len(x.split()) for x in n2], dtype=np.float32)
    a1_toks = np.array([len(x.split()) for x in a1], dtype=np.float32)
    a2_toks = np.array([len(x.split()) for x in a2], dtype=np.float32)
    
    n_tlen_diff = np.abs(n1_toks - n2_toks)
    n_lrat      = np.minimum(n1_lens, n2_lens) / np.maximum(np.maximum(n1_lens, n2_lens), 1.0)
    a_tlen_diff = np.abs(a1_toks - a2_toks)
    a_lrat      = np.minimum(a1_lens, a2_lens) / np.maximum(np.maximum(a1_lens, a2_lens), 1.0)
    
    first_tok_match = np.array([
        float(bool(x.split() and y.split() and x.split()[0] == y.split()[0]))
        for x, y in zip(n1, n2)
    ], dtype=np.float32)
    
    if prescores is not None:
        ps = np.asarray(prescores, dtype=np.float32)
    else:
        ps = np.maximum(name_lev, name_tr_lev) * 1.2 + addr_lev
        
    mat = np.column_stack([
        name_tj, name_lev, name_jw, name_tsr, name_c3, name_c4,
        name_tr_lev, name_tr_jw, name_tr_tsr,
        addr_tj, addr_lev, addr_jw, addr_tsr, addr_c3, addr_nov,
        num_exact, acro_match, both_legal, legal_xor,
        n_tlen_diff, n_lrat, a_tlen_diff, a_lrat,
        first_tok_match, ps
    ]).astype(np.float32)
    
    return mat
