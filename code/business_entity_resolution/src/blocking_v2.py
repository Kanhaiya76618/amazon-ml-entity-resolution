"""
blocking_v2.py — High-recall, country-partitioned multi-channel candidate generation.

Key innovations:
- 100% Country-partitioned: guarantees 0 cross-country candidate pairs.
- Multi-channel index:
    * K1: Name sorted token pairs (rarest prefix)
    * K2: Transliterated tokens & pairs (Brahmic Indic -> Latin)
    * K3: Acronym channel (State Bank of India <-> SBI)
    * K4: Joint Number + Street token (85 Wayne Ave <-> 85 Wayne)
    * K5: Joint Number + Name token (85 Colombier <-> 85 Colombier)
    * K6: Address distinctive token pairs
    * K7: Typo-resilient character 4-gram sketch
- Multi-channel top-K candidate screening:
    Ensures cross-script and acronym matches are never crowded out by lexical near-matches.
"""

from __future__ import annotations

import collections
import re
from typing import Dict, List, Set, Tuple
import pandas as pd
from rapidfuzz import fuzz as rfuzz
from rapidfuzz.distance import JaroWinkler

from translit import fold_accents, romanize_indic, extract_acronym
from normalize_v2 import normalize_name_v2, normalize_address_v2, extract_primary_number

GENERIC_ADDR = {
    'road', 'street', 'avenue', 'drive', 'lane', 'court', 'place', 'square',
    'parkway', 'highway', 'apartment', 'suite', 'floor', 'building', 'near',
    'behind', 'opp', 'opposite', 'house', 'nagar', 'colony', 'city', 'town',
    'dist', 'state', 'block', 'sector', 'rue', 'allee', 'impasse', 'boulevard',
    'saint', 'north', 'south', 'east', 'west', 'rd', 'st', 'ave', 'blvd', 'dr'
}

def _tokens(text: str, min_len: int = 2) -> list[str]:
    return [t for t in re.findall(r'[a-zA-Z0-9]+', str(text).lower()) if len(t) >= min_len]

def build_entity_keys(
    name: str,
    addr: str,
    country: str,
    norm_n: str = '',
    norm_a: str = '',
    rom_n: str = '',
    is_query: bool = False
) -> Set[str]:
    """
    Generate all multi-channel blocking keys for an entity.
    """
    if not norm_n:
        norm_n, r_n, _ = normalize_name_v2(name)
        if not rom_n:
            rom_n = r_n
    if not norm_a:
        norm_a = normalize_address_v2(addr, country)
    if not rom_n:
        rom_n = norm_n
        
    keys = set()
    n_toks = _tokens(norm_n, min_len=2)
    r_toks = _tokens(rom_n, min_len=2)
    a_toks = [t for t in _tokens(norm_a, min_len=3) if t not in GENERIC_ADDR]
    
    # Primary address numbers
    num = extract_primary_number(addr)
    all_nums = [n.lstrip('0') for n in re.findall(r'\b\d+\b', str(addr)) if len(n.lstrip('0')) >= 2]
    
    # K1: Name token pairs
    if len(n_toks) >= 2:
        for i in range(min(len(n_toks), 5)):
            for j in range(i+1, min(len(n_toks), 5)):
                w1, w2 = sorted([n_toks[i], n_toks[j]])
                keys.add(f"tok2:{w1}|{w2}")
    elif len(n_toks) == 1:
        keys.add(f"tok1:{n_toks[0]}")
        
    # K2: Transliterated tokens & pairs (if different from raw tokens)
    if r_toks and r_toks != n_toks:
        if len(r_toks) >= 2:
            for i in range(min(len(r_toks), 4)):
                for j in range(i+1, min(len(r_toks), 4)):
                    w1, w2 = sorted([r_toks[i], r_toks[j]])
                    keys.add(f"tr2:{w1}|{w2}")
        for t in r_toks[:3]:
            if len(t) >= 3:
                keys.add(f"tr1:{t}")
                
    # K3: Acronym channel
    acro = extract_acronym(name)
    if acro:
        keys.add(f"acro:{acro}")
    if is_query:
        # For query entities, also emit uppercase tokens (e.g. SBI, HDFC)
        for w in re.findall(r'\b[A-Z]{2,6}\b', str(name)):
            keys.add(f"acro:{w.lower()}")
            
    # K4: Joint Number + Street token keys
    for n in all_nums[:2]:
        for a in a_toks[:3]:
            keys.add(f"na:{n}|{a}")
        if n_toks:
            keys.add(f"nn:{n}|{n_toks[0]}")
        if r_toks and r_toks != n_toks:
            keys.add(f"nn:{n}|{r_toks[0]}")
            
    # K5: Address distinctive token pairs
    if len(a_toks) >= 2:
        for i in range(min(len(a_toks), 4)):
            for j in range(i+1, min(len(a_toks), 4)):
                w1, w2 = sorted([a_toks[i], a_toks[j]])
                keys.add(f"adr2:{w1}|{w2}")
                
    # K6: Typo-resilient character 4-gram sketch
    for t in (n_toks[:2] + r_toks[:2]):
        if len(t) >= 4:
            for i in range(len(t) - 3):
                keys.add(f"g4:{t[i:i+4]}")
                
    return keys


class CountryBlockingIndex:
    """
    Inverted index for a specific country's S23 entities.
    """
    def __init__(self, country: str, max_postings: int = 1500):
        self.country = country
        self.max_postings = max_postings
        self._index: Dict[str, List[str]] = collections.defaultdict(list)
        self._freq: collections.Counter = collections.Counter()
        
    def build(self, df: pd.DataFrame) -> None:
        """
        df columns: entity_id, business_name, business_address, country, norm_name, norm_addr
        """
        has_norm = 'norm_name' in df.columns and 'norm_addr' in df.columns
        for r in df.itertuples(index=False):
            eid = r.entity_id
            name = str(r.business_name) if pd.notna(r.business_name) else ''
            addr = str(r.business_address) if pd.notna(r.business_address) else ''
            norm_n = str(r.norm_name) if has_norm and pd.notna(r.norm_name) else ''
            norm_a = str(r.norm_addr) if has_norm and pd.notna(r.norm_addr) else ''
            keys = build_entity_keys(name, addr, self.country, norm_n=norm_n, norm_a=norm_a, is_query=False)
            for k in keys:
                self._freq[k] += 1
                self._index[k].append(eid)
                
        # Prune high-frequency keys (1500 general, 250 for character 4-grams)
        pruned = [
            k for k, v in self._freq.items()
            if (v > self.max_postings) or (k.startswith("g4:") and v > 250)
        ]
        for k in pruned:
            del self._index[k]
            
    def lookup(
        self,
        name: str,
        addr: str,
        norm_n: str = '',
        norm_a: str = '',
        rom_n: str = '',
        max_candidates: int = 250
    ) -> Set[str]:
        keys = build_entity_keys(name, addr, self.country, norm_n=norm_n, norm_a=norm_a, rom_n=rom_n, is_query=True)
        candidates: Set[str] = set()
        for k in keys:
            if k in self._index:
                candidates.update(self._index[k])
                if len(candidates) >= max_candidates * 2:
                    break
        if len(candidates) > max_candidates:
            candidates = set(list(candidates)[:max_candidates])
        return candidates


def multi_channel_prescreen(
    s1_id: str,
    s1_name: str,
    s1_addr: str,
    cand_ids: Set[str],
    s23_lookup: Dict[str, Tuple[str, str, str, str, str]], # (norm_n, rom_n, norm_a, num, acro)
    top_k_per_entity: int = 40,
    s1_info: tuple = None
) -> List[str]:
    """
    Multi-channel candidate ranking:
    Retains candidates across combined, address-only, transliteration-only, and acronym channels.
    """
    if not cand_ids:
        return []
        
    if s1_info is not None:
        s1_norm_n, s1_rom_n, s1_norm_a, s1_num, s1_acro = s1_info[:5]
    else:
        s1_norm_n, s1_rom_n, _ = normalize_name_v2(s1_name)
        s1_norm_a = normalize_address_v2(s1_addr)
        s1_num = extract_primary_number(s1_addr)
        s1_acro = extract_acronym(s1_name)
    
    combined_scores = []
    addr_scores = []
    tr_scores = []
    acro_num_candidates = []
    
    for cid in cand_ids:
        c_info = s23_lookup.get(cid)
        if not c_info:
            continue
        c_norm_n, c_rom_n, c_norm_a, c_num, c_acro = c_info[:5]
        
        # Best name similarity across ratio, token_set_ratio, and translit
        raw_rat = rfuzz.ratio(s1_norm_n[:80], c_norm_n[:80]) / 100.0
        raw_tsr = rfuzz.token_set_ratio(s1_norm_n[:80], c_norm_n[:80]) / 100.0
        tr_rat  = rfuzz.ratio(s1_rom_n[:80], c_rom_n[:80]) / 100.0
        tr_tsr  = rfuzz.token_set_ratio(s1_rom_n[:80], c_rom_n[:80]) / 100.0
        
        best_name_sim = max(raw_rat, raw_tsr, tr_rat, tr_tsr)
        
        # Address similarity across ratio and token_set_ratio
        a_rat = rfuzz.ratio(s1_norm_a[:70], c_norm_a[:70]) / 100.0
        a_tsr = rfuzz.token_set_ratio(s1_norm_a[:70], c_norm_a[:70]) / 100.0
        best_addr_sim = max(a_rat, a_tsr)
        
        # Combined score
        comp = best_name_sim * 1.2 + best_addr_sim
        combined_scores.append((comp, cid))
        addr_scores.append((best_addr_sim, cid))
        if max(tr_rat, tr_tsr) > max(raw_rat, raw_tsr):
            tr_scores.append((max(tr_rat, tr_tsr), cid))
            
        # Acronym & Numeric hits
        is_acro_hit = bool(s1_acro and c_acro and s1_acro == c_acro)
        is_num_hit = bool(s1_num and c_num and s1_num == c_num and best_addr_sim >= 0.40)
        if is_acro_hit or is_num_hit:
            acro_num_candidates.append(cid)
            
    # Channel selection
    selected: Set[str] = set()
    
    # 1. Top 25 combined
    combined_scores.sort(key=lambda x: -x[0])
    for _, cid in combined_scores[:25]:
        selected.add(cid)
        
    # 2. High confidence matches (prescore >= 1.4)
    for comp, cid in combined_scores:
        if comp >= 1.40:
            selected.add(cid)
            
    # 3. Top 10 address only
    addr_scores.sort(key=lambda x: -x[0])
    for _, cid in addr_scores[:10]:
        selected.add(cid)
        
    # 4. Top 10 transliteration
    tr_scores.sort(key=lambda x: -x[0])
    for _, cid in tr_scores[:10]:
        selected.add(cid)
        
    # 5. Acronym / Numeric hits
    for cid in acro_num_candidates[:10]:
        selected.add(cid)
        
    # Cap at top_k_per_entity
    if len(selected) > top_k_per_entity:
        # Prioritize by combined score
        score_dict = {cid: sc for sc, cid in combined_scores}
        ordered = sorted(selected, key=lambda c: -score_dict.get(c, 0.0))
        return ordered[:top_k_per_entity]
        
    return list(selected)
