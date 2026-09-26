"""
blocking.py — High-recall candidate generation via inverted index.

Strategy (union of multiple cheap keys for maximum recall):
  K1: sorted name tokens (all 2-token subsequences / full sorted bag)
  K2: first-3-chars prefix of longest name token
  K3: any shared "rare" name token  (IDF-weighted, keep top-N)
  K4: first numeric token of address (house/street number)
  K5: sorted pair of the two longest address tokens
  K6: country equality hard gate (applied AFTER unioning — just restricts cross-country noise)
       NOTE: we do NOT filter on country — we include cross-country pairs but
       country_match=False becomes a feature. For recall we keep everything.

For scale (2.2M S1, 10M+ S2/S3): we build the index over S2+S3 combined,
then for each S1 entity look up all its keys and union candidates.
"""

from __future__ import annotations

import re
import collections
from itertools import combinations
from typing import Iterable
import pandas as pd

from normalize import normalize_name, normalize_address


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def _tokens(text: str) -> list[str]:
    return [t for t in text.split() if len(t) >= 2]


def _sorted_token_pairs(tokens: list[str]) -> list[str]:
    """All sorted 2-token combinations → blocking keys."""
    keys = []
    toks = sorted(set(tokens))
    for a, b in combinations(toks, 2):
        keys.append(f"{a}|{b}")
    # Also the full sorted bag as a single key (for exact-ish matches)
    if toks:
        keys.append('|'.join(toks[:6]))  # cap at 6 tokens for key size
    return keys


def _prefix_key(tokens: list[str], n: int = 4) -> str | None:
    """Prefix of the longest token."""
    if not tokens:
        return None
    longest = max(tokens, key=len)
    if len(longest) >= n:
        return f"pfx:{longest[:n]}"
    return None


def _numeric_token(addr: str) -> str | None:
    """First digit-containing token of an address."""
    for t in addr.split():
        if re.search(r'\d', t):
            return f"num:{t}"
    return None


def _top_addr_tokens(addr: str, n: int = 2) -> str | None:
    """Sorted pair of the two longest (≥4 char) address tokens."""
    toks = sorted([t for t in addr.split() if len(t) >= 4], key=len, reverse=True)[:n]
    if len(toks) >= 2:
        return 'adr:' + '|'.join(sorted(toks[:2]))
    elif toks:
        return f"adr1:{toks[0]}"
    return None


def _strip_domain(text: str) -> str:
    """
    If text looks like a domain/URL (contains '.com'/'.org'/etc. or '#'),
    extract the base name part for additional blocking.
    e.g. 'warrenfitness.com' → 'warrenfitness'
         '#gulfassociation'  → 'gulfassociation'
    """
    text = re.sub(r'^#+', '', text)  # strip leading hashes
    text = re.sub(r'\.(com|org|net|co|in|fr|biz|info|io|us|uk)$', '', text, flags=re.IGNORECASE)
    # Split CamelCase / run-together words at case boundaries
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return text.lower().strip()


def _build_keys(norm_name: str, norm_addr: str, raw_name: str = '') -> list[str]:
    """Build all blocking keys for one entity."""
    name_toks = _tokens(norm_name)
    keys = []
    # K1: sorted name token pairs
    keys.extend(_sorted_token_pairs(name_toks))
    # K2: prefix of longest name token
    pfx = _prefix_key(name_toks, n=4)
    if pfx:
        keys.append(pfx)
    # K3: each individual name token (for rare tokens — frequency filtered at index time)
    for t in name_toks:
        keys.append(f"tok:{t}")
    # K4: first numeric address token
    num = _numeric_token(norm_addr)
    if num:
        keys.append(num)
    # K5: top address token pair
    atp = _top_addr_tokens(norm_addr)
    if atp:
        keys.append(atp)
    # K6: domain-stripped name (handles warrenfitness.com → warrenfitness)
    if raw_name:
        stripped = _strip_domain(str(raw_name))
        s_toks = _tokens(stripped)
        if s_toks and s_toks != name_toks:
            keys.extend(_sorted_token_pairs(s_toks))
            for t in s_toks:
                keys.append(f"tok:{t}")
    return list(set(keys))


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

class BlockingIndex:
    """
    Inverted index over S2+S3 entities.
    """

    def __init__(self, max_postings: int = 500):
        """
        max_postings: skip keys whose posting list exceeds this length
                      (very common tokens like 'and','llc' become useless).
        """
        self.max_postings = max_postings
        self._index: dict[str, list[str]] = collections.defaultdict(list)
        self._freq: dict[str, int] = collections.Counter()

    def build(self, df: pd.DataFrame) -> None:
        """
        Build index from a combined S2+S3 DataFrame.
        Columns: entity_id, business_name, business_address, country
        """
        print(f"  Building blocking index over {len(df):,} S2/S3 entities...")
        for row in df.itertuples(index=False):
            raw_name  = row.business_name if pd.notna(row.business_name) else ''
            norm_name, _ = normalize_name(raw_name)
            norm_addr    = normalize_address(row.business_address if pd.notna(row.business_address) else '')
            for key in _build_keys(norm_name, norm_addr, raw_name=raw_name):
                self._freq[key] += 1
                self._index[key].append(row.entity_id)

        # Prune over-populated keys (global frequency > max_postings)
        pruned = {k for k, v in self._freq.items() if v > self.max_postings}
        for k in pruned:
            del self._index[k]
        print(f"  Index: {len(self._index):,} keys, pruned {len(pruned):,} high-freq keys")

    def lookup(self, norm_name: str, norm_addr: str, raw_name: str = '') -> set[str]:
        """Return candidate S2/S3 entity IDs for a single S1 entity."""
        candidates: set[str] = set()
        for key in _build_keys(norm_name, norm_addr, raw_name=raw_name):
            if key in self._index:
                candidates.update(self._index[key])
        return candidates


# ---------------------------------------------------------------------------
# Batch candidate generation
# ---------------------------------------------------------------------------

def generate_candidates(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    max_postings: int = 500,
    max_candidates_per_entity: int = 300,
) -> dict[str, list[str]]:
    """
    For each S1 entity, return a list of candidate S2/S3 entity IDs.

    Returns:
        dict  {s1_entity_id: [candidate_ids...]}
    """
    idx = BlockingIndex(max_postings=max_postings)
    idx.build(s23_df)

    result: dict[str, list[str]] = {}
    total = 0

    for row in s1_df.itertuples(index=False):
        raw_name  = row.business_name if pd.notna(row.business_name) else ''
        norm_name, _ = normalize_name(raw_name)
        norm_addr    = normalize_address(row.business_address if pd.notna(row.business_address) else '')
        cands = idx.lookup(norm_name, norm_addr, raw_name=raw_name)
        # Cap per entity
        if len(cands) > max_candidates_per_entity:
            cands = set(list(cands)[:max_candidates_per_entity])
        result[row.entity_id] = list(cands)
        total += len(cands)

    n = len(result)
    print(f"  Blocking: {n:,} S1 entities → {total:,} candidate pairs "
          f"(avg {total/max(n,1):.1f}/entity)")
    return result


if __name__ == '__main__':
    # Quick smoke test
    import pandas as pd
    s1 = pd.DataFrame([
        {'entity_id': 'S1-1', 'business_name': "Prime Money Corp",
         'business_address': "17560 Ellis Road, Tahlequah, OK", 'country': 'US'},
    ])
    s23 = pd.DataFrame([
        {'entity_id': 'S2-1', 'business_name': "PRIME MONEY CORPORATION",
         'business_address': "17560 Ellis Rd, Tahlequah", 'country': 'US'},
        {'entity_id': 'S3-1', 'business_name': "Prime Money Ltd",
         'business_address': "Ellis Road Tahlequah", 'country': 'US'},
        {'entity_id': 'S2-2', 'business_name': "Completely Different Inc",
         'business_address': "100 Oak St, Dallas TX", 'country': 'US'},
    ])
    cands = generate_candidates(s1, s23, max_postings=500)
    print("Candidates for S1-1:", cands['S1-1'])
