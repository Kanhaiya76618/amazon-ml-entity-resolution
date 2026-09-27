"""
reference_scorer.py — Official entity-level macro F0.5 metric evaluation.

Matches the competition metric exactly:
- Macro-averaged over all test S1 entities.
- For each entity:
    - If truth is empty (singleton) and pred is empty -> score = 1.0
    - If truth is non-empty and pred is empty -> score = 0.0 (recall=0, precision=0)
    - If truth is empty and pred is non-empty -> score = 0.0 (precision=0)
    - Otherwise F_beta with beta=0.5 (beta^2 = 0.25):
        F0.5 = (1 + 0.25) * P * R / (0.25 * P + R)
"""

from typing import Dict, Set, Tuple

def fbeta(p: float, r: float, b2: float = 0.25) -> float:
    if p + r == 0.0:
        return 0.0
    denom = b2 * p + r
    if denom == 0.0:
        return 0.0
    return (1.0 + b2) * p * r / denom

def compute_macro_f05(truth: Dict[str, Set[str]], pred: Dict[str, Set[str]]) -> Tuple[float, Dict[str, float]]:
    """
    truth: {s1_entity_id: set(matched_ids)}
    pred:  {s1_entity_id: set(matched_ids)}
    
    Returns:
        macro_f05: float
        per_entity_scores: {s1_entity_id: score}
    """
    per_entity = {}
    for eid, T in truth.items():
        P = pred.get(eid, set())
        if not T and not P:
            per_entity[eid] = 1.0
        elif not T or not P:
            per_entity[eid] = 0.0
        else:
            tp = len(T & P)
            p = tp / len(P) if P else 0.0
            r = tp / len(T) if T else 0.0
            per_entity[eid] = fbeta(p, r, b2=0.25)
            
    score = sum(per_entity.values()) / max(len(per_entity), 1)
    return score, per_entity

if __name__ == "__main__":
    # Unit tests
    t = {"e1": {"a", "b"}, "e2": set(), "e3": {"c"}}
    p = {"e1": {"a", "b"}, "e2": set(), "e3": set()}
    s, _ = compute_macro_f05(t, p)
    print(f"Test score (1.0, 1.0, 0.0) -> Expected: 0.6667, Got: {s:.4f}")
