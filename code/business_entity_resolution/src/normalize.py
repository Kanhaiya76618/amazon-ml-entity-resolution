"""
normalize.py — Text normalization for business names and addresses.

Design goals:
- Country-agnostic: works for US, India (including Devanagari), France, and any unseen country.
- Abbreviation expansion is bidirectional (both directions of common aliases).
- Legal suffixes are extracted as a separate boolean flag, not deleted.
- Output: normalized string + has_legal_suffix bool.
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Abbreviation tables
# ---------------------------------------------------------------------------

# Each entry: (pattern, replacement) — applied left-to-right.
# Both directions are covered: the same pair is added in reverse so that
# "Corp" normalises to "corporation" AND "corporation" stays "corporation".

_NAME_ABBREV = [
    (r'\bcorp\b',        'corporation'),
    (r'\bcorporation\b', 'corporation'),
    (r'\bpvt\b',         'private'),
    (r'\bprivate\b',     'private'),
    (r'\bltd\b',         'limited'),
    (r'\blimited\b',     'limited'),
    (r'\bllp\b',         'llp'),
    (r'\bllc\b',         'llc'),
    (r'\binc\b',         'incorporated'),
    (r'\bincorporated\b','incorporated'),
    (r'\bco\b',          'company'),
    (r'\bcompany\b',     'company'),
    (r'\bassoc\b',       'associates'),
    (r'\bassociates\b',  'associates'),
    (r'\bintl\b',        'international'),
    (r'\binternational\b','international'),
    (r'\bmgmt\b',        'management'),
    (r'\bmgr\b',         'manager'),
    (r'\bdept\b',        'department'),
    (r'\bsvcs\b',        'services'),
    (r'\bsvc\b',         'service'),
    (r'\bgrp\b',         'group'),
    (r'\bgroup\b',       'group'),
    (r'\bnational\b',    'national'),
    (r'\bnatl\b',        'national'),
    (r'&',               'and'),
    (r'\band\b',         'and'),
    (r'\bsarl\b',        'sarl'),    # French legal form
    (r'\bsas\b',         'sas'),
    (r'\bsasu\b',        'sasu'),
    (r'\bsa\b',          'sa'),
    (r'\beurl\b',        'eurl'),
    (r'\bsci\b',         'sci'),
    (r'\bsei\b',         'sei'),
]

_ADDR_ABBREV = [
    (r'\brd\b',     'road'),
    (r'\broad\b',   'road'),
    (r'\bst\b',     'street'),
    (r'\bstr\b',    'street'),
    (r'\bstreet\b', 'street'),
    (r'\bave\b',    'avenue'),
    (r'\bav\b',     'avenue'),
    (r'\bavenue\b', 'avenue'),
    (r'\bblvd\b',   'boulevard'),
    (r'\bboulevard\b', 'boulevard'),
    (r'\bdr\b',     'drive'),
    (r'\bdrive\b',  'drive'),
    (r'\bln\b',     'lane'),
    (r'\blane\b',   'lane'),
    (r'\bct\b',     'court'),
    (r'\bcourt\b',  'court'),
    (r'\bpl\b',     'place'),
    (r'\bplace\b',  'place'),
    (r'\bsq\b',     'square'),
    (r'\bsquare\b', 'square'),
    (r'\bpkwy\b',   'parkway'),
    (r'\bparkway\b','parkway'),
    (r'\bhwy\b',    'highway'),
    (r'\bhighway\b','highway'),
    (r'\bapt\b',    'apartment'),
    (r'\bapartment\b','apartment'),
    (r'\bste\b',    'suite'),
    (r'\bsuite\b',  'suite'),
    (r'\bflr\b',    'floor'),
    (r'\bfloor\b',  'floor'),
    (r'\bbldg\b',   'building'),
    (r'\bbuilding\b','building'),
    (r'&',          'and'),
    # Indian address abbreviations
    (r'\bh\.?no\b', 'house no'),
    (r'\bkh\.?no\b','kh no'),
    (r'\bnear\b',   'near'),
    (r'\bopp\b',    'opposite'),
    (r'\bopposite\b','opposite'),
    # French address abbreviations
    (r'\br\b',      'rue'),
    (r'\brue\b',    'rue'),
    (r'\ball\b',    'allee'),
    (r'\ballee\b',  'allee'),
    (r'\bimp\b',    'impasse'),
]

# Precompile
_NAME_ABBREV_RE = [(re.compile(p, re.IGNORECASE), r) for p, r in _NAME_ABBREV]
_ADDR_ABBREV_RE = [(re.compile(p, re.IGNORECASE), r) for p, r in _ADDR_ABBREV]

# Legal suffix patterns — used only for the flag, not for removal
_LEGAL_SUFFIXES = re.compile(
    r'\b(corporation|corp|incorporated|inc|limited|ltd|llc|llp|pvt|private|'
    r'company|co|associates|assoc|partnership|sarl|sas|sasu|sa|eurl|sci|sei|'
    r'pty|plc|gmbh|bv|nv|ag|ab|oy|as|kk|ou|sp\.?z\.?o\.?o)\b',
    re.IGNORECASE
)


def _unicode_normalize(text: str) -> str:
    """NFC normalize; keep all scripts (Devanagari, Latin, etc.)."""
    return unicodedata.normalize('NFC', text)


def _strip_punct_unicode_safe(text: str) -> str:
    """
    Strip punctuation while preserving ALL Unicode letters, marks, digits, spaces.
    Python's \\w only covers ASCII + underscore in char-class context; use
    unicodedata category instead to keep Devanagari combining marks (Mc/Mn).
    """
    keep_cats = {'Lu', 'Ll', 'Lt', 'Lm', 'Lo',   # letters
                 'Nd', 'Nl', 'No',                  # numbers
                 'Mc', 'Mn', 'Me',                  # combining marks
                 'Zs'}                              # spaces
    result = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in keep_cats or ch in ('-', "'"):
            result.append(ch)
        else:
            result.append(' ')
    return ''.join(result)


def normalize_name(text: str) -> tuple[str, bool]:
    """
    Normalize a business name.

    Returns:
        (normalized_text, has_legal_suffix)
    """
    if not text or (isinstance(text, float)):
        return '', False
    text = str(text)
    text = _unicode_normalize(text)

    # Detect legal suffix BEFORE stripping
    has_legal = bool(_LEGAL_SUFFIXES.search(text))

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Expand abbreviations
    for pattern, repl in _NAME_ABBREV_RE:
        text = pattern.sub(repl, text)

    # Strip punctuation — Unicode-safe (preserves Devanagari combining marks)
    text = _strip_punct_unicode_safe(text)
    # Remove leading/trailing hyphens (standalone)
    text = re.sub(r'(?<!\w)-|-(?!\w)', ' ', text)

    # Unify whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text, has_legal


def normalize_address(text: str) -> str:
    """
    Normalize a business address.

    Returns normalized text (no legal-suffix concept for addresses).
    """
    if not text or (isinstance(text, float)):
        return ''
    text = str(text)
    text = _unicode_normalize(text)
    text = text.lower()

    # Expand abbreviations
    for pattern, repl in _ADDR_ABBREV_RE:
        text = pattern.sub(repl, text)

    # Normalise separators (commas → spaces for tokenisation)
    text = re.sub(r'[,;|/\\]', ' ', text)

    # Strip remaining punctuation — Unicode-safe
    text = _strip_punct_unicode_safe(text)
    text = re.sub(r"(?<!\w)-|-(?!\w)", ' ', text)

    # Unify whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_numeric_tokens(text: str) -> set:
    """Extract all digit-containing tokens from a (pre-normalized) address."""
    return {t for t in text.split() if re.search(r'\d', t)}


if __name__ == '__main__':
    # Quick smoke test
    tests = [
        "Orelee's Barbershop Corp",
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "B+ Retail Inc",
        "Thermal & Fils SASU",
        "1795 Westchester Dr, High Point, NC",
        "H.NO 204 C ROAD HOSHIARPUR, PUNJAB",
        "175 Boulevard du Président Franklin Roosevelt",
    ]
    for t in tests:
        if 'Drive' in t or 'ROAD' in t or 'Boulevard' in t or 'H.NO' in t:
            print(f"  addr: {normalize_address(t)!r}")
        else:
            n, s = normalize_name(t)
            print(f"  name: {n!r}  suffix={s}")
