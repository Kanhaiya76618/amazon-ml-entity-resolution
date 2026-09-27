"""
normalize_v2.py — Enhanced text normalization for multilingual Entity Resolution.

Improvements:
- Country-aware abbreviation expansion (France: 'St' -> 'saint', US: 'St' -> 'street').
- NFKD accent folding (e.g. 'Payne Énterprises' -> 'payne enterprises').
- Transliterated name and address representations via translit.py.
- Distinctive numeric token extraction (e.g. house number, plot, flat, phone).
"""

from __future__ import annotations
import re
import unicodedata
from translit import fold_accents, romanize_indic, extract_acronym

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
    (r'\bsarl\b',        'sarl'),
    (r'\bsas\b',         'sas'),
    (r'\bsasu\b',        'sasu'),
    (r'\bsa\b',          'sa'),
    (r'\beurl\b',        'eurl'),
    (r'\bsci\b',         'sci'),
    (r'\bsei\b',         'sei'),
]

_ADDR_COMMON = [
    (r'\brd\b',     'road'),
    (r'\broad\b',   'road'),
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
    (r'\bh\.?no\b', 'house no'),
    (r'\bkh\.?no\b','kh no'),
    (r'\bnear\b',   'near'),
    (r'\bopp\b',    'opposite'),
    (r'\bopposite\b','opposite'),
    (r'\br\b',      'rue'),
    (r'\brue\b',    'rue'),
    (r'\ball\b',    'allee'),
    (r'\ballee\b',  'allee'),
    (r'\bimp\b',    'impasse'),
    (r'\bbd\b',     'boulevard'),
]

_NAME_ABBREV_RE = [(re.compile(p, re.IGNORECASE), r) for p, r in _NAME_ABBREV]
_ADDR_COMMON_RE = [(re.compile(p, re.IGNORECASE), r) for p, r in _ADDR_COMMON]
_ST_US_RE       = (re.compile(r'\bst\b', re.IGNORECASE), 'street')
_ST_FR_RE       = (re.compile(r'\bst\b', re.IGNORECASE), 'saint')

_LEGAL_SUFFIXES = re.compile(
    r'\b(corporation|corp|incorporated|inc|limited|ltd|llc|llp|pvt|private|'
    r'company|co|associates|assoc|partnership|sarl|sas|sasu|sa|eurl|sci|sei|'
    r'pty|plc|gmbh|bv|nv|ag|ab|oy|as|kk|ou|sp\.?z\.?o\.?o)\b',
    re.IGNORECASE
)

def clean_text(text: str) -> str:
    """Basic cleanup: accent fold, strip symbols, collapse spaces."""
    if not text or text == 'nan':
        return ""
    text = fold_accents(str(text))
    # Replace non-alphanumeric except spaces, hyphens, and apostrophes
    text = re.sub(r'[^a-zA-Z0-9\s\-\']', ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

def normalize_name_v2(name: str) -> tuple[str, str, bool]:
    """
    Returns:
        norm_name: cleaned, accent-folded, abbreviation-expanded name
        rom_name:  transliterated name (if Indic), else same as norm_name
        has_legal: bool flag indicating legal suffix presence
    """
    if not name or name == 'nan':
        return "", "", False
    raw = str(name).strip()
    has_legal = bool(_LEGAL_SUFFIXES.search(raw))
    
    # Romanize if Indic
    rom = romanize_indic(raw)
    
    # Clean and expand abbreviations
    norm = clean_text(raw)
    for pat, rep in _NAME_ABBREV_RE:
        norm = pat.sub(rep, norm)
    norm = re.sub(r'\s+', ' ', norm).strip()
    
    rom_norm = clean_text(rom)
    for pat, rep in _NAME_ABBREV_RE:
        rom_norm = pat.sub(rep, rom_norm)
    rom_norm = re.sub(r'\s+', ' ', rom_norm).strip()
    
    if not norm:
        norm = rom_norm
    
    return norm, rom_norm, has_legal

def normalize_address_v2(addr: str, country: str = 'US') -> str:
    """Normalize address with country-aware St expansion and number cleaning."""
    if not addr or addr == 'nan':
        return ""
    raw = str(addr).strip()
    raw = romanize_indic(raw)
    norm = clean_text(raw)
    
    # Country-aware St expansion
    c_lower = str(country).lower()
    if 'france' in c_lower:
        norm = _ST_FR_RE[0].sub(_ST_FR_RE[1], norm)
    else:
        norm = _ST_US_RE[0].sub(_ST_US_RE[1], norm)
        
    for pat, rep in _ADDR_COMMON_RE:
        norm = pat.sub(rep, norm)
        
    return re.sub(r'\s+', ' ', norm).strip()

def extract_primary_number(addr: str) -> str:
    """Extract primary numeric identifier from address (e.g. house/street number)."""
    nums = re.findall(r'\b\d+\b', str(addr))
    if nums:
        # Strip leading zeros
        n = nums[0].lstrip('0')
        return n if n else "0"
    return ""
