"""
translit.py — Fast, deterministic Indic script transliteration and Unicode accent folding.

Supports:
- Accent folding (NFKD: é -> e, etc.)
- Brahmic Indic scripts (Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam, Gurmukhi, Oriya)
- Acronym extraction
- Vowel skeleton and Soundex
"""

import unicodedata
import re

DEV_CONS = {
    'क':'k','ख':'kh','ग':'g','घ':'gh','ङ':'n',
    'च':'ch','छ':'chh','ज':'j','झ':'jh','ञ':'n',
    'ट':'t','ठ':'th','ड':'d','ढ':'dh','ण':'n',
    'त':'t','थ':'th','द':'d','ध':'dh','न':'n',
    'प':'p','फ':'ph','ब':'b','भ':'bh','म':'m',
    'य':'y','र':'r','ल':'l','व':'v','श':'sh','ष':'sh','स':'s','ह':'h','ळ':'l',
    'क़':'k','ख़':'kh','ग़':'g','ज़':'z','ड़':'r','ढ़':'rh','फ़':'f','य़':'y',
    'ऩ':'n','ऱ':'r','ऴ':'l'
}
DEV_MATRA = {
    'ा':'a','ि':'i','ी':'i','ु':'u','ू':'u','ृ':'ri',
    'े':'e','ै':'ai','ो':'o','ौ':'au','ॅ':'e','ॉ':'o',
    'ॆ':'e','ॊ':'o'
}
DEV_INDEP = {
    'अ':'a','आ':'a','इ':'i','ई':'i','उ':'u','ऊ':'u',
    'ए':'e','ऐ':'ai','ओ':'o','औ':'au','ऋ':'ri',
    'ऎ':'e','ऒ':'o','ॲ':'a','ऑ':'o'
}
DEV_MISC = {'ं':'n','ँ':'n','ः':'h','्':''}

def fold_accents(text: str) -> str:
    """Normalize accents (NFKD) and remove non-spacing marks (e.g. é -> e)."""
    if not text:
        return ""
    nfkd = unicodedata.normalize('NFKD', str(text))
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def indic_to_devanagari(text: str) -> str:
    """Map any Brahmic Indic Unicode block to standard Devanagari by relative code point offset."""
    res = []
    for ch in text:
        cp = ord(ch)
        for base in [0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]:
            if base <= cp < base + 0x80:
                offset = cp - base
                res.append(chr(0x0900 + offset))
                break
        else:
            res.append(ch)
    return ''.join(res)

def romanize_indic(s: str) -> str:
    """Romanize text if it contains Brahmic Indic characters."""
    # Quick check if text contains characters in Indic Unicode range (0x0900 - 0x0D7F)
    has_indic = any(0x0900 <= ord(c) <= 0x0D7F for c in s)
    if not has_indic:
        return fold_accents(s).lower()
        
    s = indic_to_devanagari(s)
    s = unicodedata.normalize('NFC', s)
    out, i, n = [], 0, len(s)
    while i < n:
        ch = s[i]
        if ch in DEV_CONS:
            j = i + 1
            if j < n and s[j] in DEV_MATRA:
                out.append(DEV_CONS[ch] + DEV_MATRA[s[j]])
                i = j + 1
            elif j < n and s[j] == '्':
                out.append(DEV_CONS[ch])
                i = j + 1
            else:
                out.append(DEV_CONS[ch] + 'a')
                i += 1
        elif ch in DEV_INDEP:
            out.append(DEV_INDEP[ch])
            i += 1
        elif ch in DEV_MATRA:
            out.append(DEV_MATRA[ch])
            i += 1
        elif ch in DEV_MISC:
            out.append(DEV_MISC[ch])
            i += 1
        else:
            out.append(ch)
            i += 1
    res = ''.join(out)
    res = re.sub(r'a+', 'a', res)
    return fold_accents(res).lower()

def extract_acronym(text: str) -> str:
    """Generate acronym from first letters of multi-word business names (len 2-6)."""
    if not text or str(text) == 'nan':
        return ""
    words = [w for w in re.findall(r'[a-zA-Z0-9]+', str(text)) if w.lower() not in {'and', 'of', 'the', 'for', 'at', 'by', 'in', 'llc', 'inc', 'ltd'}]
    if 2 <= len(words) <= 6:
        return ''.join(w[0] for w in words).lower()
    return ""

def vowel_skeleton(word: str) -> str:
    """Vowel-insensitive key: keep first letter, drop inner vowels."""
    if not word or str(word) == 'nan':
        return ""
    w = fold_accents(str(word)).lower()
    w = re.sub(r'[^a-z0-9]', '', w)
    if not w:
        return ""
    return w[0] + ''.join(c for c in w[1:] if c not in 'aeiou')
