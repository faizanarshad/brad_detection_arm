"""
Language / script-aware text preparation for BiLSTM (Armenian + Latin brand names).

Does not replace the model: it normalizes inputs so training and inference see consistent Unicode.
"""
from __future__ import annotations

import re
import unicodedata

# Armenian Unicode blocks (simplified letter detection for diagnostics)
_ARM_MIN, _ARM_MAX = 0x0530, 0x058F
_ARM_SUP_MIN, _ARM_SUP_MAX = 0xFB10, 0xFB17

# Applied after base cleaning (lowercase, spacing, etc.). Regex → canonical token (underscore).
# Order: longer / more specific patterns first. Extend here as needed.
BRAND_CANONICAL_REPLACEMENTS: list[tuple[str, str]] = [
    (r"\bcoca\s+cola\b", "coca_cola"),
    (r"\bկոկա\s+կոլա\b", "coca_cola"),
    (r"\bfanta\b", "fanta"),
    (r"\bsprite\b", "sprite"),
]


def canonicalize_brands(text: str) -> str:
    """Map variant spellings to single tokens (e.g. coca cola → coca_cola). Input must already be lowercased."""
    s = text
    for pattern, repl in BRAND_CANONICAL_REPLACEMENTS:
        s = re.sub(pattern, repl, s, flags=re.UNICODE)
    return s


def normalize_good_name(text: str) -> str:
    """
    Mandatory text cleaning before BiLSTM (training and inference must match).

    - Unicode NFC; strip zero-width / BOM noise
    - Remove Excel escape literals: _x000D_, _x000A_ (case-insensitive)
    - Remove asterisks *
    - Replace slash runs (/, //, ///, …) with a space
    - Replace hyphens with a space (e.g. Coca-Cola → coca cola)
    - Lowercase (Unicode-aware for Armenian + Latin)
    - Collapse whitespace
    - Canonical brand tokens (see BRAND_CANONICAL_REPLACEMENTS), e.g. coca cola → coca_cola

    Example: "Coca-Cola 0.25լ///" → "coca_cola 0.25լ"
    """
    if not text or not isinstance(text, str):
        return ""
    s = unicodedata.normalize("NFC", text)
    for z in ("\u200b", "\u200c", "\u200d", "\ufeff", "\u00ad"):
        s = s.replace(z, "")
    s = re.sub(r"_x000D_", " ", s, flags=re.I)
    s = re.sub(r"_x000A_", " ", s, flags=re.I)
    s = s.replace("\u00b6", " ")
    s = s.replace("*", "")
    s = re.sub(r"/+", " ", s)
    s = s.replace("-", " ")
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = canonicalize_brands(s)
    return s


def _is_armenian_letter(ch: str) -> bool:
    if len(ch) != 1:
        return False
    o = ord(ch)
    return _ARM_MIN <= o <= _ARM_MAX or _ARM_SUP_MIN <= o <= _ARM_SUP_MAX


def _is_latin_letter(ch: str) -> bool:
    if len(ch) != 1:
        return False
    return ("A" <= ch <= "Z") or ("a" <= ch <= "z")


def script_mix(text: str) -> dict[str, float]:
    """
    Rough letter fractions for logging / QA (not used by the model).
    Returns armenian, latin, other in [0,1] over alphabetic characters only.
    """
    hy = la = ot = 0
    for ch in text:
        if ch.isspace() or ch.isdigit():
            continue
        cat = unicodedata.category(ch)
        if cat not in ("Lu", "Ll", "Lt", "Lm", "Lo"):
            continue
        if _is_armenian_letter(ch):
            hy += 1
        elif _is_latin_letter(ch):
            la += 1
        else:
            ot += 1
    total = hy + la + ot
    if total == 0:
        return {"armenian": 0.0, "latin": 0.0, "other": 0.0}
    return {
        "armenian": hy / total,
        "latin": la / total,
        "other": ot / total,
    }
