#!/usr/bin/env python3
"""
Simple cleaning for ML training — GOOD_NAME, ADG_CODE, plus optional BRAND and INDUSTRY.

Apply to brand_task.xlsx and write a cleaned copy (default: brand_task_cleaned.xlsx).
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# Allowed after stripping junk: word chars, space, Armenian blocks, presentation forms,
# slash, hyphen, plus decimal/comma/percent for quantities (e.g. 3.2%, 1,5լ).
_ALLOWED_GOOD_NAME = re.compile(
    r"[^\w\s\u0530-\u058F\uFB00-\uFB06/.\-,%]",
    re.UNICODE,
)

# Canonical column names (case-insensitive match on input)
_COLUMN_ALIASES: dict[str, str] = {
    "good_name": "GOOD_NAME",
    "adg_code": "ADG_CODE",
    "brand": "BRAND",
    "industry": "INDUSTRY",
}


def _rename_columns_case_insensitive(df: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for c in df.columns:
        key = str(c).strip().lower()
        if key in _COLUMN_ALIASES:
            rename[c] = _COLUMN_ALIASES[key]
    return df.rename(columns=rename)


def canonicalize_excel_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known headers and ensure BRAND / INDUSTRY exist (empty if missing)."""
    df = _rename_columns_case_insensitive(df.copy())
    for col in ("BRAND", "INDUSTRY"):
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")
    return df


def clean_armenian_text(text: object) -> str:
    """Clean product name / brand / industry text for training."""
    if pd.isna(text) or text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", " ", text)
    for q in (
        "\u00ab",
        "\u00bb",
        "\u2018",
        "\u2019",
        "\u00b4",
        "`",
    ):
        text = text.replace(q, "")
    text = re.sub(r"[,;:]+\s*", " ", text)
    text = _ALLOWED_GOOD_NAME.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_adg_code(code: object) -> int:
    """Preserve ADG as in source (int of numeric cell); invalid / missing → -1."""
    if pd.isna(code) or code is None:
        return -1
    if isinstance(code, str) and code.strip() == "":
        return -1
    if isinstance(code, bool):
        return -1
    try:
        return int(float(str(code).strip()))
    except (ValueError, TypeError, OverflowError):
        return -1


def _column_order(df: pd.DataFrame) -> list[str]:
    """GOOD_NAME, BRAND, INDUSTRY, ADG_CODE first; then any other columns."""
    preferred = ["GOOD_NAME", "BRAND", "INDUSTRY", "ADG_CODE"]
    first = [c for c in preferred if c in df.columns]
    rest = [c for c in df.columns if c not in first]
    return first + rest


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean brand_task.xlsx for ML training")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "brand_task.xlsx",
        help="Source Excel path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "brand_task_cleaned.xlsx",
        help="Where to write cleaned data",
    )
    parser.add_argument(
        "--drop-invalid-code",
        action="store_true",
        help="Drop rows where ADG_CODE is missing or invalid after cleaning",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    df = pd.read_excel(args.input)
    df = _rename_columns_case_insensitive(df)

    if "GOOD_NAME" not in df.columns or "ADG_CODE" not in df.columns:
        raise SystemExit("Expected columns: GOOD_NAME, ADG_CODE (and optional BRAND, INDUSTRY)")

    for col in ("BRAND", "INDUSTRY"):
        if col not in df.columns:
            df[col] = ""

    n0 = len(df)
    df = df.copy()
    df["GOOD_NAME"] = df["GOOD_NAME"].map(clean_armenian_text)
    df["BRAND"] = df["BRAND"].map(clean_armenian_text)
    df["INDUSTRY"] = df["INDUSTRY"].map(clean_armenian_text)
    df["ADG_CODE"] = df["ADG_CODE"].map(clean_adg_code)

    df = df[df["GOOD_NAME"].str.len() > 0]
    n_empty_name = n0 - len(df)
    if n_empty_name:
        print(f"Dropped {n_empty_name} rows with empty GOOD_NAME after cleaning")

    if args.drop_invalid_code:
        before = len(df)
        df = df[df["ADG_CODE"] >= 0]
        print(f"Dropped {before - len(df)} rows with invalid ADG_CODE")

    for col in ("BRAND", "INDUSTRY"):
        df[col] = df[col].fillna("").astype(str).replace("nan", "")

    df = df[_column_order(df)]
    df.to_excel(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output} (from {n0} input rows)")
    print(f"Columns: {', '.join(df.columns.tolist())}")


if __name__ == "__main__":
    main()
