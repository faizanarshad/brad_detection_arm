#!/usr/bin/env python3
"""
Clean product data for ML: cleaned names, ADG codes, extracted brand, industry.
Reads brand_task.xlsx (default) or brand_task.csv; writes cleaned_product_data_with_brands.csv
"""
from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================================
# SIMPLE CLEANING FOR ML TRAINING - ONLY ESSENTIAL COLUMNS
# ============================================================================


def clean_armenian_text(text: object) -> str:
    """Clean Armenian product names."""
    if pd.isna(text) or text is None:
        return ""

    text = str(text)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", " ", text)

    text = text.replace("«", '"').replace("»", '"')
    text = text.replace("՛՛", '"').replace("՛", "'")
    text = text.replace("`", "'").replace("´", "'")

    text = re.sub(r"[,;:]+\s*", " ", text)

    text = re.sub(r"[^\w\s\u0530-\u058F\uFB00-\uFB06/-]", " ", text)

    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def clean_adg_code(code: object) -> int:
    """Preserve ADG codes exactly as numeric values in the sheet (same as Excel/training)."""
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


def extract_brand(text: object) -> str:
    """Extract brand from product name."""
    brands = {
        "artfood": "Artfood",
        "art food": "Artfood",
        "ատրֆուդ": "Artfood",
        "nescafe": "Nescafe",
        "նեսկաֆե": "Nescafe",
        "coca-cola": "Coca-Cola",
        "coca cola": "Coca-Cola",
        "կոկա-կոլա": "Coca-Cola",
        "կոկա կոլա": "Coca-Cola",
        "royal": "Royal",
        "ռոյալ": "Royal",
        "milka": "Milka",
        "միլկա": "Milka",
        "apache": "Apache",
        "ապաչի": "Apache",
        "marianna": "Marianna",
        "մարիաննա": "Marianna",
        "grand candy": "Grand Candy",
        "գրանդ քենդի": "Grand Candy",
        "yashkino": "Yashkino",
        "յաշկինո": "Yashkino",
        "daroink": "Daroink",
        "դարոինք": "Daroink",
        "դարոյնք": "Daroink",
        "roshen": "Roshen",
        "ռոշեն": "Roshen",
        "ritter sport": "Ritter Sport",
        "ganjasar": "Ganjasar",
        "գանձասար": "Ganjasar",
        "athenk": "Athenk",
        "athens": "Athens",
        "աթենք": "Athenk",
        "art armenia": "Art Armenia",
        "արտ արմենիա": "Art Armenia",
        "armenia": "Armenia",
        "nivea": "Nivea",
        "նիվեա": "Nivea",
        "colgate": "Colgate",
        "քոլգեյթ": "Colgate",
        "կոլգեյթ": "Colgate",
        "garnier": "Garnier",
        "գարնիեր": "Garnier",
        "gillette": "Gillette",
        "ժիլետ": "Gillette",
        "silky soft": "Silky Soft",
        "silk soft": "Silk Soft",
        "սիլկ սոֆթ": "Silk Soft",
        "salve": "Salve",
        "սալվե": "Salve",
        "savex": "Savax",
        "սավեքս": "Savax",
        "pampers": "Pampers",
        "պամպերս": "Pampers",
        "alex": "Alex",
        "ալեքս": "Alex",
        "finder": "Finder",
        "samsung": "Samsung",
        "սամսունգ": "Samsung",
        "mercedes-benz": "Mercedes-Benz",
        "mercedes benz": "Mercedes-Benz",
        "mercedes": "Mercedes-Benz",
        "zara": "Zara",
        "զառա": "Zara",
        "bmw": "BMW",
        "toyota": "Toyota",
        "nissan": "Nissan",
    }

    text_lower = str(text).lower()

    for brand_key, brand_name in brands.items():
        if brand_key in text_lower:
            return brand_name

    return "Unknown"


def classify_industry(text: object, brand: str) -> str:
    """Classify product into industry."""
    text_lower = str(text).lower()
    brand_lower = str(brand).lower()

    food_keywords = [
        "artfood",
        "nescafe",
        "coca-cola",
        "royal",
        "milka",
        "apache",
        "marianna",
        "grand candy",
        "yashkino",
        "daroink",
        "roshen",
        "ganjasar",
        "athenk",
        "կոմպոտ",
        "սուրճ",
        "գինի",
        "ըմպելիք",
        "պանիր",
        "երշիկ",
        "կոնֆետ",
        "շոկոլադ",
        "վաֆլի",
        "թխվածք",
        "կետչուպ",
        "պաղպաղակ",
        "յոգուրտ",
        "կաթ",
        "կարագ",
    ]

    for keyword in food_keywords:
        if keyword in text_lower or keyword in brand_lower:
            return "Food & Beverage"

    personal_care_keywords = [
        "nivea",
        "colgate",
        "garnier",
        "gillette",
        "silky soft",
        "salve",
        "ատամի մածուկ",
        "շամպուն",
        "ներկ",
        "կրեմ",
        "դեզոդորանտ",
        "սափրվելու",
        "լոգանքի",
    ]

    for keyword in personal_care_keywords:
        if keyword in text_lower or keyword in brand_lower:
            return "Personal Care & Cosmetics"

    household_keywords = [
        "savex",
        "pampers",
        "alex",
        "finder",
        "տակդիր",
        "անձեռոցիկ",
        "լվացքի",
        "փոշի",
        "մաքրող",
    ]

    for keyword in household_keywords:
        if keyword in text_lower or keyword in brand_lower:
            return "Household Products"

    electronics_keywords = [
        "samsung",
        "apple",
        "xiaomi",
        "հեռախոս",
        "հեռուստացույց",
        "մոնիտոր",
        "պլանշետ",
        "համակարգիչ",
    ]

    for keyword in electronics_keywords:
        if keyword in text_lower or keyword in brand_lower:
            return "Electronics"

    automotive_keywords = [
        "mercedes",
        "bmw",
        "toyota",
        "nissan",
        "ավտոմեքենա",
        "շարժիչ",
        "արգելակ",
    ]

    for keyword in automotive_keywords:
        if keyword in text_lower or keyword in brand_lower:
            return "Automotive"

    clothing_keywords = [
        "zara",
        "h&m",
        "adidas",
        "nike",
        "շապիկ",
        "գուլպա",
        "տաբատ",
        "կոշիկ",
        "լողազգեստ",
    ]

    for keyword in clothing_keywords:
        if keyword in text_lower or keyword in brand_lower:
            return "Clothing & Apparel"

    return "Other"


def _read_table(filepath: str | Path) -> pd.DataFrame:
    """Load Excel or CSV with encoding fallbacks for CSV."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)

    encodings = ["utf-8", "utf-8-sig", "cp1252", "iso-8859-1", "latin1"]
    last_err: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    return pd.read_csv(path)


def load_and_clean_data(filepath: str | Path = "brand_task.xlsx") -> pd.DataFrame | None:
    """Load and clean data; output essential columns only."""
    filepath = Path(filepath)
    print("=" * 60)
    print("CLEANING PRODUCT DATA FOR ML TRAINING")
    print("=" * 60)

    if not filepath.is_file():
        print(f"Error: File not found: {filepath}")
        print(f"Current directory: {os.getcwd()}")
        return None

    df = _read_table(filepath)
    used_enc = "excel" if filepath.suffix.lower() in (".xlsx", ".xls") else "csv"
    print(f"Loaded: {filepath} ({used_enc})")

    if "GOOD_NAME" not in df.columns or "ADG_CODE" not in df.columns:
        print("Error: need columns GOOD_NAME and ADG_CODE")
        return None

    print(f"Original dataset: {len(df)} rows")

    print("\nCleaning data...")

    cleaned_df = pd.DataFrame()
    print("  - Cleaning product names...")
    cleaned_df["cleaned_product_name"] = df["GOOD_NAME"].map(clean_armenian_text)

    print("  - Cleaning ADG codes...")
    cleaned_df["adg_code"] = df["ADG_CODE"].map(clean_adg_code)

    print("  - Extracting brands...")
    cleaned_df["brand"] = cleaned_df["cleaned_product_name"].map(extract_brand)

    print("  - Classifying industries...")
    cleaned_df["industry"] = cleaned_df.apply(
        lambda x: classify_industry(x["cleaned_product_name"], x["brand"]),
        axis=1,
    )

    empty_names = (cleaned_df["cleaned_product_name"] == "").sum()
    if empty_names > 0:
        print(f"  - Removing {empty_names} rows with empty product names")
        cleaned_df = cleaned_df[cleaned_df["cleaned_product_name"] != ""]

    original_count = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates(subset=["cleaned_product_name", "adg_code"])
    print(f"  - Removed {original_count - len(cleaned_df)} duplicate entries")

    cleaned_df = cleaned_df.sort_values("adg_code").reset_index(drop=True)

    valid_adg = (cleaned_df["adg_code"] != -1).sum()
    n = len(cleaned_df)
    print(f"\nFinal dataset: {n} rows")
    if n:
        print(f"Valid ADG codes: {valid_adg} ({100 * valid_adg / n:.1f}%)")
        print(f"Unique brands: {cleaned_df['brand'].nunique()}")
        print(f"Industries: {cleaned_df['industry'].nunique()}")

    return cleaned_df


def save_cleaned_data(df: pd.DataFrame, output_file: str | Path = "cleaned_product_data_with_brands.csv") -> Path:
    """Save cleaned data to CSV (UTF-8 BOM for Excel)."""
    output_file = Path(output_file)
    print("\n" + "=" * 60)
    print("SAVING CLEANED DATA")
    print("=" * 60)

    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    file_size = output_file.stat().st_size / 1024
    print(f"Saved to: {output_file}")
    print(f"File size: {file_size:.1f} KB")
    print(f"Columns: {', '.join(df.columns)}")

    return output_file


def display_sample(df: pd.DataFrame, n: int = 10) -> None:
    """Display sample of cleaned data."""
    print("\n" + "=" * 60)
    print(f"SAMPLE CLEANED DATA (first {n} rows)")
    print("=" * 60)

    for i in range(min(n, len(df))):
        row = df.iloc[i]
        prod = str(row["cleaned_product_name"])
        print(f"\nRow {i + 1}:")
        print(f"  ADG Code: {row['adg_code']}")
        print(f"  Brand: {row['brand']}")
        print(f"  Industry: {row['industry']}")
        print(f"  Product: {prod[:100]}")
        if len(prod) > 100:
            print("  ...")


def show_distributions(df: pd.DataFrame) -> None:
    """Show brand and industry distributions."""
    if len(df) == 0:
        return

    print("\n" + "=" * 60)
    print("BRAND DISTRIBUTION (top 10)")
    print("=" * 60)
    brand_counts = df["brand"].value_counts().head(10)
    for brand, count in brand_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {brand}: {count:,} ({percentage:.1f}%)")

    print("\n" + "=" * 60)
    print("INDUSTRY DISTRIBUTION")
    print("=" * 60)
    industry_counts = df["industry"].value_counts()
    for industry, count in industry_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {industry}: {count:,} ({percentage:.1f}%)")

    print("\n" + "=" * 60)
    print("ADG CODE DISTRIBUTION (top 10)")
    print("=" * 60)
    adg_counts = df[df["adg_code"] != -1]["adg_code"].value_counts().head(10)
    for code, count in adg_counts.items():
        print(f"  ADG {code}: {count:,} products")


def main() -> pd.DataFrame | None:
    project_dir = Path(__file__).resolve().parent
    default_xlsx = project_dir / "brand_task.xlsx"
    default_csv = project_dir / "brand_task.csv"
    input_path = default_xlsx if default_xlsx.is_file() else default_csv

    cleaned_df = load_and_clean_data(input_path)

    if cleaned_df is None:
        return None

    display_sample(cleaned_df)
    show_distributions(cleaned_df)

    output_file = save_cleaned_data(cleaned_df, project_dir / "cleaned_product_data_with_brands.csv")

    print("\n" + "=" * 60)
    print("Done. Output file created.")
    print("=" * 60)
    print(f"\nFile: {output_file}")
    print("\nColumns:")
    for col in cleaned_df.columns:
        print(f"  - {col}")
    print("\nUse cleaned_product_name as feature and adg_code as target for ML.")
    print("For train_bilstm.py, rename columns to GOOD_NAME and ADG_CODE or load this CSV in a custom loader.")

    return cleaned_df


if __name__ == "__main__":
    main()
