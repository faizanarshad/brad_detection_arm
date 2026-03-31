"""
Predict ADG_CODE from product name (GOOD_NAME), or look up example product names for a given ADG_CODE
using brand_task.xlsx (reverse lookup; no model needed for --code).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from fasttext_embeddings import WordVocab
from text_language import normalize_good_name
from train_bilstm import BiLSTMClassifier, TOKENIZER_WORD_FASTTEXT


def _normalize_adg_key(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        return str(int(float(value)))
    except (ValueError, TypeError):
        return str(value).strip()


def _parse_keywords(keywords_csv: str | None) -> list[str]:
    if not keywords_csv or not keywords_csv.strip():
        return []
    return [k.strip().lower() for k in keywords_csv.split(",") if k.strip()]


def lookup_names_for_code(
    excel_path: Path,
    code_input: str,
    max_examples: int,
    min_count: int = 1,
    keywords_csv: str | None = None,
) -> tuple[str, list[tuple[str, int]], int, int]:
    """
    Return (normalized_code, pairs, n_rows_for_code, n_rows_after_keyword_filter).
    If no --keywords, n_rows_after_keyword_filter == n_rows_for_code (before min_count).
    """
    target = _normalize_adg_key(code_input.strip())
    if not target:
        return target, [], 0, 0

    df = pd.read_excel(excel_path)
    if "ADG_CODE" not in df.columns or "GOOD_NAME" not in df.columns:
        raise SystemExit(f"Expected columns GOOD_NAME, ADG_CODE in {excel_path}")

    df = df.dropna(subset=["GOOD_NAME", "ADG_CODE"]).copy()
    df["GOOD_NAME"] = df["GOOD_NAME"].astype(str).map(normalize_good_name)
    df = df[df["GOOD_NAME"].str.len() > 0]
    df["_code_key"] = df["ADG_CODE"].map(_normalize_adg_key)
    sub = df[df["_code_key"] == target]
    if sub.empty:
        return target, [], 0, 0
    n_rows = len(sub)

    kws = _parse_keywords(keywords_csv)
    if kws:

        def _matches(name: str) -> bool:
            low = name.lower()
            return any(kw in low for kw in kws)

        sub = sub[sub["GOOD_NAME"].map(_matches)]
    n_after_kw = len(sub)

    if sub.empty:
        return target, [], n_rows, n_after_kw

    counts = sub["GOOD_NAME"].value_counts()
    pairs: list[tuple[str, int]] = [
        (str(name), int(cnt)) for name, cnt in counts.items() if int(cnt) >= min_count
    ]
    return target, pairs[:max_examples], n_rows, n_after_kw


def encode_text(text: str, meta: dict) -> list[int]:
    max_len = meta["max_len"]
    if meta.get("tokenizer") == TOKENIZER_WORD_FASTTEXT:
        return WordVocab(word2idx=meta["word2idx"]).encode(text, max_len)
    char2idx = meta["char2idx"]
    pad = char2idx["<PAD>"]
    unk = char2idx["<UNK>"]
    text = normalize_good_name(text)
    ids = [char2idx.get(ch, unk) for ch in text[:max_len]]
    if len(ids) < max_len:
        ids.extend([pad] * (max_len - len(ids)))
    return ids


def load_model(ckpt_path: Path, device: torch.device) -> tuple[BiLSTMClassifier, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    tok = meta.get("tokenizer", "char")
    num_classes = len(meta["classes"])
    if tok == TOKENIZER_WORD_FASTTEXT:
        word2idx = meta["word2idx"]
        vocab_size = max(word2idx.values()) + 1
    else:
        char2idx = meta["char2idx"]
        vocab_size = max(char2idx.values()) + 1
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        emb_dim=meta["emb_dim"],
        hidden_dim=meta["hidden_dim"],
        num_layers=meta["num_layers"],
        dropout=meta["dropout"],
        embedding_weight=None,
        padding_idx=0,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, meta


@torch.no_grad()
def predict_one(
    model: BiLSTMClassifier,
    meta: dict,
    text: str,
    device: torch.device,
    topk: int = 1,
) -> list[tuple[str, float]]:
    classes = meta["classes"]
    x = torch.tensor([encode_text(text, meta)], dtype=torch.long, device=device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    k = min(topk, probs.size(1))
    values, indices = torch.topk(probs[0], k=k)
    out: list[tuple[str, float]] = []
    for i in range(k):
        code = classes[indices[i].item()]
        out.append((str(code), values[i].item()))
    return out


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Product name → ADG_CODE (model), or ADG_CODE → example product names (Excel lookup)"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_dir / "bilstm_adg.pt",
        help="Path to bilstm_adg.pt from train_bilstm.py (not used with --code)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=project_dir / "brand_task.xlsx",
        help="Training table for --code lookup (GOOD_NAME, ADG_CODE)",
    )
    parser.add_argument(
        "-c",
        "--code",
        type=str,
        default=None,
        metavar="ADG_CODE",
        dest="code",
        help="Code → products: list example GOOD_NAME rows for this ADG_CODE from --data (Excel). Same as: enter code, see product lines. No model.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=30,
        help="With --code: max GOOD_NAME lines to print (after filters)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        metavar="N",
        help="With --code: only names that appear at least N times. If every line is unique, "
        "--min-count 2 shows nothing — use 1 (default).",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default=None,
        metavar="K1,K2,...",
        help='With --code: comma-separated substrings; keep rows whose GOOD_NAME contains '
        'any of them (case-insensitive). Example for soft drinks: --keywords coca,cola,fanta,sprite,կոկա',
    )
    parser.add_argument("--topk", type=int, default=1, help="Return top-k ADG_CODE candidates (name→code mode)")
    parser.add_argument("--text", type=str, default=None, help="Single product name")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Text file: one GOOD_NAME per line (UTF-8)",
    )
    parser.add_argument(
        "text_positional",
        nargs="*",
        default=[],
        help="Optional product name as words (join with spaces)",
    )
    args = parser.parse_args()

    if args.code is not None:
        if not args.data.is_file():
            raise SystemExit(f"Data file not found: {args.data}")
        key, names, n_rows, n_after_kw = lookup_names_for_code(
            args.data,
            args.code,
            args.max_examples,
            min_count=args.min_count,
            keywords_csv=args.keywords,
        )
        print(f"ADG_CODE {key} — top product names by row count in {args.data.name}:")
        if args.keywords:
            print(f"  (filter: --keywords {args.keywords!r} — {n_after_kw} of {n_rows} rows match)")
        else:
            print("  (sorted by frequency; unrelated lines are often mislabeled — use --keywords to narrow)")
        if n_rows == 0:
            print(f"  (no rows found for code {key!r})")
        elif args.keywords and n_after_kw == 0:
            print(
                f"  (nothing matches your keywords among {n_rows} rows; try other substrings or drop --keywords)"
            )
        elif not names and args.min_count > 1:
            print(
                f"  (no product name appears ≥{args.min_count} times — all {n_rows} rows are "
                f"distinct lines. Use --min-count 1 or omit it.)"
            )
        elif not names:
            print(f"  (no rows after filters)")
        else:
            for n, cnt in names:
                if cnt > 1:
                    print(f"  {n}  [{cnt}×]")
                else:
                    print(f"  {n}")
        return

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(args.checkpoint, device)

    lines: list[str] = []
    if args.input is not None:
        lines = args.input.read_text(encoding="utf-8").splitlines()
    elif args.text is not None:
        lines = [args.text]
    elif args.text_positional:
        lines = [" ".join(args.text_positional)]
    else:
        parser.error("Provide --code ADG_CODE, or --text / --input / product name arguments")

    for line in lines:
        name = line.strip()
        if not name:
            continue
        preds = predict_one(model, meta, name, device, topk=args.topk)
        if args.topk <= 1:
            code, p = preds[0]
            print(f"{code}\t{p:.6f}\t{name}")
        else:
            print(name)
            for code, p in preds:
                print(f"  {code}\t{p:.6f}")


if __name__ == "__main__":
    main()
