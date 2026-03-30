"""
Predict ADG_CODE using a fine-tuned XLM-RoBERTa checkpoint (train_xlm_roberta.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def id_to_code(model: AutoModelForSequenceClassification, idx: int) -> str:
    m = model.config.id2label
    if m is None:
        raise ValueError("Model config has no id2label")
    if isinstance(m, dict):
        return m.get(idx, m.get(str(idx)))
    return m[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict ADG_CODE with XLM-RoBERTa")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "xlm_roberta_adg",
    )
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--input", type=Path, default=None, help="UTF-8 file, one name per line")
    parser.add_argument("text_positional", nargs="*", default=[])
    args = parser.parse_args()

    if not (args.model_dir / "config.json").is_file():
        raise SystemExit(f"Missing Hugging Face model in {args.model_dir} (run train_xlm_roberta.py first)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    meta_path = args.model_dir / "training_meta.json"
    max_len = 256
    if meta_path.is_file():
        with open(meta_path, encoding="utf-8") as f:
            max_len = int(json.load(f).get("max_length", 256))

    if not model.config.id2label:
        raise SystemExit("Model config missing id2label")

    lines: list[str] = []
    if args.input is not None:
        lines = args.input.read_text(encoding="utf-8").splitlines()
    elif args.text is not None:
        lines = [args.text]
    elif args.text_positional:
        lines = [" ".join(args.text_positional)]
    else:
        parser.error("Provide --text, --input FILE, or a product name")

    for line in lines:
        name = line.strip()
        if not name:
            continue
        enc = tokenizer(
            name,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits[0]
            probs = torch.softmax(logits, dim=-1)
        k = min(args.topk, probs.numel())
        topv, topi = torch.topk(probs, k=k)
        if args.topk <= 1:
            idx = int(topi[0].item())
            code = id_to_code(model, idx)
            print(f"{code}\t{topv[0].item():.6f}\t{name}")
        else:
            print(name)
            for i in range(k):
                idx = int(topi[i].item())
                code = id_to_code(model, idx)
                print(f"  {code}\t{topv[i].item():.6f}")


if __name__ == "__main__":
    main()
