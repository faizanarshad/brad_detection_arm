"""
Predict ADG_CODE from GOOD_NAME using a trained BiLSTM checkpoint (train_bilstm.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fasttext_embeddings import WordVocab
from text_language import normalize_good_name
from train_bilstm import BiLSTMClassifier, TOKENIZER_WORD_FASTTEXT


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
    parser = argparse.ArgumentParser(description="Predict ADG_CODE from product name")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "bilstm_adg.pt",
        help="Path to bilstm_adg.pt from train_bilstm.py",
    )
    parser.add_argument("--topk", type=int, default=1, help="Return top-k ADG_CODE candidates")
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
        parser.error("Provide --text, --input FILE, or a product name as arguments")

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
