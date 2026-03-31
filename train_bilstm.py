"""
Train a BiLSTM on GOOD_NAME → ADG_CODE.

Default: character-level embeddings. Pass --fasttext-model PATH.cc.hy.300.bin for word-level FastText (optional).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from fasttext_embeddings import (
    WordVocab,
    build_fasttext_embedding_matrix,
    load_fasttext_bins,
)
from text_language import normalize_good_name

TEXT_NORMALIZE_VERSION = "nfc_lower_ws_slash_star_brand_v3"
TOKENIZER_CHAR = "char"
TOKENIZER_WORD_FASTTEXT = "word_fasttext"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CharVocab:
    PAD, UNK = 0, 1

    def __init__(self, texts: list[str]) -> None:
        chars = set()
        for t in texts:
            chars.update(t)
        # stable order for reproducibility
        sorted_chars = sorted(chars)
        self.idx2char = {self.PAD: "<PAD>", self.UNK: "<UNK>"}
        self.char2idx = {"<PAD>": self.PAD, "<UNK>": self.UNK}
        n = 2
        for c in sorted_chars:
            self.char2idx[c] = n
            self.idx2char[n] = c
            n += 1
        self.size = n

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.char2idx.get(ch, self.UNK) for ch in text[:max_len]]
        if len(ids) < max_len:
            ids.extend([self.PAD] * (max_len - len(ids)))
        return ids


class ProductDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, vocab: CharVocab, max_len: int) -> None:
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.vocab.encode(self.texts[i], self.max_len)
        return torch.tensor(x, dtype=torch.long), torch.tensor(self.labels[i], dtype=torch.long)


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        emb_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        embedding_weight: torch.Tensor | None = None,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        if embedding_weight is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_weight, freeze=False, padding_idx=padding_idx
            )
            emb_dim = embedding_weight.size(1)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.embedding(x))
        out, _ = self.lstm(emb)
        # Mask padded positions for mean pool
        mask = (x != self.padding_idx).float().unsqueeze(-1)
        summed = (out * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / lengths
        return self.fc(self.dropout(pooled))


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalized to mean 1 (helps imbalanced ADG_CODE)."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = 1.0 / counts
    w *= num_classes / w.sum()
    return torch.tensor(w, dtype=torch.float32)


def make_sample_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """Per-example weights for WeightedRandomSampler (balanced mini-batches)."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w_class = 1.0 / counts
    sw = w_class[y]
    return torch.from_numpy(sw).double()


class FocalLoss(nn.Module):
    """Cross-entropy modulated by (1-p)^gamma — often helps long-tail classes vs. plain CE."""

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def load_labeled_data(excel_path: Path) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    df = pd.read_excel(excel_path)
    df = df.dropna(subset=["GOOD_NAME", "ADG_CODE"]).copy()
    df["GOOD_NAME"] = df["GOOD_NAME"].astype(str).map(normalize_good_name)
    df = df[df["GOOD_NAME"].str.len() > 0]
    le = LabelEncoder()
    y = le.fit_transform(df["ADG_CODE"].astype(int).astype(str))
    return df, y, le


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion_eval: nn.Module,
    device: torch.device,
    topk: tuple[int, ...] = (3, 5),
) -> tuple[float, float, float, float, dict[int, float]]:
    """Unweighted val loss, accuracy, macro/weighted F1, top-k accuracies."""
    model.eval()
    total_loss = 0.0
    n = 0
    correct = 0
    topk_hits = {k: 0 for k in topk}
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion_eval(logits, yb)
        bs = xb.size(0)
        total_loss += loss.item() * bs
        n += bs
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        for k in topk:
            tk = logits.topk(min(k, logits.size(1)), dim=1)[1]
            topk_hits[k] += (tk == yb.unsqueeze(1)).any(dim=1).sum().item()
        ys.append(yb.cpu().numpy())
        preds.append(pred.cpu().numpy())
    y = np.concatenate(ys)
    pred = np.concatenate(preds)
    f1_macro = f1_score(y, pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y, pred, average="weighted", zero_division=0)
    val_loss = total_loss / n
    acc = correct / n
    topk_acc = {k: topk_hits[k] / n for k in topk}
    return val_loss, acc, f1_macro, f1_weighted, topk_acc


def metric_for_checkpoint(
    best_metric: str,
    val_acc: float,
    f1_macro: float,
    f1_weighted: float,
    top3_acc: float,
) -> float:
    if best_metric == "acc":
        return val_acc
    if best_metric == "macro_f1":
        return f1_macro
    if best_metric == "weighted_f1":
        return f1_weighted
    if best_metric == "top3":
        return top3_acc
    raise ValueError(best_metric)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="BiLSTM ADG_CODE classifier")
    parser.add_argument(
        "--data",
        type=Path,
        default=project_dir / "brand_task.xlsx",
        help="Path to brand_task.xlsx",
    )
    parser.add_argument(
        "--char",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fasttext-model",
        type=Path,
        action="append",
        dest="fasttext_models",
        default=None,
        help="If set: word BiLSTM + FastText .bin (pass twice to concat languages, e.g. hy+en). "
        "Default: character BiLSTM (no FastText).",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Max sequence length: characters (default) or words (with --fasttext-model). Defaults: 384 / 128.",
    )
    parser.add_argument("--emb-dim", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=project_dir / "bilstm_adg.pt")
    parser.add_argument(
        "--no-class-weight",
        action="store_true",
        help="Disable inverse-frequency class weights in the loss (not recommended for imbalanced codes)",
    )
    parser.add_argument(
        "--oversample",
        action="store_true",
        help="WeightedRandomSampler on training set (often use *without* class-weight to avoid double emphasis)",
    )
    parser.add_argument(
        "--no-lr-scheduler",
        action="store_true",
        help="Disable ReduceLROnPlateau on validation metric",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=0.0,
        help="If >0, use focal loss with this gamma (e.g. 2.0). 0 = standard cross-entropy. Often helps imbalanced ADG codes.",
    )
    parser.add_argument(
        "--best-metric",
        choices=("acc", "macro_f1", "weighted_f1", "top3"),
        default="top3",
        help="Metric for best checkpoint / LR schedule. top3 = recall@3 (often more informative than top-1 with 200+ classes).",
    )
    parser.add_argument(
        "--best-metric-legacy",
        action="store_true",
        help="Shortcut: set --best-metric weighted_f1 (previous default).",
    )
    parser.add_argument("--early-stopping", type=int, default=0, help="Stop if no improvement for N epochs (0=off)")
    args = parser.parse_args()

    if args.best_metric_legacy:
        args.best_metric = "weighted_f1"

    use_char = not args.fasttext_models
    ft_paths_resolved: list[Path] = []
    if not use_char:
        ft_paths_resolved = [p.expanduser().resolve() for p in args.fasttext_models]
    if use_char:
        max_len = args.max_len if args.max_len is not None else 384
        tokenizer = TOKENIZER_CHAR
    else:
        max_len = args.max_len if args.max_len is not None else 128
        tokenizer = TOKENIZER_WORD_FASTTEXT

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, y, label_encoder = load_labeled_data(args.data)
    texts = df["GOOD_NAME"].tolist()

    # Stratified split fails when any class has fewer than 2 samples; fall back to random split.
    try:
        idx_train, idx_val = train_test_split(
            np.arange(len(texts)),
            test_size=args.val_size,
            random_state=args.seed,
            stratify=y,
        )
    except ValueError:
        idx_train, idx_val = train_test_split(
            np.arange(len(texts)),
            test_size=args.val_size,
            random_state=args.seed,
        )
    train_texts = [texts[i] for i in idx_train]

    embedding_weight: torch.Tensor | None = None
    if use_char:
        vocab = CharVocab(train_texts)
        emb_dim_effective = args.emb_dim
    else:
        ft_models = load_fasttext_bins(ft_paths_resolved)
        vocab = WordVocab(train_texts)
        embedding_weight = build_fasttext_embedding_matrix(vocab.word2idx, ft_models)
        emb_dim_effective = embedding_weight.size(1)

    y_train = y[idx_train]
    train_ds = ProductDataset([texts[i] for i in idx_train], y_train, vocab, max_len)
    val_ds = ProductDataset([texts[i] for i in idx_val], y[idx_val], vocab, max_len)

    num_classes = len(label_encoder.classes_)
    use_class_weight = not args.no_class_weight
    if args.oversample:
        use_class_weight = False
    class_weights: torch.Tensor | None = None
    if use_class_weight:
        class_weights = compute_class_weights(y_train, num_classes).to(device)

    if args.focal_gamma and args.focal_gamma > 0:
        criterion_train = FocalLoss(weight=class_weights, gamma=args.focal_gamma).to(device)
    else:
        criterion_train = nn.CrossEntropyLoss(weight=class_weights)
    criterion_eval = nn.CrossEntropyLoss()

    if args.oversample:
        sw = make_sample_weights(y_train, num_classes)
        sampler = WeightedRandomSampler(sw, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = BiLSTMClassifier(
        vocab_size=vocab.size,
        num_classes=num_classes,
        emb_dim=args.emb_dim if use_char else emb_dim_effective,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        embedding_weight=embedding_weight,
        padding_idx=WordVocab.PAD if not use_char else CharVocab.PAD,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if not args.no_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
        )

    best_score = -1.0
    best_state = None
    stale = 0
    history: dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1_macro": [],
        "val_f1_weighted": [],
        "val_top3": [],
        "val_top5": [],
    }
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion_train, device)
        val_loss, val_acc, f1_macro, f1_weighted, topk_acc = evaluate(
            model, val_loader, criterion_eval, device
        )
        score = metric_for_checkpoint(
            args.best_metric, val_acc, f1_macro, f1_weighted, topk_acc[3]
        )
        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1_macro"].append(f1_macro)
        history["val_f1_weighted"].append(f1_weighted)
        history["val_top3"].append(topk_acc[3])
        history["val_top5"].append(topk_acc[5])

        if scheduler is not None:
            scheduler.step(score)

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        lr_cur = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:02d}  lr={lr_cur:.2e}  train_loss={tr_loss:.4f}  "
            f"val_loss={val_loss:.4f}  acc={val_acc:.4f}  "
            f"f1_macro={f1_macro:.4f}  f1_w={f1_weighted:.4f}  "
            f"top3={topk_acc[3]:.4f}  top5={topk_acc[5]:.4f}",
            flush=True,
        )

        if args.early_stopping > 0 and stale >= args.early_stopping:
            print(
                f"Early stopping at epoch {epoch} (no improvement {stale} epochs on {args.best_metric})",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    meta: dict = {
        "tokenizer": tokenizer,
        "max_len": max_len,
        "emb_dim": emb_dim_effective if not use_char else args.emb_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.layers,
        "dropout": args.dropout,
        "classes": label_encoder.classes_.tolist(),
        "text_normalize": TEXT_NORMALIZE_VERSION,
    }
    if use_char:
        meta["char2idx"] = vocab.char2idx
    else:
        meta["word2idx"] = vocab.word2idx
        meta["fasttext_model_paths"] = [str(p) for p in ft_paths_resolved]
    history_path = args.out.parent / f"{args.out.stem}.history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    torch.save(
        {
            "model_state": model.state_dict(),
            "meta": meta,
            "history": history,
        },
        args.out,
    )
    with open(args.out.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved checkpoint to {args.out} (best {args.best_metric}={best_score:.4f})", flush=True)


if __name__ == "__main__":
    main()
