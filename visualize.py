"""
Plot training curves (loss, accuracy) and validation scores (accuracy, F1, confusion matrix).
Uses the same data split as train_bilstm.py (--seed, --val-size).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from train_bilstm import (
    BiLSTMClassifier,
    CharVocab,
    ProductDataset,
    evaluate as evaluate_val,
    load_labeled_data,
)


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pr = logits.argmax(dim=1).cpu().numpy()
        ys.append(yb.numpy())
        preds.append(pr)
    return np.concatenate(ys), np.concatenate(preds)


def load_model_from_checkpoint(ckpt: dict, device: torch.device) -> tuple[torch.nn.Module, dict]:
    meta = ckpt["meta"]
    char2idx = meta["char2idx"]
    vocab_size = max(char2idx.values()) + 1
    num_classes = len(meta["classes"])
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        emb_dim=meta["emb_dim"],
        hidden_dim=meta["hidden_dim"],
        num_layers=meta["num_layers"],
        dropout=meta["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, meta


def plot_training_history(history: dict, out_path: Path) -> None:
    epochs = history["epoch"]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    (ln1,) = ax1.plot(epochs, history["train_loss"], color="tab:blue", label="Train loss")
    (ln2,) = ax1.plot(epochs, history["val_loss"], color="tab:cyan", linestyle="--", label="Val loss")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Validation accuracy", color="tab:green")
    (ln3,) = ax2.plot(epochs, history["val_acc"], color="tab:green", linewidth=2, label="Val accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="tab:green")

    ax1.set_title("BiLSTM training — loss and validation accuracy")
    fig.legend(handles=[ln1, ln2, ln3], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score_bars(
    accuracy: float,
    f1_macro: float,
    f1_weighted: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["Accuracy", "F1 (macro)", "F1 (weighted)"]
    vals = [accuracy, f1_macro, f1_weighted]
    colors = ["#2ecc71", "#3498db", "#9b59b6"]
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Validation set — classification scores")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{v:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_topk(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    top_k: int,
    out_path: Path,
) -> None:
    """Confusion matrix restricted to the K most frequent classes in y_true."""
    uniq, counts = np.unique(y_true, return_counts=True)
    order = np.argsort(-counts)[:top_k]
    top_labels = uniq[order]
    label_to_pos = {int(l): i for i, l in enumerate(top_labels)}
    mask = np.isin(y_true, top_labels)
    yt = y_true[mask]
    yp = y_pred[mask]
    yt_m = np.array([label_to_pos[int(t)] for t in yt])
    yp_m = np.array([label_to_pos.get(int(p), -1) for p in yp])
    valid = yp_m >= 0
    yt_m, yp_m = yt_m[valid], yp_m[valid]
    cm = confusion_matrix(yt_m, yp_m, labels=np.arange(len(top_labels)))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums

    tick_labels = [str(class_names[i])[:12] for i in top_labels]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(top_labels)))
    ax.set_yticks(np.arange(len(top_labels)))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_ylabel("True ADG_CODE")
    ax.set_xlabel("Predicted ADG_CODE")
    ax.set_title(f"Normalized confusion (top {len(top_labels)} classes by frequency in val set)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize BiLSTM metrics")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "bilstm_adg.pt",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "brand_task.xlsx",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "figures")
    parser.add_argument("--top-classes", type=int, default=25, help="Confusion matrix: top-K frequent classes")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    history_path = args.checkpoint.parent / f"{args.checkpoint.stem}.history.json"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    history = ckpt.get("history")
    if history is None and history_path.is_file():
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)

    if history:
        plot_training_history(history, args.out_dir / "training_curves.png")
        print(f"Wrote {args.out_dir / 'training_curves.png'}")
    else:
        print("No training history in checkpoint; skipped training_curves.png (re-run train_bilstm.py to log history)")

    model, meta = load_model_from_checkpoint(ckpt, device)

    df, y, _ = load_labeled_data(args.data)
    texts = df["GOOD_NAME"].tolist()
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
    vocab = CharVocab(train_texts)
    max_len = meta["max_len"]
    val_ds = ProductDataset([texts[i] for i in idx_val], y[idx_val], vocab, max_len)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    criterion_eval = nn.CrossEntropyLoss()
    val_loss, acc, f1_mac, f1_w, topk_acc = evaluate_val(
        model, val_loader, criterion_eval, device
    )
    y_true, y_pred = predict_loader(model, val_loader, device)
    classes = meta["classes"]
    n_classes = len(classes)

    print(f"Validation samples: {len(y_true)}")
    print(f"Val loss (unweighted CE): {val_loss:.4f}")
    print(f"Accuracy:           {acc:.4f}")
    print(f"F1 (macro):         {f1_mac:.4f}")
    print(f"F1 (weighted):      {f1_w:.4f}")
    print(f"Top-3 accuracy:     {topk_acc[3]:.4f}")
    print(f"Top-5 accuracy:     {topk_acc[5]:.4f}")

    plot_score_bars(acc, f1_mac, f1_w, args.out_dir / "validation_scores.png")
    print(f"Wrote {args.out_dir / 'validation_scores.png'}")

    plot_confusion_topk(y_true, y_pred, classes, args.top_classes, args.out_dir / "confusion_topk.png")
    print(f"Wrote {args.out_dir / 'confusion_topk.png'}")

    summary = {
        "val_samples": int(len(y_true)),
        "val_loss": float(val_loss),
        "accuracy": float(acc),
        "f1_macro": float(f1_mac),
        "f1_weighted": float(f1_w),
        "top3_accuracy": float(topk_acc[3]),
        "top5_accuracy": float(topk_acc[5]),
        "num_classes": n_classes,
    }
    with open(args.out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
