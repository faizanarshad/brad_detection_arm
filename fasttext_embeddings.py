"""
FastText (.bin) helpers: word tokenization and pretrained embedding matrices for BiLSTM.

Uses official `fasttext` wheels; .bin models support subword vectors for OOV-ish tokens.
For multilingual, pass multiple models (e.g. cc.hy.300.bin + cc.en.300.bin) — vectors are concatenated.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from text_language import normalize_good_name

if TYPE_CHECKING:
    import fasttext as _fasttext  # noqa: F401


def tokenize_words(text: str) -> list[str]:
    """Whitespace tokens after the same normalization as the char model."""
    s = normalize_good_name(text)
    if not s:
        return []
    return re.findall(r"\S+", s)


class WordVocab:
    PAD, UNK = 0, 1

    def __init__(self, texts: list[str] | None = None, word2idx: dict[str, int] | None = None) -> None:
        if word2idx is not None:
            self.word2idx = dict(word2idx)
        elif texts is not None:
            self.word2idx = self._from_texts(texts)
        else:
            raise ValueError("Provide texts or word2idx")
        self.size = len(self.word2idx)

    def _from_texts(self, texts: list[str]) -> dict[str, int]:
        words: set[str] = set()
        for t in texts:
            words.update(tokenize_words(t))
        sorted_words = sorted(words)
        word2idx: dict[str, int] = {"<PAD>": self.PAD, "<UNK>": self.UNK}
        n = 2
        for w in sorted_words:
            word2idx[w] = n
            n += 1
        return word2idx

    def encode(self, text: str, max_len: int) -> list[int]:
        pad = self.PAD
        unk = self.UNK
        words = tokenize_words(text)
        ids = [self.word2idx.get(w, unk) for w in words[:max_len]]
        if len(ids) < max_len:
            ids.extend([pad] * (max_len - len(ids)))
        return ids


def load_fasttext_bins(paths: list[Path]) -> list:
    import fasttext

    out: list = []
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(
                f"FastText model not found: {p}\n"
                "Download e.g. cc.hy.300.bin from "
                "https://fasttext.cc/docs/en/crawl-vectors.html (extract .bin next to .gz)"
            )
        out.append(fasttext.load_model(str(p)))
    return out


def _vector_dim(ft) -> int:
    return int(ft.get_word_vector("a").shape[0])


def build_fasttext_embedding_matrix(
    word2idx: dict[str, int],
    ft_models: list,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """
    Rows align with word2idx indices. PAD=0 row is zeros; UNK=1 row is small random.
    Each word row is concatenation of get_word_vector from each model (multilingual).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(word2idx)
    dims = [_vector_dim(m) for m in ft_models]
    emb_dim = sum(dims)
    mat = np.zeros((n, emb_dim), dtype=np.float32)
    mat[WordVocab.UNK] = rng.standard_normal(emb_dim).astype(np.float32) * 0.01
    words = [w for w, i in word2idx.items() if i >= 2]
    for w in words:
        idx = word2idx[w]
        parts = []
        for m in ft_models:
            parts.append(m.get_word_vector(w))
        mat[idx] = np.concatenate(parts).astype(np.float32)
    return torch.from_numpy(mat)
