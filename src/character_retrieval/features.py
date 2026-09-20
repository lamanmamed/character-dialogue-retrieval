"""TF-IDF, optional character n-grams, style features, and SVD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize


@dataclass(frozen=True)
class FeatureConfig:
    word_ngram_range: tuple[int, int] = (1, 2)
    min_df: int | float = 2
    max_df: int | float = 1.0
    sublinear_tf: bool = True
    use_character_ngrams: bool = False
    character_ngram_range: tuple[int, int] = (3, 5)
    include_style: bool = True
    style_weight: float = 0.1
    use_svd: bool = False
    svd_components: int = 200
    l2_normalize: bool = True


def style_features(token_lists: Sequence[Sequence[str]]) -> np.ndarray:
    """Compute the five small style measurements used in the notebook.

    The features are average token length, type-token ratio, digit ratio,
    short-token ratio, and long-token ratio.
    """
    rows: list[list[float]] = []
    for tokens in token_lists:
        token_count = max(len(tokens), 1)
        average_length = float(np.mean([len(token) for token in tokens])) if tokens else 0.0
        type_token_ratio = len(set(tokens)) / token_count
        digit_ratio = sum(
            any(char.isdigit() for char in token) or token == "<NUM>"
            for token in tokens
        ) / token_count
        short_ratio = sum(len(token) <= 3 for token in tokens) / token_count
        long_ratio = sum(len(token) >= 8 for token in tokens) / token_count
        rows.append([
            average_length,
            type_token_ratio,
            digit_ratio,
            short_ratio,
            long_ratio,
        ])
    return np.asarray(rows, dtype=float)


class DialogueVectorizer:
    """Fit and apply the text representation used for character retrieval."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self.word_vectorizer: TfidfVectorizer | None = None
        self.character_vectorizer: TfidfVectorizer | None = None
        self.style_scaler: StandardScaler | None = None
        self.svd: TruncatedSVD | None = None

    @staticmethod
    def _texts(token_lists: Sequence[Sequence[str]]) -> list[str]:
        return [" ".join(tokens) for tokens in token_lists]

    def fit_transform(self, token_lists: Sequence[Sequence[str]]):
        texts = self._texts(token_lists)
        cfg = self.config

        self.word_vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            token_pattern=None,
            preprocessor=None,
            lowercase=False,
            min_df=cfg.min_df,
            max_df=cfg.max_df,
            ngram_range=cfg.word_ngram_range,
            sublinear_tf=cfg.sublinear_tf,
            norm=None,
        )
        word_matrix = self.word_vectorizer.fit_transform(texts)
        parts = [word_matrix]

        if cfg.use_character_ngrams:
            self.character_vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=cfg.character_ngram_range,
                min_df=1,
                max_df=1.0,
                sublinear_tf=cfg.sublinear_tf,
                norm=None,
            )
            parts.append(self.character_vectorizer.fit_transform(texts))

        if cfg.include_style:
            self.style_scaler = StandardScaler()
            scaled = self.style_scaler.fit_transform(style_features(token_lists))
            parts.append(csr_matrix(scaled * cfg.style_weight))

        matrix = parts[0] if len(parts) == 1 else hstack(parts).tocsr()

        if cfg.use_svd:
            self.svd = TruncatedSVD(
                n_components=cfg.svd_components,
                random_state=42,
            )
            matrix = self.svd.fit_transform(matrix)

        if cfg.l2_normalize:
            matrix = normalize(matrix, norm="l2", copy=False)
        return matrix

    def transform(self, token_lists: Sequence[Sequence[str]]):
        if self.word_vectorizer is None:
            raise RuntimeError("DialogueVectorizer must be fitted before transform().")

        texts = self._texts(token_lists)
        cfg = self.config
        parts = [self.word_vectorizer.transform(texts)]

        if cfg.use_character_ngrams:
            if self.character_vectorizer is None:
                raise RuntimeError("Character vectorizer was not fitted.")
            parts.append(self.character_vectorizer.transform(texts))

        if cfg.include_style:
            if self.style_scaler is None:
                raise RuntimeError("Style scaler was not fitted.")
            scaled = self.style_scaler.transform(style_features(token_lists))
            parts.append(csr_matrix(scaled * cfg.style_weight))

        matrix = parts[0] if len(parts) == 1 else hstack(parts).tocsr()

        if cfg.use_svd:
            if self.svd is None:
                raise RuntimeError("SVD model was not fitted.")
            matrix = self.svd.transform(matrix)

        if cfg.l2_normalize:
            matrix = normalize(matrix, norm="l2", copy=False)
        return matrix
