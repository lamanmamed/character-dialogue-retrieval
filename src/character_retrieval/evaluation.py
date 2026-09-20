"""Ranking metrics and similarity analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CharacterResult:
    character: str
    rank: int
    self_similarity: float
    predicted_character: str
    predicted_similarity: float


@dataclass(frozen=True)
class EvaluationResult:
    mean_rank: float
    accuracy: float
    mean_self_similarity: float
    per_character: tuple[CharacterResult, ...]


def cosine_similarity_matrix(query_matrix, reference_matrix) -> np.ndarray:
    """Return query-by-reference cosine similarities.

    The project vectorizer L2-normalizes rows, so matrix multiplication gives
    cosine similarity directly. The fallback below also works for dense arrays.
    """
    if not hasattr(reference_matrix, "T"):
        reference_matrix = np.asarray(reference_matrix, dtype=float)
    if not hasattr(query_matrix, "T"):
        query_matrix = np.asarray(query_matrix, dtype=float)

    product = query_matrix @ reference_matrix.T
    if hasattr(product, "toarray"):
        product = product.toarray()
    return np.asarray(product, dtype=float)


def evaluate_rankings(
    reference_matrix,
    query_matrix,
    reference_labels: Sequence[str],
    query_labels: Sequence[str],
) -> EvaluationResult:
    """Evaluate where the correct character appears in each similarity ranking."""
    reference_labels = list(reference_labels)
    query_labels = list(query_labels)
    similarities = cosine_similarity_matrix(query_matrix, reference_matrix)

    results: list[CharacterResult] = []
    for row_index, character in enumerate(query_labels):
        if character not in reference_labels:
            raise ValueError(f"No training document exists for {character!r}.")

        row = similarities[row_index]
        order = np.argsort(row)[::-1]
        correct_index = reference_labels.index(character)
        rank = int(np.where(order == correct_index)[0][0]) + 1
        predicted_index = int(order[0])

        results.append(
            CharacterResult(
                character=character,
                rank=rank,
                self_similarity=float(row[correct_index]),
                predicted_character=reference_labels[predicted_index],
                predicted_similarity=float(row[predicted_index]),
            )
        )

    return EvaluationResult(
        mean_rank=float(np.mean([result.rank for result in results])),
        accuracy=float(np.mean([result.rank == 1 for result in results])),
        mean_self_similarity=float(np.mean([result.self_similarity for result in results])),
        per_character=tuple(results),
    )
