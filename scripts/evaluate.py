"""Evaluate the final retrieval configuration on local dialogue CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from character_retrieval import CharacterDialogueRetriever
from character_retrieval.data import build_character_documents, load_dialogue_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_csv", type=Path)
    parser.add_argument("heldout_csv", type=Path)
    parser.add_argument("--training-lines", type=int, default=300)
    parser.add_argument("--heldout-lines", type=int, default=50)
    args = parser.parse_args()

    training = build_character_documents(
        load_dialogue_csv(args.training_csv),
        args.training_lines,
    )
    heldout = build_character_documents(
        load_dialogue_csv(args.heldout_csv),
        args.heldout_lines,
    )

    retriever = CharacterDialogueRetriever().fit(training)
    result = retriever.evaluate(heldout)

    print(f"Mean rank: {result.mean_rank:.4f}")
    print(f"Top-1 accuracy: {result.accuracy:.4f}")
    print(f"Mean self-similarity: {result.mean_self_similarity:.4f}")
    print()
    for row in result.per_character:
        print(
            f"{row.character:10s} rank={row.rank:<2d} "
            f"predicted={row.predicted_character:10s} "
            f"self_similarity={row.self_similarity:.4f}"
        )


if __name__ == "__main__":
    main()
