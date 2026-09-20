"""End-to-end character ranking from dialogue documents."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from .evaluation import EvaluationResult, cosine_similarity_matrix, evaluate_rankings
from .features import DialogueVectorizer, FeatureConfig
from .preprocessing import PreprocessingConfig, preprocess_text


class CharacterDialogueRetriever:
    """Represent each character's dialogue and rank new dialogue by similarity."""

    def __init__(
        self,
        *,
        preprocessing: PreprocessingConfig | None = None,
        features: FeatureConfig | None = None,
        preprocessor: Callable[[str], list[str]] | None = None,
    ) -> None:
        self.preprocessing = preprocessing or PreprocessingConfig()
        self.vectorizer = DialogueVectorizer(features)
        self._preprocessor = preprocessor
        self.labels: list[str] = []
        self.reference_matrix = None

    def _process(self, text: str) -> list[str]:
        if self._preprocessor is not None:
            return self._preprocessor(text)
        return preprocess_text(text, self.preprocessing)

    def fit(self, documents: Mapping[str, str]) -> "CharacterDialogueRetriever":
        self.labels = sorted(documents)
        tokens = [self._process(documents[label]) for label in self.labels]
        self.reference_matrix = self.vectorizer.fit_transform(tokens)
        return self

    def rank(self, text: str) -> list[tuple[str, float]]:
        if self.reference_matrix is None:
            raise RuntimeError("Call fit() before rank().")
        query = self.vectorizer.transform([self._process(text)])
        similarities = cosine_similarity_matrix(query, self.reference_matrix)[0]
        order = similarities.argsort()[::-1]
        return [(self.labels[i], float(similarities[i])) for i in order]

    def evaluate(self, documents: Mapping[str, str]) -> EvaluationResult:
        if self.reference_matrix is None:
            raise RuntimeError("Call fit() before evaluate().")
        query_labels = sorted(documents)
        tokens = [self._process(documents[label]) for label in query_labels]
        query_matrix = self.vectorizer.transform(tokens)
        return evaluate_rankings(
            self.reference_matrix,
            query_matrix,
            self.labels,
            query_labels,
        )
