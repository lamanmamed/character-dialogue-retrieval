"""Character identification from dialogue with vector-space retrieval."""

from .features import FeatureConfig, DialogueVectorizer
from .preprocessing import PreprocessingConfig, preprocess_text
from .retrieval import CharacterDialogueRetriever

__all__ = [
    "CharacterDialogueRetriever",
    "DialogueVectorizer",
    "FeatureConfig",
    "PreprocessingConfig",
    "preprocess_text",
]
