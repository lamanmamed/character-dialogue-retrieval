"""Preprocessing used before building character dialogue vectors."""

from __future__ import annotations

from dataclasses import dataclass
import re
import string
import unicodedata

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


@dataclass(frozen=True)
class PreprocessingConfig:
    unicode_normalization: bool = True
    lowercase: bool = True
    normalize_numbers: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = True
    lemmatize: bool = True
    min_token_length: int = 2


def _load_stopwords() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except LookupError as exc:
        raise LookupError(
            "NLTK English stopwords are required. Run "
            "`python -m nltk.downloader stopwords`."
        ) from exc


def preprocess_text(
    text: str,
    config: PreprocessingConfig | None = None,
    *,
    stop_words: set[str] | None = None,
    lemmatizer: WordNetLemmatizer | None = None,
    tokenizer=word_tokenize,
) -> list[str]:
    """Normalize and tokenize one character document.

    Optional tokenizer, stop-word, and lemmatizer arguments make the function
    easier to test without changing the default pipeline used by the experiment.
    """
    config = config or PreprocessingConfig()

    if config.unicode_normalization:
        text = unicodedata.normalize("NFKD", text).encode(
            "ascii", "ignore"
        ).decode("utf-8", "ignore")

    if config.lowercase:
        text = text.lower()

    if config.normalize_numbers:
        text = re.sub(r"\d+", "<NUM>", text)

    try:
        tokens = tokenizer(text)
    except LookupError as exc:
        raise LookupError(
            "NLTK tokenization data is required. Run "
            "`python -m nltk.downloader punkt punkt_tab`."
        ) from exc

    if config.remove_stopwords and stop_words is None:
        stop_words = _load_stopwords()
    stop_words = stop_words or set()

    if config.lemmatize and lemmatizer is None:
        lemmatizer = WordNetLemmatizer()

    processed: list[str] = []
    for token in tokens:
        if config.remove_punctuation and token in string.punctuation:
            continue
        if config.remove_stopwords and (token in stop_words or token == "_eol_"):
            continue

        if config.lemmatize:
            try:
                token = lemmatizer.lemmatize(token)
            except LookupError as exc:
                raise LookupError(
                    "NLTK WordNet data is required. Run "
                    "`python -m nltk.downloader wordnet omw-1.4`."
                ) from exc

        if len(token) < config.min_token_length and token not in {"i", "a"}:
            continue

        processed.append(token)

    return processed
