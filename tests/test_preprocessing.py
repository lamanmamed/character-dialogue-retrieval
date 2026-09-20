import unittest

from character_retrieval.preprocessing import PreprocessingConfig, preprocess_text


class FakeLemmatizer:
    def lemmatize(self, token):
        return {"cars": "car", "dogs": "dog"}.get(token, token)


class PreprocessingTests(unittest.TestCase):
    def test_normalizes_case_numbers_stopwords_and_punctuation(self):
        config = PreprocessingConfig(
            remove_stopwords=True,
            lemmatize=False,
            min_token_length=1,
        )
        tokens = preprocess_text(
            "The CAFE has 12 cats! _EOL_",
            config,
            stop_words={"the", "has"},
            tokenizer=lambda text: text.replace("!", " ! ").split(),
        )
        self.assertEqual(tokens, ["cafe", "<NUM>", "cats"])

    def test_uses_injected_lemmatizer(self):
        config = PreprocessingConfig(
            remove_stopwords=False,
            lemmatize=True,
            min_token_length=1,
        )
        tokens = preprocess_text(
            "Cars dogs",
            config,
            lemmatizer=FakeLemmatizer(),
            tokenizer=str.split,
        )
        self.assertEqual(tokens, ["car", "dog"])


if __name__ == "__main__":
    unittest.main()
