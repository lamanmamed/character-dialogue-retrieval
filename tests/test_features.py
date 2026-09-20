import unittest

import numpy as np

from character_retrieval.features import DialogueVectorizer, FeatureConfig, style_features


class FeatureTests(unittest.TestCase):
    def test_style_features_have_five_columns(self):
        matrix = style_features([["short", "verylongword", "12"], ["a", "b"]])
        self.assertEqual(matrix.shape, (2, 5))
        self.assertTrue(np.isfinite(matrix).all())

    def test_unigram_bigram_vectorizer_transforms_heldout_text(self):
        vectorizer = DialogueVectorizer(
            FeatureConfig(
                word_ngram_range=(1, 2),
                min_df=1,
                include_style=False,
            )
        )
        train = vectorizer.fit_transform([
            ["tea", "please", "tea"],
            ["football", "team", "goal"],
        ])
        heldout = vectorizer.transform([["football", "goal"]])
        self.assertEqual(train.shape[0], 2)
        self.assertEqual(heldout.shape[0], 1)
        self.assertEqual(train.shape[1], heldout.shape[1])


if __name__ == "__main__":
    unittest.main()
