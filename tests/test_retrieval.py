import unittest

from character_retrieval import CharacterDialogueRetriever, FeatureConfig
from character_retrieval.evaluation import evaluate_rankings


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.retriever = CharacterDialogueRetriever(
            features=FeatureConfig(min_df=1, include_style=False),
            preprocessor=lambda text: text.lower().split(),
        ).fit(
            {
                "ALICE": "tea cake tea lovely",
                "BOB": "football goal team match",
                "CAROL": "garden rose flower garden",
            }
        )

    def test_ranks_matching_dialogue_first(self):
        ranking = self.retriever.rank("football team goal")
        self.assertEqual(ranking[0][0], "BOB")

    def test_evaluation_reports_perfect_synthetic_accuracy(self):
        result = self.retriever.evaluate(
            {
                "ALICE": "tea cake lovely",
                "BOB": "team football goal",
                "CAROL": "rose garden flower",
            }
        )
        self.assertEqual(result.mean_rank, 1.0)
        self.assertEqual(result.accuracy, 1.0)

    def test_rank_metric_distinguishes_second_from_third(self):
        reference = [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]
        query = [[0.9, 0.1]]
        result = evaluate_rankings(reference, query, ["A", "B", "C"], ["B"])
        self.assertEqual(result.per_character[0].rank, 2)


if __name__ == "__main__":
    unittest.main()
