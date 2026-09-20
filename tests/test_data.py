import unittest

import pandas as pd

from character_retrieval.data import build_character_documents


class DataTests(unittest.TestCase):
    def test_caps_lines_per_character_and_skips_empty(self):
        frame = pd.DataFrame(
            {
                "Character_name": ["A", "A", "A", "B", "B"],
                "Line": ["one", "two", "three", "hello", None],
            }
        )
        docs = build_character_documents(frame, max_lines_per_character=2)
        self.assertEqual(docs["A"], "one _EOL_ two")
        self.assertEqual(docs["B"], "hello")


if __name__ == "__main__":
    unittest.main()
