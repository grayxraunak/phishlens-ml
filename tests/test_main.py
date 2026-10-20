import tempfile
import unittest
from pathlib import Path

import pandas as pd

import main


class FeatureAndDataTests(unittest.TestCase):
    def test_feature_extraction_does_not_fetch_url(self):
        features = main.extract_features("https://example.com/login?next=1")
        self.assertEqual(features["https_scheme"], 1)
        self.assertEqual(features["hostname_length"], len("example.com"))
        self.assertEqual(features["query_length"], len("next=1"))

    def test_bundled_dataset_has_two_binary_classes(self):
        _, labels = main.load_data()
        self.assertEqual(set(labels.unique()), {0, 1})
        self.assertGreaterEqual(labels.value_counts().min(), 2)

    def test_invalid_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            pd.DataFrame({"url": ["https://example.com"], "label": [2]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                main.load_data(path)


class EvaluationTests(unittest.TestCase):
    def test_evaluation_is_repeatable_for_same_seed(self):
        first = main.evaluate(random_state=42)[0]
        second = main.evaluate(random_state=42)[0]
        pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))


if __name__ == "__main__":
    unittest.main()
