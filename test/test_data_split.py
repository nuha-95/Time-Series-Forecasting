import unittest
import numpy as np
import sys
from pathlib import Path

# Dynamically add the project root to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]  
sys.path.insert(0, str(project_root))

from utils import data_split

class TestDataSplit(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[5], [3], [8], [1], [6], [9]])
        self.y = np.array([50, 30, 80, 10, 60, 90])

    def test_split_ratio(self):
        X_train, X_test, y_train, y_test = data_split(self.X, self.y, test_size=0.33)
        total = len(X_train) + len(X_test)
        self.assertEqual(total, len(self.X))
        self.assertEqual(len(y_train), len(X_train))
        self.assertEqual(len(y_test), len(X_test))
        self.assertEqual(len(y_train), int(len(self.X) * (1 - 0.33)))

    def test_sort_by_column_index(self):
        X_train, X_test, y_train, y_test = data_split(self.X, self.y, sort_key=0)
        combined = np.concatenate((X_train, X_test)).flatten()
        self.assertTrue(np.all(np.diff(combined) >= 0))  # Should be sorted ascending

    def test_sort_by_callable(self):
        sort_func = lambda row: -row[0]  # Sort descending by value
        X_train, X_test, _, _ = data_split(self.X, self.y, test_size=0.5, sort_key=sort_func)
        combined = np.concatenate((X_train, X_test)).flatten()
        self.assertTrue(np.all(np.diff(combined) <= 0))  # Should be sorted descending

    def test_mismatched_lengths(self):
        y_bad = self.y[:-1]  # One item too short
        with self.assertRaises(ValueError):
            data_split(self.X, y_bad)

    def test_no_sorting(self):
        X_train, X_test, _, _ = data_split(self.X, self.y, test_size=0.5)
        # Should preserve order if no sort_key is passed
        expected_X_train = self.X[:3]
        np.testing.assert_array_equal(X_train, expected_X_train)

if __name__ == '__main__':
    unittest.main()
