import unittest
import sys
from pathlib import Path

# Dynamically add the project root to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]  # Go up to TDD-Project-Nuha/
sys.path.insert(0, str(project_root))

from utils import dataloader


import pandas as pd
# from your_module import dataloader  # Replace 'your_module' with your actual filename

class TestDataLoader(unittest.TestCase):

    def test_dataloader_returns_dataframe(self):
        df = dataloader()
        self.assertIsInstance(df, pd.DataFrame)

    def test_index_is_datetime(self):
        df = dataloader()
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df.index))

    def test_no_missing_values(self):
        df = dataloader()
        self.assertFalse(df.isnull().values.any())

    def test_non_empty_dataframe(self):
        df = dataloader()
        self.assertGreater(len(df), 0)

if __name__ == '__main__':
    unittest.main()
