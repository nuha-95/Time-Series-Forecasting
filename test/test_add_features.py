import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Dynamically add the project root to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]  
sys.path.insert(0, str(project_root))

from utils import add_features

class TestAddFeatures(unittest.TestCase):

    def setUp(self):
        # Create a minimal dummy DataFrame with a datetime index and 'Close' column
        dates = pd.date_range(start="2022-01-01", periods=60, freq='D')
        close_prices = np.linspace(100, 160, 60)
        self.df = pd.DataFrame({'Close': close_prices}, index=dates)

    def test_output_is_dataframe(self):
        result = add_features(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_expected_columns_exist(self):
        result = add_features(self.df)
        expected_cols = [
            'MA5', 'MA10', 'MA20', 'Volatility',
            'Momentum_1d', 'Momentum_5d', 'Monthly_Return', 'target'
        ]
        for col in expected_cols:
            self.assertIn(col, result.columns)

    def test_no_missing_values(self):
        result = add_features(self.df)
        self.assertFalse(result.isnull().values.any())

    def test_target_shift(self):
        result = add_features(self.df)
        # Since target = Close.shift(-1), check if that's true for a few rows
        for i in range(5):
            self.assertAlmostEqual(
                result['target'].iloc[i], 
                result['Close'].iloc[i + 1],
                places=5
            )

    def test_output_not_empty(self):
        result = add_features(self.df)
        self.assertGreater(len(result), 0)

if __name__ == '__main__':
    unittest.main()
