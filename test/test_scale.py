import unittest
import numpy as np
import sys
from pathlib import Path

# Dynamically add the project root to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]  
sys.path.insert(0, str(project_root))

from utils import scale

class TestScaleFunction(unittest.TestCase):

    def setUp(self):
        # Simulate some training and testing data
        self.X_train = np.array([[1, 2], [2, 3], [3, 4]], dtype=float)
        self.X_test = np.array([[2, 2], [3, 3]], dtype=float)
        self.y_train = np.array([10, 15, 20], dtype=float)
        self.y_test = np.array([12, 18], dtype=float)

    def test_output_shapes(self):
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, _ = scale(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        self.assertEqual(X_train_scaled.shape, self.X_train.shape)
        self.assertEqual(X_test_scaled.shape, self.X_test.shape)
        self.assertEqual(y_train_scaled.shape, self.y_train.shape)
        self.assertEqual(y_test_scaled.shape, self.y_test.shape)

    def test_scaled_range_X(self):
        X_train_scaled, _, _, _, _ = scale(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        self.assertTrue(np.all(X_train_scaled >= 0) and np.all(X_train_scaled <= 1))

    def test_scaled_range_y(self):
        _, _, y_train_scaled, _, _ = scale(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        self.assertTrue(np.all(y_train_scaled >= 0) and np.all(y_train_scaled <= 1))

    def test_y_scaler_inverse(self):
        _, _, y_train_scaled, _, y_scaler = scale(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        # Test inverse transform
        y_inverse = y_scaler.inverse_transform(y_train_scaled.reshape(-1, 1)).flatten()
        np.testing.assert_allclose(y_inverse, self.y_train, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
