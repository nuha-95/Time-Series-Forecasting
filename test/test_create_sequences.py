import unittest
import numpy as np
import sys
from pathlib import Path

# Dynamically add the project root to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]  
sys.path.insert(0, str(project_root))

from utils import create_sequences  

class TestCreateSequences(unittest.TestCase):

    def setUp(self):
        # Example data
        self.X = np.array([[i] for i in range(10)])  # Shape (10, 1)
        self.y = np.array([i for i in range(10)])    # Shape (10,)

    def test_output_shapes(self):
        time_steps = 3
        X_seq, y_seq = create_sequences(self.X, self.y, time_steps=time_steps)
        
        expected_num_sequences = len(self.X) - time_steps  # 10 - 3 = 7
        self.assertEqual(X_seq.shape, (expected_num_sequences, time_steps, self.X.shape[1]))
        self.assertEqual(y_seq.shape, (expected_num_sequences,))

    def test_sequence_content(self):
        time_steps = 2
        X_seq, y_seq = create_sequences(self.X, self.y, time_steps=time_steps)
        
        # Check first sequence manually
        expected_first_X = np.array([[0], [1]])
        expected_first_y = 2
        np.testing.assert_array_equal(X_seq[0], expected_first_X)
        self.assertEqual(y_seq[0], expected_first_y)

    def test_empty_input(self):
        X_empty = np.array([])
        y_empty = np.array([])
        X_seq, y_seq = create_sequences(X_empty, y_empty, time_steps=3)
        self.assertEqual(len(X_seq), 0)
        self.assertEqual(len(y_seq), 0)

    def test_insufficient_length(self):
        # When input length < time_steps, should return empty arrays
        short_X = np.array([[1], [2]])
        short_y = np.array([1, 2])
        X_seq, y_seq = create_sequences(short_X, short_y, time_steps=5)
        self.assertEqual(len(X_seq), 0)
        self.assertEqual(len(y_seq), 0)

if __name__ == '__main__':
    unittest.main()
