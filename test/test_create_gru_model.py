import unittest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.optimizers import Adam

import sys
from pathlib import Path

# Dynamically add the project root to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]  
sys.path.insert(0, str(project_root))

from utils import create_gru_model

class TestCreateGRUModel(unittest.TestCase):

    def test_model_creation(self):
        input_shape = (10, 5)  # 10 time steps, 5 features
        model = create_gru_model(input_shape)
        
        # Check instance
        self.assertIsInstance(model, Sequential)

    def test_model_layers(self):
        input_shape = (15, 3)
        model = create_gru_model(input_shape)
        
        # Expected layer types
        layer_types = [GRU, Dropout, GRU, Dropout, Dense, Dense]
        self.assertEqual(len(model.layers), len(layer_types))
        for layer, expected_type in zip(model.layers, layer_types):
            self.assertIsInstance(layer, expected_type)

    def test_model_output(self):
        input_shape = (5, 2)
        model = create_gru_model(input_shape)
        
        # Output layer should have 1 unit
        self.assertEqual(model.layers[-1].units, 1)

    def test_model_compilation(self):
        input_shape = (20, 6)
        model = create_gru_model(input_shape)

        # Check loss and optimizer
        self.assertEqual(model.loss, 'mean_squared_error')
        self.assertIsInstance(model.optimizer, Adam)
        self.assertAlmostEqual(model.optimizer.learning_rate.numpy(), 0.001, places=5)

if __name__ == '__main__':
    unittest.main()
