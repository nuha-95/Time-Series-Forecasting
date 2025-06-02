# import unittest
# import numpy as np
# import pandas as pd
# from unittest.mock import MagicMock
# from sklearn.preprocessing import MinMaxScaler

# import sys
# from pathlib import Path

# # Dynamically add the project root to sys.path
# current_file = Path(__file__).resolve()
# project_root = current_file.parents[1]  
# sys.path.insert(0, str(project_root))

# import utils  
# from class_util import EnsembleModel, find_optimal_weights

# # Dummy sequence generator
# def dummy_create_sequences(X, y, time_steps):
#     X_seq = np.array([X[i:i + time_steps] for i in range(len(X) - time_steps)])
#     y_seq = y[time_steps:]
#     return X_seq, y_seq

# class TestEnsembleModel(unittest.TestCase):
#     def setUp(self):
#         # Mock models
#         self.rf_model = MagicMock()
#         self.lstm_model = MagicMock()

#         # Dummy data
#         self.time_steps = 3
#         self.X_test_scaled = np.array([[i] for i in range(20)])
#         self.y_test_scaled = np.array([i / 20.0 for i in range(20)])
#         self.y_scaler = MinMaxScaler()
#         self.y_scaler.fit(np.array(self.y_test_scaled).reshape(-1, 1))

#         # Mock predictions
#         self.rf_model.predict.return_value = np.linspace(0, 1, 20)
#         self.lstm_model.predict.return_value = np.linspace(0, 1, 17 - self.time_steps).reshape(-1, 1)

        
#         self.original_create_sequences = utils.create_sequences
#         utils.create_sequences = dummy_create_sequences

#         # Dummy DataFrame
#         self.df = pd.DataFrame({
#             'Close': np.linspace(100, 120, 40)
#         }, index=pd.date_range(start="2020-01-01", periods=40))

#         self.split_idx = 20

#         # Model
#         self.model = EnsembleModel(self.rf_model, self.lstm_model, time_steps=self.time_steps)

#     def tearDown(self):
        
#         utils.create_sequences = self.original_create_sequences

#     def test_predict_shapes(self):
#         ensemble_pred, rf_pred, lstm_pred, _, rf_pred_scaled, lstm_pred_scaled = self.model.predict(
#             self.X_test_scaled, self.y_scaler
#         )
#         min_len = min(len(rf_pred_scaled), len(lstm_pred_scaled))
#         self.assertEqual(len(ensemble_pred), min_len)
#         self.assertEqual(len(rf_pred), min_len)
#         self.assertEqual(len(lstm_pred), min_len)

#     def test_predict_range(self):
#         ensemble_pred, _, _, _, _, _ = self.model.predict(self.X_test_scaled, self.y_scaler)
#         self.assertTrue(np.all(ensemble_pred >= 0) and np.all(ensemble_pred <= 1))

#     def test_evaluate_metrics_keys(self):
#         results = self.model.evaluate(self.X_test_scaled, self.y_test_scaled, self.y_scaler, self.df, self.split_idx)
#         self.assertIn('Ensemble', results)
#         self.assertIn('LSTM', results)
#         self.assertIn('Random Forest (adjusted)', results)
#         for metrics in results.values():
#             self.assertIn('MSE', metrics)
#             self.assertIn('RMSE', metrics)
#             self.assertIn('MAE', metrics)
#             self.assertIn('R²', metrics)

#     def test_optimal_weights_sum_to_one(self):
#         weights = find_optimal_weights(self.rf_model, self.lstm_model, self.X_test_scaled, self.y_test_scaled, time_steps=self.time_steps)
#         self.assertAlmostEqual(sum(weights), 1.0, places=5)
#         self.assertEqual(len(weights), 2)

# if __name__ == '__main__':
#     unittest.main()

import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import sys
from pathlib import Path

# Dynamically add the project root to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]  
sys.path.insert(0, str(project_root))
import utils  
from class_util import EnsembleModel, find_optimal_weights

# Dummy LSTM-like model class
class DummyLSTMModel:
    def __init__(self, input_shape=(None, 1), units=50):
        self.input_shape = input_shape
        self.units = units
        self.is_fitted = False
        self.weights = None
        
    def fit(self, X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=0):
        """Simulate LSTM training"""
        self.is_fitted = True
        # Store some dummy weights based on training data
        self.weights = np.random.random((X.shape[-1], self.units))
        return self
    
    def predict(self, X):
        """Simulate LSTM prediction"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Simple linear transformation for prediction simulation
        predictions = []
        for sequence in X:
            # Simple calculation: weighted sum of last few values
            pred = np.mean(sequence[-3:]) + np.random.normal(0, 0.01)
            predictions.append([pred])
        
        return np.array(predictions)

# Dummy sequence generator
def dummy_create_sequences(X, y, time_steps):
    X_seq = np.array([X[i:i + time_steps] for i in range(len(X) - time_steps)])
    y_seq = y[time_steps:]
    return X_seq, y_seq

class TestEnsembleModel(unittest.TestCase):
    def setUp(self):
        # Create dummy training data
        np.random.seed(42)  # For reproducible results
        self.time_steps = 3
        
        # Generate synthetic time series data
        n_samples = 100
        time_index = pd.date_range(start="2020-01-01", periods=n_samples, freq='D')
        
        # Create a trending time series with some noise
        trend = np.linspace(100, 150, n_samples)
        noise = np.random.normal(0, 2, n_samples)
        prices = trend + noise
        
        self.df = pd.DataFrame({
            'Close': prices
        }, index=time_index)
        
        # Split data
        self.split_idx = 80
        train_data = self.df.iloc[:self.split_idx]
        test_data = self.df.iloc[self.split_idx:]
        
        # Prepare training data
        self.y_scaler = MinMaxScaler()
        self.X_scaler = MinMaxScaler()
        
        # Scale training data
        train_scaled = self.y_scaler.fit_transform(train_data[['Close']])
        test_scaled = self.y_scaler.transform(test_data[['Close']])
        
        # Prepare features (using price as feature for simplicity)
        X_train = train_scaled.flatten()
        y_train = train_scaled.flatten()
        
        self.X_test_scaled = test_scaled.flatten()
        self.y_test_scaled = test_scaled.flatten()
        
        # Create and train Random Forest model
        self.rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        # Create feature matrix for RF (using lagged values)
        X_train_rf = np.array([X_train[i:i+self.time_steps] for i in range(len(X_train) - self.time_steps)])
        y_train_rf = y_train[self.time_steps:]
        
        # Train RF model
        self.rf_model.fit(X_train_rf, y_train_rf)
        
        # Create and train dummy LSTM model
        self.lstm_model = DummyLSTMModel(input_shape=(self.time_steps, 1))
        
        # Prepare LSTM training data
        X_train_lstm = X_train_rf.reshape(-1, self.time_steps, 1)
        y_train_lstm = y_train_rf
        
        # Train LSTM model
        self.lstm_model.fit(X_train_lstm, y_train_lstm)
        
        # Mock utils.create_sequences for testing
        self.original_create_sequences = utils.create_sequences
        utils.create_sequences = dummy_create_sequences
        
        # Create ensemble model
        self.model = EnsembleModel(self.rf_model, self.lstm_model, time_steps=self.time_steps)
    
    def tearDown(self):
        utils.create_sequences = self.original_create_sequences
    
    def test_models_are_trained(self):
        """Test that both models are properly trained"""
        # RF model should be fitted
        self.assertTrue(hasattr(self.rf_model, 'estimators_'))
        
        # LSTM model should be fitted
        self.assertTrue(self.lstm_model.is_fitted)
    
    def test_models_can_predict(self):
        """Test that trained models can make predictions"""
        # Test RF prediction
        X_test_rf = np.array([self.X_test_scaled[i:i+self.time_steps] 
                             for i in range(len(self.X_test_scaled) - self.time_steps)])
        rf_pred = self.rf_model.predict(X_test_rf)
        self.assertIsInstance(rf_pred, np.ndarray)
        self.assertGreater(len(rf_pred), 0)
        
        # Test LSTM prediction
        X_test_lstm = X_test_rf.reshape(-1, self.time_steps, 1)
        lstm_pred = self.lstm_model.predict(X_test_lstm)
        self.assertIsInstance(lstm_pred, np.ndarray)
        self.assertGreater(len(lstm_pred), 0)
    
    def test_predict_shapes(self):
        """Test that ensemble predictions have correct shapes"""
        ensemble_pred, rf_pred, lstm_pred, rf_pred_scaled, lstm_pred_scaled = self.model.predict(
            self.X_test_scaled, self.y_scaler
        )
        
        min_len = min(len(rf_pred_scaled), len(lstm_pred_scaled))
        self.assertEqual(len(ensemble_pred), min_len)
        self.assertEqual(len(rf_pred), min_len)
        self.assertEqual(len(lstm_pred), min_len)
        self.assertGreater(min_len, 0)
    
    def test_predict_range(self):
        """Test that predictions are within reasonable range"""
        ensemble_pred, _, _, _, _ = self.model.predict(self.X_test_scaled, self.y_scaler)
        
        # Predictions should be within a reasonable range (not necessarily 0-1 after inverse scaling)
        self.assertFalse(np.any(np.isnan(ensemble_pred)))
        self.assertFalse(np.any(np.isinf(ensemble_pred)))
    
    def test_evaluate_metrics_keys(self):
        """Test that evaluation returns all expected metrics"""
        results = self.model.evaluate(
            self.X_test_scaled, 
            self.y_test_scaled, 
            self.y_scaler, 
            self.df, 
            self.split_idx
        )
        
        # Check that all model results are present
        self.assertIn('Ensemble', results)
        self.assertIn('LSTM', results)
        self.assertIn('Random Forest (adjusted)', results)
        
        # Check that all metrics are present for each model
        for model_name, metrics in results.items():
            self.assertIn('MSE', metrics)
            self.assertIn('RMSE', metrics)
            self.assertIn('MAE', metrics)
            self.assertIn('R²', metrics)
            
            # Check that metrics are numeric
            for metric_name, value in metrics.items():
                self.assertIsInstance(value, (int, float, np.number))
                self.assertFalse(np.isnan(value))
    
    def test_optimal_weights_sum_to_one(self):
        """Test that optimal weights sum to 1"""
        weights = find_optimal_weights(
            self.rf_model, 
            self.lstm_model, 
            self.X_test_scaled, 
            self.y_test_scaled, 
            time_steps=self.time_steps
        )
        
        self.assertAlmostEqual(sum(weights), 1.0, places=5)
        self.assertEqual(len(weights), 2)
        
        # Weights should be non-negative
        self.assertTrue(all(w >= 0 for w in weights))
    
    def test_ensemble_performance(self):
        """Test that ensemble model performs reasonably"""
        results = self.model.evaluate(
            self.X_test_scaled, 
            self.y_test_scaled, 
            self.y_scaler, 
            self.df, 
            self.split_idx
        )
        
        # Ensemble should have reasonable performance metrics
        ensemble_metrics = results['Ensemble']
        
        # R² should be reasonable (not perfect, but not terrible)
        self.assertGreaterEqual(ensemble_metrics['R²'], -1.0)  # R² can be negative
        self.assertLessEqual(ensemble_metrics['R²'], 1.0)
        
        # MSE and RMSE should be positive
        self.assertGreater(ensemble_metrics['MSE'], 0)
        self.assertGreater(ensemble_metrics['RMSE'], 0)
        self.assertGreater(ensemble_metrics['MAE'], 0)
    
    def test_data_consistency(self):
        """Test that training and test data are consistent"""
        # Check that scalers are fitted
        self.assertTrue(hasattr(self.y_scaler, 'scale_'))
        
        # Check data shapes
        self.assertGreater(len(self.X_test_scaled), self.time_steps)
        self.assertEqual(len(self.X_test_scaled), len(self.y_test_scaled))
        
        # Check that test data is properly split
        self.assertEqual(len(self.df) - self.split_idx, len(self.X_test_scaled))

if __name__ == '__main__':
    unittest.main()