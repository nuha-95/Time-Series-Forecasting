
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from utils import create_sequences

class EnsembleModel:
    def __init__(self, rf_model, lstm_model, time_steps=10, blend_weights=None):
        """
        Ensemble model that combines RF and LSTM predictions
        
        Parameters:
        -----------
        rf_model : RandomForestRegressor
            Trained Random Forest model
        lstm_model : Keras Sequential
            Trained LSTM model
        time_steps : int
            Number of time steps for LSTM input
        blend_weights : list or None
            Weights for blending model predictions [rf_weight, lstm_weight]
            If None, use equal weights
        """
        self.rf_model = rf_model
        self.lstm_model = lstm_model
        self.time_steps = time_steps
        
        # Default to equal weights if not specified
        self.blend_weights = blend_weights if blend_weights is not None else [0.5, 0.5]
        
    def predict(self, X_test_scaled, y_scaler):
        """
        Generate ensemble predictions
        
        Parameters:
        -----------
        X_test_scaled : array
            Scaled test features
        y_scaler : scaler
            Scaler used to transform target values
            
        Returns:
        --------
        ensemble_pred : array
            Ensemble predictions (unscaled)
        rf_pred : array
            Random Forest predictions (unscaled)
        lstm_pred : array
            LSTM predictions (unscaled)
        """
        # Random Forest prediction (all test samples)
        rf_pred_scaled = self.rf_model.predict(X_test_scaled)
        
        # LSTM prediction (need to create sequences)
        X_test_seq, _ = create_sequences(X_test_scaled, np.zeros(len(X_test_scaled)), self.time_steps)
        X_test_lstm = X_test_seq.reshape(X_test_seq.shape[0], X_test_seq.shape[1], X_test_seq.shape[2])
        lstm_pred_scaled = self.lstm_model.predict(X_test_lstm).flatten()
        
        # Handle sequence truncation (LSTM will have fewer predictions due to sequence creation)
        offset = self.time_steps
        rf_pred_scaled_trimmed = rf_pred_scaled[offset:]
        
        # Length check to avoid issues
        min_len = min(len(rf_pred_scaled_trimmed), len(lstm_pred_scaled))
        rf_pred_scaled_trimmed = rf_pred_scaled_trimmed[:min_len]
        lstm_pred_scaled = lstm_pred_scaled[:min_len]
        
        # Blend predictions using weights
        ensemble_pred_scaled = (
            self.blend_weights[0] * rf_pred_scaled_trimmed + 
            self.blend_weights[1] * lstm_pred_scaled
        )
        
        # Inverse transform predictions back to original scale
        ensemble_pred = y_scaler.inverse_transform(ensemble_pred_scaled.reshape(-1, 1)).flatten()
        rf_pred = y_scaler.inverse_transform(rf_pred_scaled_trimmed.reshape(-1, 1)).flatten()
        lstm_pred = y_scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()
        
        return ensemble_pred, rf_pred, lstm_pred, ensemble_pred_scaled, rf_pred_scaled_trimmed, lstm_pred_scaled
    
    def evaluate(self, X_test_scaled, y_test_scaled, y_scaler, df, split_idx):
        """
        Evaluate ensemble model and compare with individual models
        
        Parameters:
        -----------
        X_test_scaled : array
            Scaled test features
        y_test_scaled : array
            Scaled test target values
        y_scaler : scaler
            Scaler used to transform target values
        df : DataFrame
            Original dataframe with stock data
        split_idx : int
            Index where train/test split occurs
            
        Returns:
        --------
        results : dict
            Dictionary with performance metrics
        """
        # Generate predictions
        ensemble_pred, rf_pred, lstm_pred, ensemble_pred_scaled, rf_pred_scaled, lstm_pred_scaled = self.predict(
            X_test_scaled, y_scaler
        )
        
        # Adjust y_test for fair comparison (account for sequence truncation)
        offset = self.time_steps
        y_test_scaled_adjusted = y_test_scaled[offset:offset+len(ensemble_pred_scaled)]
        y_test_adjusted = y_scaler.inverse_transform(y_test_scaled_adjusted.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        results = {}
        
        # Ensemble metrics
        ensemble_mse = mean_squared_error(y_test_scaled_adjusted, ensemble_pred_scaled)
        ensemble_rmse = np.sqrt(ensemble_mse)
        ensemble_mae = mean_absolute_error(y_test_scaled_adjusted, ensemble_pred_scaled)
        ensemble_r2 = r2_score(y_test_scaled_adjusted, ensemble_pred_scaled)
        
        results['Ensemble'] = {
            'MSE': ensemble_mse,
            'RMSE': ensemble_rmse,
            'MAE': ensemble_mae,
            'R²': ensemble_r2
        }
        
        # RF metrics (on adjusted test set)
        rf_mse = mean_squared_error(y_test_scaled_adjusted, rf_pred_scaled)
        rf_rmse = np.sqrt(rf_mse)
        rf_mae = mean_absolute_error(y_test_scaled_adjusted, rf_pred_scaled)
        rf_r2 = r2_score(y_test_scaled_adjusted, rf_pred_scaled)
        
        results['Random Forest (adjusted)'] = {
            'MSE': rf_mse,
            'RMSE': rf_rmse,
            'MAE': rf_mae,
            'R²': rf_r2
        }
        
        # LSTM metrics
        lstm_mse = mean_squared_error(y_test_scaled_adjusted, lstm_pred_scaled)
        lstm_rmse = np.sqrt(lstm_mse)
        lstm_mae = mean_absolute_error(y_test_scaled_adjusted, lstm_pred_scaled)
        lstm_r2 = r2_score(y_test_scaled_adjusted, lstm_pred_scaled)
        
        results['LSTM'] = {
            'MSE': lstm_mse,
            'RMSE': lstm_rmse,
            'MAE': lstm_mae,
            'R²': lstm_r2
        }
        
        print("\nEvaluation Metrics:")
        for model_name, metrics in results.items():
            print(f"\n {model_name}")
            for metric, value in metrics.items():
                print(f"   {metric}: {value:.4f}")

        # Plot results
        #self.plot_comparison(df, split_idx, offset, y_test_adjusted, ensemble_pred)


 
        
        return results
    
    # def plot_comparison(self, df, split_idx, offset, y_test, rf_pred, lstm_pred, ensemble_pred):
    #     """
    #     Plot comparison of ensemble and individual model predictions
    #     """
    #     plt.figure(figsize=(16, 10))
        
    #     # Get the dates
    #     train_dates = df.index[:split_idx]
    #     test_dates = df.index[split_idx+offset:split_idx+offset+len(ensemble_pred)]
        
    #     # Plot training data
    #     plt.plot(train_dates, df['Close'][:split_idx], color='cyan', label='Train')
        
    #     # Plot test data
    #     plt.plot(test_dates, y_test, color='white', label='Test', linewidth=2)
        
    #     # Plot predictions
    #     plt.plot(test_dates, rf_pred, color='magenta', label='Random Forest', alpha=0.7)
    #     plt.plot(test_dates, lstm_pred, color='yellow', label='LSTM', alpha=0.7)
    #     plt.plot(test_dates, ensemble_pred, color='lime', label='Ensemble', linewidth=2)
        
    #     # Add grid
    #     plt.grid(True, alpha=0.3)
        
    #     # Set labels and title
    #     plt.title('Model Comparison', fontsize=18)
    #     plt.xlabel('Date', fontsize=16)
    #     plt.ylabel('Close Price USD ($)', fontsize=16)
        
    #     # Add legend
    #     plt.legend(loc='best')
        
    #     # Format x-axis dates
    #     plt.gcf().autofmt_xdate()
        
    #     plt.tight_layout()
    #     plt.show()


    

    
    

    def plot_comparison(self, df, split_idx, offset, y_test, ensemble_pred):
        """
        Plot comparison of Ensemble model predictions with training and test data
        """
        # Get dates
        train_dates = df.index[:split_idx]
        test_dates = df.index[split_idx+offset:split_idx+offset+len(ensemble_pred)]

        # Set up the figure
        plt.figure(figsize=(16, 8))

        # Plot training data
        plt.plot(train_dates, df['Close'][:split_idx], color='magenta', label='Train', linestyle='--')

        # Plot test data
        plt.plot(test_dates, y_test, color='cyan', label='Test', linestyle='--')

        # Plot ensemble prediction
        plt.plot(test_dates, ensemble_pred, color='yellow', label='Ensemble Prediction', linestyle='--')

        # Set title and labels
        plt.title('Ensemble Model vs Actual', fontsize=18)
        plt.xlabel('Date', fontsize=16)
        plt.ylabel('Close Price USD ($)', fontsize=16)

        # Grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')

        # Format x-axis for dates
        plt.gcf().autofmt_xdate()

        plt.tight_layout()
        plt.show()





def find_optimal_weights(rf_model, lstm_model, X_test_scaled, y_test_scaled, time_steps=10):
    """
    Find optimal weights for blending RF and LSTM predictions
    
    Parameters:
    -----------
    rf_model : RandomForestRegressor
        Trained Random Forest model
    lstm_model : Keras Sequential
        Trained LSTM model
    X_test_scaled : array
        Scaled test features
    y_test_scaled : array
        Scaled test target values
    time_steps : int
        Number of time steps for LSTM input
        
    Returns:
    --------
    optimal_weights : list
        Optimal weights [rf_weight, lstm_weight]
    """
    # Create sequences for LSTM
    X_test_seq, _ = create_sequences(X_test_scaled, np.zeros(len(X_test_scaled)), time_steps)
    X_test_lstm = X_test_seq.reshape(X_test_seq.shape[0], X_test_seq.shape[1], X_test_seq.shape[2])
    
    # Get individual model predictions
    rf_pred_scaled = rf_model.predict(X_test_scaled)
    lstm_pred_scaled = lstm_model.predict(X_test_lstm).flatten()
    
    # Handle sequence truncation
    offset = time_steps
    rf_pred_scaled_trimmed = rf_pred_scaled[offset:]
    y_test_scaled_adjusted = y_test_scaled[offset:offset+len(lstm_pred_scaled)]
    
    # Length check to avoid issues
    min_len = min(len(rf_pred_scaled_trimmed), len(lstm_pred_scaled), len(y_test_scaled_adjusted))
    rf_pred_scaled_trimmed = rf_pred_scaled_trimmed[:min_len]
    lstm_pred_scaled = lstm_pred_scaled[:min_len]
    y_test_scaled_adjusted = y_test_scaled_adjusted[:min_len]
    
    # Stack predictions for regression
    X_blend = np.column_stack((rf_pred_scaled_trimmed, lstm_pred_scaled))
    
    # Fit a linear regression model to find optimal weights
    lr_blend = LinearRegression(fit_intercept=False)
    lr_blend.fit(X_blend, y_test_scaled_adjusted)
    
    # Get weights and normalize them to sum to 1
    weights = lr_blend.coef_
    optimal_weights = weights / np.sum(weights)
    
    return optimal_weights.tolist()
