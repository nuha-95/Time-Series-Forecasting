import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def dataloader():
    df = pd.read_csv('./Tasla_Stock_Updated_V2.csv')  
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.dropna()
    return df

def add_features(df):
    df = df.copy()
    # Technical indicators that don't use future information
    df['MA5'] = df['Close'].rolling(window=5).mean().shift(1)
    df['MA10'] = df['Close'].rolling(window=10).mean().shift(1)
    df['MA20'] = df['Close'].rolling(window=20).mean().shift(1)
    df['Volatility'] = df['Close'].rolling(window=10).std().shift(1)
    
    # Price momentum features
    df['Momentum_1d'] = df['Close'].pct_change(periods=1).shift(1)
    df['Momentum_5d'] = df['Close'].pct_change(periods=5).shift(1)
    
    # Monthly Return (without using future information)
    monthly_return = df['Close'].resample('M').ffill().pct_change().shift(1)
    monthly_return = monthly_return.fillna(method='bfill')
    df['Monthly_Return'] = monthly_return.resample('D').ffill().reindex(df.index, method='ffill')
    
    # Target: next day's closing price
    df['target'] = df['Close'].shift(-1)
    
    # Remove rows with missing values
    df.dropna(inplace=True)
    return df

# def data_split(X, y, test_size=0.2):
#     # Calculate the split point based on the size of the dataset
#     split_idx = int(len(X) * (1 - test_size))
    
#     # Split the data based on time (earlier data for training, later data for testing)
#     X_train = X[:split_idx]
#     X_test = X[split_idx:]
#     y_train = y[:split_idx]
#     y_test = y[split_idx:]
    
#     return X_train, X_test, y_train, y_test

def scale(X_train, X_test, y_train, y_test):
    """
    Scale the input features and target values using MinMaxScaler.
    
    Returns:
    --------
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler
    """
    # Scale features
    X_scaler = MinMaxScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    
    # Scale target
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler


def data_split(X, y, test_size=0.2, sort_key=None):
    """
    Split data into training and testing sets, with option to sort first.
    
    Parameters:
    -----------
    X : array-like
        Features dataset
    y : array-like
        Target dataset
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split (0 to 1)
    sort_key : callable or int, default=None
        If None, no sorting is performed.
        If int, will sort based on that column index in X.
        If callable, will use this function as the key for sorting.
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Split datasets for training and testing
    """
    # Convert inputs to numpy arrays if they aren't already
    X = np.array(X)
    y = np.array(y)
    
    # Check if the lengths match
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    
    # Create indices array
    indices = np.arange(len(X))
    
    # Sort the indices if a sort key is provided
    if sort_key is not None:
        if callable(sort_key):
            # Use the provided function as the sort key
            sorted_indices = sorted(indices, key=lambda i: sort_key(X[i]))
        elif isinstance(sort_key, int):
            # Sort based on a specific column in X
            sorted_indices = sorted(indices, key=lambda i: X[i][sort_key])
        else:
            raise ValueError("sort_key must be either a callable or an integer")
        
        # Reorder X and y based on the sorted indices
        X = X[sorted_indices]
        y = y[sorted_indices]
    
    # Calculate the split point
    split_idx = int(len(X) * (1 - test_size))
    
    # Split the data (earlier data for training, later data for testing)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test

# def scale(X_train, X_test, y_train, y_test):
#     """
#     Scale the input features and target values using MinMaxScaler.
    
#     Returns:
#     --------
#     X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler
#     """
#     # Scale features
#     X_scaler = MinMaxScaler()
#     X_train_scaled = X_scaler.fit_transform(X_train)
#     X_test_scaled = X_scaler.transform(X_test)
    
#     # Scale target
#     y_scaler = MinMaxScaler()
#     y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
#     y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
#     return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler

def plot_train_test_predictions(df, split_idx, y_test, y_pred, model_name='Model'):
    """
    Plot train, test and prediction data
    """
    plt.figure(figsize=(14, 8))
    
    # Get the dates
    train_dates = df.index[:split_idx]
    test_dates = df.index[split_idx:]
    
    # Plot training data
    plt.plot(train_dates, df['Close'][:split_idx], color='cyan', label='Train')
    
    # Plot test data
    plt.plot(test_dates, y_test, color='magenta', label='Test')
    
    # Plot predictions
    plt.plot(test_dates, y_pred, color='purple', label='Predictions')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Set labels and title
    plt.title(model_name, fontsize=18)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Close Price USD ($)', fontsize=16)
    
    # Add legend
    plt.legend(loc='best')
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.show()

def plot_model_comparison(results):
    """
    Create bar charts comparing model performance metrics
    """
    # Select metrics to display
    metrics_to_plot = ['R²', 'RMSE']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 6))
        
        # Extract metric values for each model
        models = list(results.keys())
        values = [results[model][metric] for model in models]
        
        # Different colors for each bar
        colors = ['cyan', 'magenta', 'yellow', 'lime', 'orange']
        
        # Create bar chart
        bars = plt.bar(models, values, color=colors)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', rotation=0, color='white')
        
        plt.title(f'Model Comparison - {metric}', fontsize=18)
        plt.ylabel(metric, fontsize=16)
        plt.xlabel('Models', fontsize=16)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()

###

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# function to prepare data for sequence models (LSTM, GRU)
def create_sequences(X, y, time_steps=1):
    """
    Create sequences of time_steps for sequence models like LSTM and GRU
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

# GRU model creation function
def create_gru_model(input_shape):
    """
    Create and compile GRU model
    """
    model = Sequential()
    model.add(GRU(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(GRU(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# LSTM model creation function
def create_lstm_model(input_shape):
    """
    Create and compile LSTM model
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

###

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
        self.plot_comparison(df, split_idx, offset, y_test_adjusted, rf_pred, lstm_pred, ensemble_pred)

 
        
        return results
    
    def plot_comparison(self, df, split_idx, offset, y_test, rf_pred, lstm_pred, ensemble_pred):
        """
        Plot comparison of ensemble and individual model predictions
        """
        plt.figure(figsize=(16, 10))
        
        # Get the dates
        train_dates = df.index[:split_idx]
        test_dates = df.index[split_idx+offset:split_idx+offset+len(ensemble_pred)]
        
        # Plot training data
        plt.plot(train_dates, df['Close'][:split_idx], color='cyan', label='Train')
        
        # Plot test data
        plt.plot(test_dates, y_test, color='white', label='Test', linewidth=2)
        
        # Plot predictions
        plt.plot(test_dates, rf_pred, color='magenta', label='Random Forest', alpha=0.7)
        plt.plot(test_dates, lstm_pred, color='yellow', label='LSTM', alpha=0.7)
        plt.plot(test_dates, ensemble_pred, color='lime', label='Ensemble', linewidth=2)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Set labels and title
        plt.title('Model Comparison', fontsize=18)
        plt.xlabel('Date', fontsize=16)
        plt.ylabel('Close Price USD ($)', fontsize=16)
        
        # Add legend
        plt.legend(loc='best')
        
        # Format x-axis dates
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






