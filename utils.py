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
    plt.plot(train_dates, df['Close'][:split_idx], color='magenta', label='Train', linestyle='--')
    
    # Plot test data
    plt.plot(test_dates, y_test, color='cyan', label='Test', linestyle='--')
    
    # Plot predictions
    plt.plot(test_dates, y_pred, color='yellow', label='Predictions', linestyle='--')
    
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



# def plot_model_comparison(results):
#     """
#     Create bar charts comparing model performance metrics
#     for both train-test split and prediction on unseen data.
#     """
#     metrics_to_plot = ['R²', 'RMSE']
#     eval_phases = ['train', 'predict']
#     phase_titles = {
#         'train': 'Train/Test Split Performance',
#         'predict': 'Future Prediction Performance'
#     }

#     for metric in metrics_to_plot:
#         for phase in eval_phases:
#             plt.figure(figsize=(12, 6))

#             models = list(results.keys())
#             values = [results[model][phase][metric] for model in models]
#             colors = ['cyan', 'magenta', 'yellow', 'lime', 'orange']

#             bars = plt.bar(models, values, color=colors)

#             for bar in bars:
#                 height = bar.get_height()
#                 plt.text(bar.get_x() + bar.get_width()/2., height,
#                          f'{height:.4f}', ha='center', va='bottom', color='white')

#             plt.title(f'{phase_titles[phase]} - {metric}', fontsize=18)
#             plt.ylabel(metric, fontsize=16)
#             plt.xlabel('Models', fontsize=16)
#             plt.grid(True, alpha=0.3, axis='y')
#             plt.tight_layout()
#             plt.show()

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



###





def run_single_inference(model_name, models, X_test_scaled, y_test_scaled, y_scaler, model_type="default", timesteps=None):
    import random
    import numpy as np

    model = models[model_name]

    # Pick a random index
    max_index = X_test_scaled.shape[0] - 1
    random_index = random.randint(0, max_index)

    # Prepare input based on model type
    if model_type in ["lstm", "gru"]:
        # Shape: [1, timesteps, features]
        sample_input = X_test_scaled[random_index].reshape(1, timesteps, -1)
        true_scaled = y_test_scaled[random_index]
    else:
        # Shape: [1, features]
        sample_input = X_test_scaled[random_index].reshape(1, -1)
        true_scaled = y_test_scaled[random_index]

    # Get actual (unscaled) value
    true_value = y_scaler.inverse_transform([[true_scaled]])[0, 0]

    # Predict
    if model_type in ["lstm", "gru",]:
        pred_scaled = model.predict(sample_input, verbose=0)[0]
    else:
        pred_scaled = model.predict(sample_input)[0]

    # If prediction is a list or array, flatten it
    if isinstance(pred_scaled, (np.ndarray, list)):
        pred_scaled = pred_scaled[0]

    # Unscale predicted value
    pred_value = y_scaler.inverse_transform([[pred_scaled]])[0, 0]

    # Compute error and accuracy
    abs_error = abs(pred_value - true_value)
    pct_error = (abs_error / abs(true_value)) * 100 if true_value != 0 else 0
    individual_accuracy = 100 - pct_error

    # Print results
    print(f"\n--- Inference for {model_name} ---")
    print(f"Sample Index: {random_index}")
    print(f"Predicted Value: {pred_value:.4f}")
    print(f"Actual Value:    {true_value:.4f}")
    print(f"Absolute Error:  {abs_error:.4f}")
    print(f"Accuracy:        {individual_accuracy:.2f}%")

    # return {
    #     "index": random_index,
    #     "predicted": pred_value,
    #     "actual": true_value,
    #     "abs_error": abs_error,
    #     "accuracy": individual_accuracy
    # }

def run_single_inference_ensemble(models, X_test_scaled, y_test_scaled, y_scaler, time_steps):
    import random
    import numpy as np

    # Get ensemble model
    ensemble_model = models["Ensemble"]

    # Pick a random index after the time_steps offset (to match LSTM requirement)
    max_index = X_test_scaled.shape[0] - 1
    random_index = random.randint(time_steps, max_index)

    # --- Prepare inputs ---
    # RF input: single flattened vector
    rf_input = X_test_scaled[random_index].reshape(1, -1)

    # LSTM input: sequence ending at that index
    lstm_input = X_test_scaled[random_index - time_steps:random_index].reshape(1, time_steps, -1)

    # True scaled and actual value
    true_scaled = y_test_scaled[random_index]
    true_value = y_scaler.inverse_transform([[true_scaled]])[0, 0]

    # --- Get predictions ---
    rf_pred_scaled = ensemble_model.rf_model.predict(rf_input)[0]
    lstm_pred_scaled = ensemble_model.lstm_model.predict(lstm_input, verbose=0)[0][0]

    # Blend predictions
    blend_weights = ensemble_model.blend_weights
    ensemble_pred_scaled = (
        blend_weights[0] * rf_pred_scaled + blend_weights[1] * lstm_pred_scaled
    )

    # Inverse transform
    pred_value = y_scaler.inverse_transform([[ensemble_pred_scaled]])[0, 0]

    # Accuracy metrics
    abs_error = abs(pred_value - true_value)
    pct_error = (abs_error / abs(true_value)) * 100 if true_value != 0 else 0
    accuracy = 100 - pct_error

    # --- Print results ---
    print(f"\n--- Ensemble Inference ---")
    print(f"Sample Index:    {random_index}")
    print(f"Predicted Value: {pred_value:.4f}")
    print(f"Actual Value:    {true_value:.4f}")
    print(f"Absolute Error:  {abs_error:.4f}")
    print(f"Accuracy:        {accuracy:.2f}%")
