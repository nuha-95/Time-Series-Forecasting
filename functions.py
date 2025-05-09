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
    
    # Trading volume features
    # if 'Volume' in df.columns:
    #     df['Volume_Change'] = df['Volume'].pct_change().shift(1)
    #     df['Volume_MA5'] = df['Volume'].rolling(window=5).mean().shift(1)
    
    # Monthly Return (without using future information)
    monthly_return = df['Close'].resample('M').ffill().pct_change().shift(1)
    monthly_return = monthly_return.fillna(method='bfill')
    df['Monthly_Return'] = monthly_return.resample('D').ffill().reindex(df.index, method='ffill')
    
    # Target: next day's closing price
    df['target'] = df['Close'].shift(-1)
    
    # Remove rows with missing values
    df.dropna(inplace=True)
    return df

def data_split(X, y, test_size=0.2):
    # Calculate the split point based on the size of the dataset
    split_idx = int(len(X) * (1 - test_size))
    
    # Split the data based on time (earlier data for training, later data for testing)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test

def scale(X_train, X_test, y_train, y_test):

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1,1)).flatten()
    y_test_scaled = scaler.transform(y_test.reshape(-1,1)).flatten()

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled