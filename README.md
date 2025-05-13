

# 📈 Tesla Stock Price Forecasting using Machine Learning & Deep Learning

## 🚀 Project Overview

Financial time series forecasting is a critical area in quantitative finance. This project focuses on forecasting Tesla’s monthly **closing stock price** using historical data. Due to its volatility and high trading volume, Tesla stock offers a challenging yet insightful case study for modeling and prediction.

We build a comprehensive machine learning pipeline—from **data preprocessing** and **exploratory data analysis (EDA)** to **feature engineering**, **model training**, and **performance evaluation**. The objective is to compare traditional, machine learning, and deep learning models for their predictive power and generalization in different market conditions.

---

## ❓ Problem Statement

Given historical **daily stock data** for Tesla, forecast the **closing price for the next month** using supervised learning. Compare models and evaluate their accuracy using standard metrics. Perform EDA to understand the market behavior and volatility patterns.

---

## 📦 Dataset

* **Source:** [Kaggle - Tesla Historical Stock Prices](https://www.kaggle.com/)
* **Features:**

  * `Date`
  * `Open`, `High`, `Low`, `Close`
  * `Volume`

---


## 🔄 Project Phases

### 1. 🧹 Data Collection & Preprocessing

* Convert `Date` to datetime format and set as index.
* Handle missing/null values.
* Feature engineering:

  * Monthly returns
  * Moving averages (MA5, MA10, MA20)
  * Volatility (standard deviation over rolling window)
* Normalize/standardize features where appropriate.

### 2. 📊 Exploratory Data Analysis (EDA)

* Visualize price trends and volatility.
* Seasonality and cyclical patterns.
* Correlation heatmaps.
* Outlier detection (e.g., splits, spikes).
* Volume vs price relationship.



### 🧠 Model Development & Training

#### **Baseline Model**

* **Linear Regression**: Served as the initial benchmark to evaluate the performance of more complex models.

#### **Machine Learning Models**

* **Random Forest Regressor**: Captures non-linear relationships and handles feature importance effectively.
* **XGBoost Regressor**: Optimized gradient boosting algorithm for improved accuracy and performance.
* **Decision Tree Regressor**: A simple yet powerful tree-based model to understand feature splits.
* **Support Vector Regressor (SVR)**: Useful for high-dimensional regression problems.

#### **Deep Learning Models**

* **LSTM (Long Short-Term Memory)**: Captures long-term dependencies in sequential data.
* **GRU (Gated Recurrent Unit)**: A lightweight alternative to LSTM with similar performance on time-series data.

#### **Ensemble Model**

* **Hybrid Ensemble (LSTM + Random Forest)**: Combines the temporal learning capabilities of LSTM with the robust feature-based decision power of Random Forest for enhanced prediction accuracy.



### 4. 📈 Evaluation & Optimization

* Metrics: MAE, MSE, RMSE, R² Score
* Compare actual vs predicted visually
* Analyze errors under high-volatility periods
* Fine-tune for optimal performance



---

## 📈 Results Summary

| Model             | MAE   | RMSE  | R² Score |
| ----------------- | ----- | ----- | -------- |
| Linear Regression | 0.015 | 0.021 | 0.9714   |
| Random Forest     | 0.022 | 0.029 | 0.9461   |
| XGBoost           | 0.021 | 0.027 | 0.9505   |
| Decision Tree     | 0.029 | 0.039 | 0.9024   |
| SVR               | 0.030 | 0.036 | 0.9141   |
| LSTM              | 0.035 | 0.043 | 0.8618   |
| GRU               | 0.030 | 0.038 | 00.8959  |
| Ensemble(LSTM+RF) | 0.022 | 0.029 | 0.9387   |





