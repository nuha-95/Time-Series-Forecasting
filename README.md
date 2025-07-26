

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



Here's the **modified "Model Development & Training" section** of your README, now including **Prophet** with a short description and a plot placeholder:

---

## 🧰 Setup & Testing

`git clone https://github.com/nuha-95/Time-Series-Forecasting.git
cd Time-Series-Forecasting`

### 1. Install project dependencies

`pip install -r requirements.txt`

### 2. Run the forecasting pipeline

`jupyter notebook Tesla_train.ipynb`

### 3. Run unit tests

`pytest tests/`

---

## 🧠 Model Development & Training

### **Baseline Model**

* **Linear Regression**
  Serves as the benchmark model for evaluating other complex methods. It assumes a linear relationship between features and target.

---

### **Machine Learning Models**

* **Random Forest Regressor**
  Captures non-linear relationships through multiple decision trees and reduces overfitting via ensemble averaging.

* **XGBoost Regressor**
  A high-performance gradient boosting model known for speed and accuracy in regression tasks.

* **Decision Tree Regressor**
  A simple, interpretable tree-based model that splits data based on feature importance.

* **Support Vector Regressor (SVR)**
  Uses kernel methods to perform regression in high-dimensional space, useful for capturing complex trends.

---

### **Deep Learning Models**

* **LSTM (Long Short-Term Memory)**
  A type of recurrent neural network (RNN) ideal for capturing long-term dependencies in time series data.

* **GRU (Gated Recurrent Unit)**
  A simpler alternative to LSTM with fewer parameters, offering similar performance in sequence modeling.

---

### **Ensemble Model**

* **Hybrid Ensemble (LSTM + Random Forest)**
  Combines the sequential learning power of LSTM with the feature-based decision-making of Random Forest to boost accuracy.

---

### **Statistical Models**

* **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors)**
  A classical statistical model designed for time series forecasting, SARIMAX handles trend, seasonality, and incorporates external regressors (e.g., moving averages, momentum indicators). It's useful for interpretability and performance in more stable data segments.

 <img width="1189" height="590" alt="output" src="https://github.com/user-attachments/assets/2c0dec2d-2e19-4390-b5cd-045240a7d0a7" />
 
---


### **Time Series Model**

* **Prophet (by Facebook)**
  A decomposable time series model that handles **seasonality**, **trends**, and **holidays**. Especially effective on business time series data with strong seasonal effects. It also allows inclusion of additional **regressors** to improve forecasting power.

  <img width="1162" height="450" alt="newplot" src="https://github.com/user-attachments/assets/c07c6b5d-685f-4a2a-ad0d-a8ef79edaf0a" />


### 4. 📈 Evaluation & Optimization

* Metrics: MAE, MSE, RMSE, R² Score
* Compare actual vs predicted visually
* Analyze errors under high-volatility periods
* Fine-tune for optimal performance



---

## 📈 Results Summary

| Model             | MAE     | RMSE    | R² Score |
|------------------|---------|---------|----------|
| Linear Regression | 0.015   | 0.021   | 0.9714   |
| Random Forest     | 0.022   | 0.029   | 0.9461   |
| XGBoost           | 0.021   | 0.027   | 0.9505   |
| Decision Tree     | 0.029   | 0.039   | 0.9024   |
| SVR               | 0.030   | 0.036   | 0.9141   |
| LSTM              | 0.035   | 0.043   | 0.8618   |
| GRU               | 0.030   | 0.038   | 0.8959   |
| Ensemble (LSTM+RF)| 0.022   | 0.029   | 0.9387   |
| **Prophet**       | 11.5816 | 15.0720 | 0.9100   |






