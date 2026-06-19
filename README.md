# Fuel Price Prediction Dashboard

An AI-powered Fuel Price Prediction Dashboard built using Streamlit, XGBoost, and Yahoo Finance data. The application forecasts future crude oil prices by leveraging machine learning and time-series feature engineering techniques.

## Overview

This project predicts the next trading day's crude oil price using historical market data and advanced machine learning models. It provides interactive visualizations, model evaluation metrics, and real-time forecasts through a user-friendly dashboard.

## Features

* Real-time crude oil price data from Yahoo Finance
* Interactive Streamlit dashboard
* XGBoost regression model
* Feature engineering with lag variables and technical indicators
* Actual vs Predicted price visualization
* Model evaluation using MAE and R² Score
* Next-day price prediction
* Adjustable training parameters

## Technology Stack

### Frontend

* Streamlit
* Plotly

### Data Processing

* Pandas
* NumPy

### Machine Learning

* XGBoost
* Scikit-learn

### Data Source

* Yahoo Finance API (yfinance)

## Dataset

The application uses WTI Crude Oil Futures data (Ticker: CL=F) obtained directly from Yahoo Finance.

## Machine Learning Pipeline

1. Download historical crude oil prices
2. Generate lag-based features
3. Create technical indicators
4. Train XGBoost Regressor
5. Evaluate performance
6. Forecast future prices

## Performance Metrics

The dashboard reports:

* Mean Absolute Error (MAE)
* R² Score

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/fuel-price-prediction-dashboard.git
cd fuel-price-prediction-dashboard
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run streamlit_app.py
```

## Project Structure

```text
Fuel-Price-Prediction-Dashboard/
│
├── streamlit_app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── assets/
│   ├── architecture.png
│   ├── dashboard.png
│   └── logo.png
│
└── screenshots/
    └── dashboard_demo.png
```

## Future Improvements

* LSTM-based forecasting
* Multi-day predictions
* Brent Crude Oil forecasting
* Fuel price conversion to INR/Liter
* Hyperparameter optimization
* Deployment on Streamlit Cloud

## Author

Alwyn Robin

## License

This project is licensed under the MIT License.
