# 📈 End-to-End Retail Sales Forecasting & Deployment

An end-to-end **time-series forecasting project** built on Rossmann retail sales data.
The project covers **data analysis, forecasting, evaluation, visualization, and deployment**, following industry-standard practices.


## 🚀 Project Overview

Retail businesses rely heavily on accurate demand forecasting for:

* inventory planning
* workforce management
* promotion strategy

This project analyzes historical daily sales data from Rossmann stores and builds multiple forecasting models to predict future sales.
The final solution is deployed as an **interactive Streamlit application** for real-time forecasting.


## 🧠 Business Problem

> How can we accurately forecast daily retail sales while accounting for seasonality, promotions, and store-level variations?


## 🎯 Objectives

* Analyze sales trends and seasonality
* Measure the impact of promotions and holidays
* Build and compare classical and ML-based forecasting models
* Select the best-performing model using proper time-based validation
* Communicate insights via Tableau dashboards
* Deploy forecasts through a Streamlit web app


## 🗂️ Project Structure

```
rossmann-sales-forecasting/
│
├── app/                 # Streamlit application
│   ├── app.py
│   └── utils.py
│
├── data/
│   └── processed/
│       └── rossmann_features.csv
│
├── models/
│   ├── sarima/
│   │   └── sarima_model.pkl
│   └── prophet/
│       └── prophet_model.pkl
│
├── notebooks/           # Step-by-step notebooks
├── tableau/             # Tableau dashboards
├── screenshots/         # Project screenshots
├── README.md
└── requirements.txt
```

## 📊 Exploratory Data Analysis (EDA)

### Key Findings

* Strong **weekly and yearly seasonality**
* Sales peak toward **end of the week and end of the year**
* **Promotions significantly increase sales**
* Store characteristics (type & assortment) affect sales behavior


## 📸 Project Screenshots

### Streamlit Forecasting App

![Streamlit App Home](screenshots/streamlit_home.png)

![SARIMA Forecast](screenshots/forecast_sarima.png)

![Prophet Forecast](screenshots/forecast_prophet.png)


### Tableau Dashboards

![Sales Trend & Seasonality](screenshots/tableau_dashboard_1.png)

![Promotion & Holiday Impact](screenshots/tableau_dashboard_2.png)


## 🧪 Forecasting Models

All models were evaluated using a **time-based train–test split** to avoid data leakage.

### Models Implemented

* Naive baseline
* Moving Average
* ARIMA
* SARIMA
* Prophet
* LSTM (exploratory)

### Evaluation Metrics

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* MAPE (Mean Absolute Percentage Error)


## 🏆 Model Performance Summary

| Model          | RMSE        | MAPE       |
| -------------- | ----------- | ---------- |
| **SARIMA**     | **1123.91** | **14.27%** |
| Prophet        | 1133.39     | 14.84%     |
| LSTM           | 1394.64     | 26.25%     |
| ARIMA          | 2223.61     | 21.95%     |
| Moving Average | 2298.02     | 27.55%     |
| Naive          | 5102.25     | 75.01%     |

### ✅ Best Model: **SARIMA**

* Lowest error
* Explicitly models weekly seasonality
* Stable and interpretable
* Suitable for production deployment

## 🔮 Streamlit Application

The Streamlit app allows users to:

* Select a store
* Choose a forecast horizon (7–42 days)
* Generate forecasts using SARIMA or Prophet
* View confidence intervals
* Download predictions as CSV

> The app operationalizes the forecasting models into a usable business tool.


## 🛠️ Tools & Technologies

* **Python**: Pandas, NumPy, Statsmodels, Prophet
* **Visualization**: Matplotlib, Tableau
* **Deep Learning**: TensorFlow (LSTM exploration)
* **Deployment**: Streamlit
* **Version Control**: Git & GitHub


## ⚠️ Limitations

* Forecasting is demonstrated on a representative store
* External factors (weather, macroeconomic data) are not included
* Long-term promotions are simplified


## 🔮 Future Improvements

* Cluster-based forecasting for groups of stores
* Include external regressors (weather, holidays)
* Rolling-window cross-validation
* Full cloud deployment with automated retraining


## 👤 Author

**Rupesh Sharan**
GitHub: [https://github.com/RupeshSharan](https://github.com/RupeshSharan)


## 📌 Final Note

This project demonstrates a **complete real-world forecasting pipeline**, from raw data to deployment, following best practices in time-series analysis and machine learning.

---
