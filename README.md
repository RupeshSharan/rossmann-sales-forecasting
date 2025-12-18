End-to-End Retail Sales Forecasting using Time-Series Models (Rossmann Stores)

🏪 Business Context

Rossmann operates a large chain of drug stores across multiple regions.
Accurate sales forecasting is critical for inventory planning, staffing, and promotion management.
Due to seasonal demand patterns, promotional campaigns, and store-level differences, forecasting daily sales is a non-trivial problem.

❓ Business Problem

The objective of this project is to analyze historical daily sales data from Rossmann stores and build reliable time-series forecasting models that can predict future sales.
The forecasts will help support data-driven decisions related to inventory control, promotion planning, and operational efficiency.

🎯 Project Objectives

Analyze sales trends and seasonal patterns across time
Evaluate the impact of promotions and holidays on sales
Build and compare multiple forecasting models (baseline, SARIMA, Prophet, LSTM)
Evaluate model performance using appropriate error metrics
Provide business-oriented insights through interactive Tableau dashboards

🔍 Forecasting Scope

This project focuses on:
Daily sales forecasting
A selected representative store for detailed modeling
Aggregated sales trends for overall business insight

⚠️ Assumptions & Constraints

Sales are only recorded for days when stores are open
Missing dates correspond to store closures or non-operational days
External macroeconomic variables are not included
Forecasting is based solely on historical patterns and available promotional data

✅ Success Criteria

Forecast models outperform naive baseline approaches
Models demonstrate stable performance across seasonal periods
Insights are interpretable and actionable for business stakeholders

🛠️ Tools & Technologies

Python (Pandas, NumPy, Statsmodels, Prophet, TensorFlow)
Tableau (Exploratory and forecast visualization)
Git & GitHub (version control)
Jupyter Notebook (experimentation)

### Key Observations from EDA

- Sales show strong seasonal patterns, with noticeable peaks toward the end of the year, likely driven by holiday demand and festive promotions.
- Weekly seasonality is evident, with higher average sales occurring toward the end of the week, reflecting increased customer activity during weekends.
- Promotion days consistently outperform non-promotion days in terms of average sales, highlighting the significant impact of promotional campaigns on demand.


During data preparation, non-operational days (when stores were closed) were removed, as zero sales on these days do not represent actual demand. Holiday and promotion indicators were preserved to capture demand fluctuations driven by special events. Missing competition-related values were handled using business logic, and new features such as competition age and long-term promotion indicators were created to improve interpretability and forecasting performance.

This analysis leverages historical daily sales data from over 1,100 Rossmann stores to uncover demand patterns and forecast future sales.