import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_data, load_sarima, load_prophet
from clustering import cluster_stores
from simulation import apply_holiday_uplift

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("📈 Retail Sales Forecasting Dashboard")
st.markdown("Business-ready forecasting system for retail demand planning.")

st.divider()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = load_data()

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("⚙️ Forecast Settings")

store_id = st.sidebar.selectbox(
    "Select Store",
    sorted(df["Store"].unique())
)

horizon = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=7,
    max_value=365,
    step=7,
    value=30
)

history_window = st.sidebar.slider(
    "Show last N days of history",
    min_value=30,
    max_value=365,
    step=30,
    value=90
)

model_choice = st.sidebar.radio(
    "Forecasting Model",
    ["SARIMA", "Prophet"]
)

simulate_holiday = st.sidebar.checkbox("Simulate Holiday Impact")

holiday_uplift = st.sidebar.slider(
    "Holiday Sales Uplift (%)",
    min_value=5,
    max_value=50,
    step=5,
    value=15
) if simulate_holiday else 0

if horizon > 180:
    st.sidebar.warning("⚠️ Long forecast horizons increase uncertainty.")

# --------------------------------------------------
# FILTER STORE DATA
# --------------------------------------------------
store_df = (
    df[df["Store"] == store_id]
    .sort_values("Date")
    [["Date", "Sales", "Promo"]]
    .set_index("Date")
)

recent_history = store_df.tail(history_window)
last_date = store_df.index[-1]

# --------------------------------------------------
# STORE CLUSTERING
# --------------------------------------------------
clusters = cluster_stores(df)
store_cluster = clusters.loc[
    clusters["Store"] == store_id, "Cluster"
].values[0]

# --------------------------------------------------
# LAYOUT
# --------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📊 Recent Sales — Store {store_id}")
    st.line_chart(recent_history["Sales"])

with col2:
    st.subheader("🏪 Store Insights")
    st.metric("Avg Daily Sales", f"{int(store_df['Sales'].mean()):,}")
    st.metric("Store Cluster", int(store_cluster))
    st.metric("Last Data Date", last_date.strftime("%Y-%m-%d"))

st.divider()

# --------------------------------------------------
# FORECAST SECTION
# --------------------------------------------------
st.subheader("🔮 Sales Forecast")

st.caption(
    f"Forecast starts from **{last_date.strftime('%Y-%m-%d')}** "
    f"and extends **{horizon} days into the future**."
)

if st.button("Generate Forecast"):

    with st.spinner("Generating forecast..."):

        if model_choice == "SARIMA":
            model = load_sarima()
            forecast_res = model.get_forecast(steps=horizon)
            mean_forecast = forecast_res.predicted_mean
            conf_int = forecast_res.conf_int()

            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon
            )

            forecast_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted Sales": mean_forecast.values,
                "Lower Bound": conf_int.iloc[:, 0].values,
                "Upper Bound": conf_int.iloc[:, 1].values
            })

        else:
            model = load_prophet()
            future = model.make_future_dataframe(periods=horizon)
            forecast = model.predict(future)

            forecast_df = forecast[
                ["ds", "yhat", "yhat_lower", "yhat_upper"]
            ].tail(horizon)

            forecast_df.columns = [
                "Date", "Predicted Sales", "Lower Bound", "Upper Bound"
            ]

        # Holiday simulation
        if simulate_holiday:
            forecast_df = apply_holiday_uplift(
                forecast_df, holiday_uplift
            )
            st.success(f"Holiday simulation applied (+{holiday_uplift}%)")

    # --------------------------------------------------
    # PLOT
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        recent_history.index,
        recent_history["Sales"],
        label="Recent Historical Sales",
        color="black"
    )

    ax.plot(
        forecast_df["Date"],
        forecast_df["Predicted Sales"],
        label="Forecast",
        color="blue"
    )

    ax.fill_between(
        forecast_df["Date"],
        forecast_df["Lower Bound"],
        forecast_df["Upper Bound"],
        alpha=0.2,
        label="Confidence Interval"
    )

    ax.set_title("Sales Forecast with Uncertainty")
    ax.legend()

    st.pyplot(fig)

    # --------------------------------------------------
    # FORECAST SUMMARY TABLE (FIXED)
    # --------------------------------------------------
    st.subheader("📋 Forecast Summary")

    summary_df = pd.DataFrame({
        "Metric": ["Mean Forecast", "Avg Lower Bound", "Avg Upper Bound"],
        "Value": [
            forecast_df["Predicted Sales"].mean(),
            forecast_df["Lower Bound"].mean(),
            forecast_df["Upper Bound"].mean()
        ]
    })

    st.dataframe(
        summary_df.style.format({"Value": "{:,.0f}"})
    )

    st.download_button(
        "⬇️ Download Forecast CSV",
        forecast_df.to_csv(index=False),
        f"store_{store_id}_forecast.csv"
    )
