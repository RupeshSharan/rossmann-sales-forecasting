import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_data, load_sarima, load_prophet

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
st.markdown(
    """
    **Business-ready sales forecasting application**  
    Built using **SARIMA** (best accuracy) and **Prophet** (business-friendly) models  
    on Rossmann retail sales data.
    """
)

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
    max_value=42,
    step=7
)

model_choice = st.sidebar.radio(
    "Forecasting Model",
    ["SARIMA (Best Accuracy)", "Prophet (Business Friendly)"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "SARIMA is recommended for deployment due to lowest forecast error.\n\n"
    "Prophet is useful for interpretability and stakeholder communication."
)

# --------------------------------------------------
# FILTER STORE DATA
# --------------------------------------------------
store_df = (
    df[df["Store"] == store_id]
    .sort_values("Date")
    [["Date", "Sales"]]
    .set_index("Date")
)

# --------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📊 Historical Sales — Store {store_id}")
    st.line_chart(store_df)

with col2:
    st.subheader("📌 Store Summary")
    st.metric("Average Daily Sales", f"{int(store_df['Sales'].mean()):,}")
    st.metric("Maximum Daily Sales", f"{int(store_df['Sales'].max()):,}")
    st.metric("Data Points", len(store_df))

st.divider()

# --------------------------------------------------
# FORECAST SECTION
# --------------------------------------------------
st.subheader("🔮 Sales Forecast with Confidence Intervals")

if st.button("Generate Forecast"):

    with st.spinner("Generating forecast..."):

        # =======================
        # SARIMA FORECAST
        # =======================
        if "SARIMA" in model_choice:
            model = load_sarima()

            forecast_res = model.get_forecast(steps=horizon)
            mean_forecast = forecast_res.predicted_mean
            conf_int = forecast_res.conf_int()

            future_dates = pd.date_range(
                start=store_df.index[-1] + pd.Timedelta(days=1),
                periods=horizon
            )

            forecast_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted Sales": mean_forecast.values,
                "Lower Bound": conf_int.iloc[:, 0].values,
                "Upper Bound": conf_int.iloc[:, 1].values
            })

        # =======================
        # PROPHET FORECAST
        # =======================
        else:
            model = load_prophet()

            prophet_df = store_df.reset_index().rename(
                columns={"Date": "ds", "Sales": "y"}
            )

            future = model.make_future_dataframe(periods=horizon)
            forecast = model.predict(future)

            forecast_df = forecast[
                ["ds", "yhat", "yhat_lower", "yhat_upper"]
            ].tail(horizon)

            forecast_df.columns = [
                "Date",
                "Predicted Sales",
                "Lower Bound",
                "Upper Bound"
            ]

    # --------------------------------------------------
    # PLOT WITH CONFIDENCE INTERVAL
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        store_df.index,
        store_df["Sales"],
        label="Historical Sales",
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
        color="blue",
        alpha=0.2,
        label="Confidence Interval"
    )

    ax.set_title("Sales Forecast with Uncertainty")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()

    st.pyplot(fig)

    # --------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------
    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=forecast_df.to_csv(index=False),
        file_name=f"store_{store_id}_forecast_with_ci.csv",
        mime="text/csv"
    )

st.divider()

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.caption(
    "Forecasts are estimates and include uncertainty bands. "
    "Built with Python, SARIMA, Prophet & Streamlit."
)
