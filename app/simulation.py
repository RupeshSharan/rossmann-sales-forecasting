def apply_holiday_uplift(forecast_df, uplift_pct):
    simulated = forecast_df.copy()

    factor = 1 + uplift_pct / 100

    simulated["Predicted Sales"] *= factor
    simulated["Lower Bound"] *= factor
    simulated["Upper Bound"] *= factor

    return simulated
