import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# 1. DATA LOADING (with synthetic fallback)
# ---------------------------------------------------------------------------

def generate_synthetic_sales(n_days=730, start="2024-01-01"):
    """Create a synthetic daily sales dataset with trend, weekly seasonality,
    yearly seasonality, and noise — useful for testing the pipeline."""
    rng = pd.date_range(start=start, periods=n_days, freq="D")
    t = np.arange(n_days)

    trend = 200 + 0.15 * t
    weekly_season = 30 * np.sin(2 * np.pi * t / 7)
    yearly_season = 50 * np.sin(2 * np.pi * t / 365.25)
    noise = np.random.normal(0, 15, n_days)

    sales = trend + weekly_season + yearly_season + noise
    sales = np.clip(sales, 0, None)

    df = pd.DataFrame({"Date": rng, "Sales": sales})
    return df


def load_data(csv_path=None, date_col="Date", sales_col="Sales"):
    if csv_path:
        df = pd.read_csv(csv_path)
        df = df.rename(columns={date_col: "Date", sales_col: "Sales"})
    else:
        print("No CSV provided — generating synthetic sample sales data.")
        df = generate_synthetic_sales()
    return df


# ---------------------------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------------------------

def clean_data(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Fill missing dates (important for time-series continuity)
    full_range = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    df = df.set_index("Date").reindex(full_range)
    df.index.name = "Date"

    # Interpolate missing sales values
    df["Sales"] = df["Sales"].interpolate(method="linear")

    # Remove negative values (invalid for sales/demand)
    df["Sales"] = df["Sales"].clip(lower=0)

    # Handle outliers using IQR capping
    q1, q3 = df["Sales"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df["Sales"] = df["Sales"].clip(lower=lower, upper=upper)

    df = df.reset_index()
    return df


# ---------------------------------------------------------------------------
# 3. TIME-BASED FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def create_features(df):
    df = df.copy()
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["dayofyear"] = df["Date"].dt.dayofyear
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)

    # Lag features (previous sales values)
    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"] = df["Sales"].shift(lag)

    # Rolling averages (smoothed trend signals)
    for window in [7, 30]:
        df[f"rolling_mean_{window}"] = df["Sales"].shift(1).rolling(window).mean()

    df = df.dropna().reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT (chronological — never shuffle time series)
# ---------------------------------------------------------------------------

def train_test_split_time(df, test_size=0.2):
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


# ---------------------------------------------------------------------------
# 5. MODELING (regression-based forecasting)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "year", "month", "day", "dayofweek", "is_weekend", "dayofyear", "weekofyear",
    "lag_1", "lag_7", "lag_14", "lag_30", "rolling_mean_7", "rolling_mean_30",
]


def train_models(train, test):
    X_train, y_train = train[FEATURE_COLS], train["Sales"]
    X_test, y_test = test[FEATURE_COLS], test["Sales"]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {"model": model, "preds": preds}

    return results, y_test


# ---------------------------------------------------------------------------
# 6. MODEL EVALUATION
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100

    print(f"\n--- {label} ---")
    print(f"MAE   : {mae:,.2f}")
    print(f"RMSE  : {rmse:,.2f}")
    print(f"MAPE  : {mape:,.2f}%")
    print(f"R^2   : {r2:,.3f}")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


# ---------------------------------------------------------------------------
# 7. FUTURE FORECASTING (extend beyond available data)
# ---------------------------------------------------------------------------

def forecast_future(df, model, periods=30):
    """Iteratively forecast `periods` days ahead using the trained model,
    feeding each prediction back in to compute the next lag/rolling features."""
    history = df.copy()
    future_dates = pd.date_range(history["Date"].max() + pd.Timedelta(days=1), periods=periods)
    future_preds = []

    for date in future_dates:
        row = {
            "Date": date,
            "year": date.year,
            "month": date.month,
            "day": date.day,
            "dayofweek": date.dayofweek,
            "is_weekend": int(date.dayofweek in [5, 6]),
            "dayofyear": date.dayofyear,
            "weekofyear": int(date.isocalendar().week),
        }
        row["lag_1"] = history["Sales"].iloc[-1]
        row["lag_7"] = history["Sales"].iloc[-7]
        row["lag_14"] = history["Sales"].iloc[-14]
        row["lag_30"] = history["Sales"].iloc[-30]
        row["rolling_mean_7"] = history["Sales"].iloc[-7:].mean()
        row["rolling_mean_30"] = history["Sales"].iloc[-30:].mean()

        X_next = pd.DataFrame([row])[FEATURE_COLS]
        pred = model.predict(X_next)[0]
        row["Sales"] = pred
        future_preds.append(row)

        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)

    return pd.DataFrame(future_preds)[["Date", "Sales"]]


# ---------------------------------------------------------------------------
# 8. BUSINESS-FRIENDLY VISUALIZATION
# ---------------------------------------------------------------------------

def plot_results(df, test, results, future_df, best_model_name):
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(df["Date"], df["Sales"], label="Historical Sales", color="#2c3e50", linewidth=1)
    ax.plot(test["Date"], results[best_model_name]["preds"],
            label=f"Test Predictions ({best_model_name})", color="#e67e22", linewidth=2)
    ax.plot(future_df["Date"], future_df["Sales"],
            label="Future Forecast", color="#27ae60", linestyle="--", linewidth=2)

    ax.axvline(test["Date"].iloc[0], color="gray", linestyle=":", alpha=0.7)
    ax.set_title("Sales & Demand Forecast", fontsize=15, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("sales_forecast.png", dpi=150)
    print("\nSaved chart -> sales_forecast.png")


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sales & Demand Forecasting Pipeline")
    parser.add_argument("--csv", type=str, default=None, help="Path to sales CSV file")
    parser.add_argument("--date_col", type=str, default="Date")
    parser.add_argument("--sales_col", type=str, default="Sales")
    parser.add_argument("--forecast_days", type=int, default=30)
    args = parser.parse_args()

    # 1-2. Load & clean
    raw = load_data(args.csv, args.date_col, args.sales_col)
    clean = clean_data(raw)

    # 3. Feature engineering
    featured = create_features(clean)

    # 4. Split
    train, test = train_test_split_time(featured)

    # 5. Train models
    results, y_test = train_models(train, test)

    # 6. Evaluate & pick best model
    scores = {}
    for name, res in results.items():
        scores[name] = evaluate(y_test, res["preds"], label=name)

    best_model_name = min(scores, key=lambda k: scores[k]["RMSE"])
    best_model = results[best_model_name]["model"]
    print(f"\nBest model: {best_model_name}")

    # 7. Forecast future demand
    future_df = forecast_future(featured, best_model, periods=args.forecast_days)
    print(f"\nNext {args.forecast_days} days forecast (head):")
    print(future_df.head())

    # 8. Visualization
    plot_results(clean, test, results, future_df, best_model_name)

    # Deliverable: save forecast to CSV
    future_df.to_csv("future_sales_forecast.csv", index=False)
    print("Saved forecast -> future_sales_forecast.csv")


if __name__ == "__main__":
    main()
