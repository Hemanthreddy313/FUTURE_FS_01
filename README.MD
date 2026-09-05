# Sales & Demand Forecasting Pipeline

A command-line pipeline that cleans historical daily sales data, trains regression
models, evaluates them, and forecasts future demand.

## What it does

1. **Load data** — reads a CSV of daily sales, or generates synthetic sample data
   if no file is given.
2. **Clean data** — fills missing dates, interpolates gaps, clips negative values,
   and caps outliers using the IQR method.
3. **Feature engineering** — builds calendar features (year, month, day of week,
   weekend flag, day/week of year) plus lag features (1/7/14/30 days) and rolling
   averages (7/30 days).
4. **Train/test split** — chronological 80/20 split (no shuffling, since this is
   time-series data).
5. **Train models** — fits both Linear Regression and Random Forest.
6. **Evaluate** — reports MAE, RMSE, MAPE, and R² for each model; picks the one
   with the lowest RMSE.
7. **Forecast future demand** — iteratively predicts N days ahead, feeding each
   prediction back in as input for the next day's lag/rolling features.
8. **Visualize & export** — saves a chart (`sales_forecast.png`) and the forecast
   (`future_sales_forecast.csv`).

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

Run with your own data:

```bash
python forecast.py --csv your_sales.csv --date_col Date --sales_col Sales --forecast_days 30
```

Run with synthetic sample data (no CSV needed):

```bash
python forecast.py
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--csv` | `None` | Path to a sales CSV file. If omitted, synthetic data is generated. |
| `--date_col` | `Date` | Name of the date column in your CSV. |
| `--sales_col` | `Sales` | Name of the sales column in your CSV. |
| `--forecast_days` | `30` | Number of days to forecast into the future. |

Your CSV just needs two columns: a date column and a sales/quantity column.

## Outputs

- **`sales_forecast.png`** — chart showing historical sales, test-set predictions
  from the best model, and the future forecast (dashed green).
- **`future_sales_forecast.csv`** — `Date, Sales` for each forecasted day.
- **Console output** — MAE/RMSE/MAPE/R² for both models, the chosen best model,
  and a preview of the forecast.

## Notes on the current results

Based on the attached chart and forecast file:

- The pipeline picked **Linear Regression** as the best model (shown in the
  orange test-prediction line).
- Historical sales run from Jan 2024 through Dec 2025, with a clear upward
  trend, strong weekly seasonality (the sawtooth pattern), and a yearly cycle.
- The forecast starts **2025-12-31** and covers the next 30 days, projecting
  sales in the **~275–355** range, continuing the recent upward trend and
  weekly oscillation.

## Limitations to keep in mind

- Because forecasting is iterative (each predicted day feeds the next day's
  lag features), errors can compound over longer horizons — accuracy degrades
  the further out you forecast.
- Linear Regression won here on RMSE, but check MAPE/R² too — a model can win
  on one metric and be a close second on others.
- IQR-based outlier capping on the whole series can flatten genuine demand
  spikes (e.g. holidays, promotions) if the data isn't otherwise adjusted for them.
