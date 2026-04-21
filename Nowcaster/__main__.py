import argparse
import pandas as pd
import numpy as np
from .config import series_ids, PUBLICATION_LAGS
from .Data import fetch_series, stationizer
from .features import lag_adj_features, nowcast_vector
from .models import build_model
from .evaluation import nowcast, extract_importances


def main():
    parser = argparse.ArgumentParser(description="GDP Nowcaster")
    parser.add_argument("--start", default="1990-01-01",
                        help="Start date for FRED data fetch")
    parser.add_argument("--as-of", default=None,
                        help="As-of date for the nowcast (YYYY-MM-DD), defaults to today")
    args = parser.parse_args()

    # 1 ── Fetch data from FRED
    print("Fetching data from FRED")
    raw = fetch_series(series_ids, start=args.start)

    # 2 ── Separate GDP (quarterly target) from monthly indicators
    gdp_quarterly = raw["gdp"].resample("QE").last().dropna()
    monthly = raw.drop(columns=["gdp"]).resample("ME").last()

    # 3 ── Stationarize monthly indicators
    DIFF_COLS = ["unrate", "fed_funds", "yield_10y", "spread_10y2y",
                 "sentiment", "trade_balance"]
    LOG_DIFF_COLS = ["inflation", "indus_prod", "nonf_proll", "retail_trade",
                     "ppi_ac", "lumber_ppi", "init_claims"]

    print("\nStationarizing monthly series")
    monthly_stationary = stationizer(monthly, DIFF_COLS, LOG_DIFF_COLS)

    # 4 ── Build lag-adjusted training set
    print("\nBuilding lag-adjusted feature matrix")
    X_df, y = lag_adj_features(monthly_stationary, gdp_quarterly, PUBLICATION_LAGS)

    # 5 ── Train/test split
    X = X_df.values
    y_vals = y.values
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_vals[:split], y_vals[split:]

    print(f"  {X_df.shape[0]} quarters x {X_df.shape[1]} features")
    print(f"  {X_df.index[0].date()} -> {X_df.index[-1].date()}")
    print(f"  Train: {len(X_train)} quarters | Test: {len(X_test)} quarters")

    # 6 ── Train and evaluate each model
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    models_to_run = ["elasticnet", "random_forest", "gradient_boosting"]
    best_model_name = None
    best_pipeline = None
    best_test_r2 = -np.inf

    for model_name in models_to_run:
        print(f"\n{'─' * 50}")
        print(f"  {model_name.upper()}")
        print(f"{'─' * 50}")

        pipe = build_model(model_name)
        pipe.fit(X_train, y_train)

        train_r2 = pipe.score(X_train, y_train)
        test_r2 = pipe.score(X_test, y_test)
        y_pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"  Train R²  : {train_r2:.4f}")
        print(f"  Test R²   : {test_r2:.4f}")
        print(f"  Test MAE  : {mae:.6f}")
        print(f"  Test RMSE : {rmse:.6f}")

        # Feature importances
        importances = extract_importances(pipe, X_df.columns)
        if not importances.empty:
            print(f"\n  Top features:")
            for feat, imp in importances.head(5).items():
                print(f"    {feat:<20} {imp:.4f}")

        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_model_name = model_name
            best_pipeline = pipe

    print(f"\n{'=' * 50}")
    print(f"  Best model: {best_model_name} (Test R² = {best_test_r2:.4f})")
    print(f"{'=' * 50}")

    # 7 ─ Refit best model on all data for nowcast
    best_pipeline.fit(X, y_vals)

    # 8 ─ Produce nowcast for current quarter
    as_of = pd.Timestamp(args.as_of) if args.as_of else None
    X_now = nowcast_vector(monthly_stationary, PUBLICATION_LAGS, as_of_date=as_of)

    prediction = nowcast(best_pipeline, X_now, X_df.columns)
    annualized = (np.exp(prediction * 4) - 1) * 100  # log quarterly → annualized %

    print(f"\n{'=' * 50}")
    print(f"  Nowcast GDP growth (log q/q): {prediction:.4f}")
    print(f"  Annualized approx:            {annualized:.2f}%")
    print(f"  Model used:                   {best_model_name}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()