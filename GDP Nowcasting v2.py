import os
from dotenv import load_dotenv
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from fredapi import Fred
from concurrent.futures import ThreadPoolExecutor, as_completed


# Constants

adf_pval_threshold = 0.05
max_diffs = 3
gdp_lag_days = 30
zero_threshold = 1e-5

series_ids = {
    'gdp': 'GDP',
    'inflation': 'CPIAUCSL',
    'indus_prod': 'INDPRO',
    'nonf_proll': 'PAYEMS',
    'retail_trade': 'RSAFS',
    'unrate': 'UNRATE',
    'init_claims': 'ICSA',
    'ppi_ac': 'PPIACO',
    'fed_funds': 'FEDFUNDS',
    'yield_10y': 'GS10',
    'spread_10y2y': 'T10Y2Y',
    'sentiment': 'UMCSENT',
    'trade_balance': 'BOPGSTB',
    'lumber_ppi': 'WPU081'
}
# Lag of data publication in months
# Data is available in month M + publication lag
PUBLICATION_LAGS = {
    'inflation': 2,
    'indus_prod': 1,
    'nonf_proll': 1,
    'retail_trade': 1,
    'unrate': 1,
    'init_claims': 0,
    'ppi_ac': 1,
    'fed_funds': 0,
    'yield_10y': 0,
    'spread_10y2y': 0,
    'sentiment': 1,
    'trade_balance': 2,
    'lumber_ppi': 1
}

# Separates time series for stationarization, DIFF_COLS will be differentiated and LOG_DIFF_COLS will be log differentiated (maybe looks to automate process)

DIFF_COLS    = ['unrate', 'fed_funds', 'yield_10y', 'spread_10y2y',
                 'sentiment', 'trade_balance']
LOG_DIFF_COLS = ['inflation', 'indus_prod', 'nonf_proll', 'retail_trade',
                 'ppi_ac', 'lumber_ppi', 'init_claims']


# --------------------------------Data collection--------------------------------

def fetch_series(series_ids:dict, start: str = '1990-01-01') -> pd.DataFrame:


    load_dotenv()
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))

    def _fetch(name, sid):
        return name, fred.get_series(sid, observation_start=start)
    results = {}

    with ThreadPoolExecutor(max_workers=min(8, len(series_ids))) as executor:
        futures = {
            executor.submit(_fetch, n, s): n for n, s in series_ids.items()}
        for future in as_completed(futures):
            try:
                name, series = future.result()
                results[name] = series
            except Exception as e:
                print(f"Failed to fetch {futures[future]}: {e}")
    return pd.DataFrame(results)

# --------------------------------Data visualisation--------------------------------

def plot_data(series_df: pd.DataFrame, ncols: int = 4):
    
    # Plots series in a grid
    fig, axes = plt.subplots(nrows=math.ceil(len(series_df.columns)/ncols), ncols=ncols, figsize=(15,12))
    axes = axes.flatten()
    for ax, col in zip(axes, series_df.columns):
        series = series_df[col].dropna()
        ax.plot(series.index, series)
        ax.set_title(col)
        ax.set_xlim(series_df.index[0], series_df.index[-1])
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



from statsmodels.tsa.stattools import adfuller



##########################################
def stationizer(data: pd.DataFrame, DIFF_COLS: list, LOG_DIFF_COLS: list):

    """
    Stationarizes the indicators that were used to stationarity.
    We do log diffing once, and do differentiating up to 3 times (max_diffs), should be enough for most of our data
    """
    out = data.copy()

    for col in out.columns:
        n_diffs = 0
        series = out[col].dropna()

        if len(series) < 20:
            print(f"{col}: too few observations, skipping")
            continue

        pval = adfuller(series)[1] # adfuller returns multiple outputs, [1] is p-value
        
        while pval > adf_pval_threshold and n_diffs < max_diffs:
            if col in LOG_DIFF_COLS:
                out[col] = np.log(out[col]).diff()
                pval = adfuller(out[col].dropna())[1]
                break

            elif col in DIFF_COLS:
                out[col] = out[col].diff()
                n_diffs += 1
                pval = adfuller(out[col].dropna())[1]
                
            else:
                print(f"{col} not in either list, skipping")
                continue

        status = 'Stationary' if pval < adf_pval_threshold else 'NOT stationary'
        print(f"  {col}: p={pval:.3f}  {status}  (diffs={n_diffs})")

    return out.dropna()


# --------------------------------Data processing for nowcasting--------------------------------

# Objective is to adjust the feature matrix for its lags (time when data is available relative to reporting period) to build a model that simulates real time casting

def _available_months(quarter_end: pd.Timestamp,
                      lag_months: int,
                      as_of: pd.Timestamp) -> list:
    # backbone of whole thing, will give available month of each feature in the quarter
    # month is available if release date is on or before as_of

    q_month_ends = pd.date_range(end=quarter_end, periods = 3, freq='ME')
    available = [
        m for m in q_month_ends # Checks if data available before each EOM within a quarter
        if m + pd.DateOffset(months=lag_months) <= as_of # Limit to as_of timestamp
    ]
    return available # Returns the months the features are published for

def lag_adj_features(monthly_stationary: pd.DataFrame,
                     gdp_quarterly: pd.Series,
                     pub_lags: dict) -> tuple[pd.DataFrame, pd.Series]:
    # Lag adjusts training feature matrix
    # We have to NOW cast, so reconstuct feature vectors as it looks like before GDP release
    # We use gdp log growth rate

    gdp_growth = np.log(gdp_quarterly).diff().dropna()
    gdp_growth.name = 'gdp_growth'

    records = []

    for q_end in gdp_growth.index: # loop that iterates available_months on quarter sequence
        as_of = q_end + pd.DateOffset(days=gdp_lag_days) # Simulates when GDP becomes known (30 days later roughly), only data before as_of can be used for the predictions, otherwise makes no sense 
        row = {'quarter': q_end}

        for col, lag in pub_lags.items(): #Pub lags is our dict of publication lags with name: lag
            if col not in monthly_stationary.columns:
                continue
            months = _available_months(q_end, lag, as_of)
            if not months:
                row[col] = np.nan
                continue
            vals = monthly_stationary.loc[
                monthly_stationary.index.isin(months), col
            ].dropna()
            row[col] = vals.mean() if len(vals) > 0 else np.nan
 
        records.append(row)
 
    feat_df = pd.DataFrame(records).set_index('quarter')        
    combined = feat_df.join(gdp_growth, how='inner').dropna()
    y = combined.pop('gdp_growth')

    return combined, y


# Vector that will be used for testing
def nowcast_vector(monthly_stationary: pd.DataFrame,
                   pub_lags: dict,
                   as_of_date: pd.Timestamp | None = None) -> pd.Series:
    if as_of_date is None:
        as_of_date = pd.Timestamp.today().normalize() 

    q_end = as_of_date.to_period('Q').to_timestamp('Q')
    q_label = as_of_date.to_period('Q')

    print(f"\nNowcast target : {q_label}  (quarter ends {q_end.date()})")
    print(f"As-of date     : {as_of_date.date()}")
    print(f"{'Series':<20} {'Available months':<35} {'Value':>10}")
    print("-" * 68)

    row = {}
    for col, lag in pub_lags.items():
        if col not in monthly_stationary.columns:
            continue
        months = _available_months(q_end, lag, as_of_date)
        if not months: # Checks if empty of null (so if there are any available features)
            row[col] = np.nan
            month_str = "none yet"
        else:
            vals = monthly_stationary.loc[
                monthly_stationary.index.isin(months), col
            ].dropna()
            row[col] = vals.mean() if len(vals) > 0 else np.nan
            month_str = ', '.join(m.strftime('%b') for m in months)

        val_str = f"{row[col]:.4f}" if not np.isnan(row.get(col, np.nan)) else "NaN"
        print(f"  {col:<18} {month_str:<35} {val_str:>10}")
 
    return pd.Series(row, name=q_end)






def splits(data: pd.DataFrame, target_col: str) -> tuple:
    features = data.columns.drop(target_col)

    X = data[features].values
    y = data[target_col].values

    return X, y, features

def elasticnet(X: np.ndarray, 
               y: np.ndarray,
               features: pd.Index,
               l1_ratio: list = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
               n_alphas: int = 100, 
               cvfold: int = 5,
               test_size: float = 0.2) -> tuple:

    split = int(len(X)*(1-test_size))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    tscv = TimeSeriesSplit(n_splits=cvfold)
 
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNetCV(
            l1_ratio=l1_ratio,
            n_alphas=n_alphas,
            cv=tscv,
        ))
    ])
    pipeline.fit(X_train, y_train)
 
    model = pipeline.named_steps['model']
    metrics = {
        'train_R2':        pipeline.score(X_train, y_train),
        'test_R2':         pipeline.score(X_test,  y_test),
        'optimal_alpha':   model.alpha_,
        'optimal_l1_ratio':model.l1_ratio_,
        'n_features_total':X.shape[1],
        'n_train_obs':     len(X_train),
        'n_test_obs':      len(X_test),
    }

    coefs = model.coef_
    mask = np.abs(coefs) > zero_threshold
    selected_features = features[mask]
    selected_coefs    = coefs[mask]
    sorted_idx        = np.argsort(np.abs(selected_coefs))[::-1]
 
    results = {
        'selected_features': selected_features[sorted_idx],
        'coefficients':      selected_coefs[sorted_idx],
        'n_selected':        mask.sum(),
    }

    return pipeline, metrics, results

def nowcast(pipeline,
            X_now: pd.Series,
            features_names: pd.Index) -> float:
    
    X_now_aligned = X_now.reindex(features_names).fillna(0.0).values.reshape(1,-1)
    prediction = pipeline.predict(X_now_aligned)[0]
    return prediction




if __name__ == '__main__':

    print("Fetching FRED data...")
    raw = fetch_series(series_ids, start='1990-01-01')

    # Separate GDP (kept in levels) from monthly indicators
    gdp_quarterly      = raw['gdp'].resample('QE').last().dropna()
    indicators_monthly = raw.drop(columns=['gdp']).resample('ME').last()

    # Stationize monthly indicators
    print("\nStationarity transforms:")
    stationary_monthly = stationizer(indicators_monthly, DIFF_COLS, LOG_DIFF_COLS)

    # Build lag-adjusted training matrix
    print("\nBuilding lag-adjusted feature matrix...")
    X_df, y = lag_adj_features(stationary_monthly, gdp_quarterly, PUBLICATION_LAGS)
    print(f"  {X_df.shape[0]} quarters × {X_df.shape[1]} features")
    print(f"  {X_df.index[0].date()} → {X_df.index[-1].date()}")

    # Fit
    print("\nFitting ElasticNetCV...")
    pipeline, metrics, results = elasticnet(X_df.values, y.values, X_df.columns)

    print(f"\n  Train R²        : {metrics['train_R2']:.3f}")
    print(f"  Test R²         : {metrics['test_R2']:.3f}")
    print(f"  Optimal alpha   : {metrics['optimal_alpha']:.4f}")
    print(f"  Optimal l1      : {metrics['optimal_l1_ratio']:.2f}")
    print(f"  Features used   : {results['n_selected']} / {metrics['n_features_total']}")

    print("\n  Selected features (by |coef|):")
    for feat, coef in zip(results['selected_features'], results['coefficients']):
        print(f"    {feat:<20} {coef:+.4f}")

    # Nowcast
    X_now = nowcast_vector(stationary_monthly, PUBLICATION_LAGS)

    gdp_pred   = nowcast(pipeline, X_now, X_df.columns)
    annualized = (np.exp(gdp_pred * 4) - 1) * 100

    print(f"\n{'='*50}")
    print(f"GDP nowcast  ({X_now.name.to_period('Q')})")
    print(f"  Quarterly log-growth : {gdp_pred:.4f}")
    print(f"  Annualised rate      : {annualized:.2f}%")
    print(f"{'='*50}")