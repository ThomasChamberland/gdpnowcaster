import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from fredapi import Fred
from concurrent.futures import ThreadPoolExecutor, as_completed


# Data collection
load_dotenv()
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

series_id = {
    'gdp': 'GDP',
    'inflation': 'CPIAUCSL',
    'indus_prod': 'INDPRO',
    'nonf_proll': 'PAYEMS',
    'retail_trade': 'RSAFS',
    'unrate': 'UNRATE',
    'InitClaims': 'ICSA',
    'PPI_AC': 'PPIACO',
    'FedFundRate': 'FEDFUNDS',
    '10YYield': 'GS10',
    '10Y2Y_spread': 'T10Y2Y',
    'Sentiment': 'UMCSENT',
    'Business Confidence': 'BSCICP03USM665S',
    'Trade_Balance': 'BOPGSTB',
    'Lumber PPI': 'WPU081'
}

def fetch_series(name, series_id):
    return name, fred.get_series(series_id, observation_start='1990-01-01')

with ThreadPoolExecutor(max_workers=len(series_id)) as executor:
    futures = {
        executor.submit(fetch_series, name, sid): name
        for name, sid in series_id.items()
    }
    results = {}
    for future in as_completed(futures):
        try:
            name, series = future.result()
            results[name] = series
        except Exception as e:
            print(f"Failed to fetch {futures[future]}: {e}")
# %%
data = pd.DataFrame(results)
data_me = data.resample('ME').last()
data_q = data.resample('QE').last()
data_q
# Data Visualization Function
# %%
def plot_data(series_df, ncols):
    fig, axes = plt.subplots(nrows=len(series_df.columns)//ncols, ncols=ncols, figsize=(15,12))
    axes = axes.flatten()

    for ax, col in zip(axes, series_df.columns):
        series = series_df[col].dropna()
        ax.plot(series.index, series)
        ax.set_title(col)
        ax.set_xlim(series_df.index[0], series_df.index[-1])
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Processing

used_data_q = data_q.copy() # Copy of Quarterly adj. data

used_data_q = used_data_q.drop(columns=['Business Confidence']).dropna(how='any').iloc[:-1]


# Stationarizing data

## First difference and log diffed

fdiff_cols = ['unrate','FedFundRate','10YYield','10Y2Y_spread', 'Sentiment','Trade_Balance']
log_diff_cols = ['gdp', 'inflation', 'indus_prod', 'nonf_proll', 'retail_trade', 'PPI_AC', 'Lumber PPI', 'InitClaims']



# Test of stationarity

from statsmodels.tsa.stattools import adfuller



##########################################
# %% 
def stationizer(used_data, fdiff_cols, log_diff_cols):
    max_diffs = 3
    stationized_data = used_data.copy()
    for col in stationized_data:
        n_diffs = 0
        pval = adfuller(stationized_data[col].dropna())[1]
        while pval > 0.05 and n_diffs < max_diffs:
            if col in log_diff_cols:
                stationized_data[col] = np.log(stationized_data[col]).diff()
                pval = adfuller(stationized_data[col].dropna())[1]
                break
            elif col in fdiff_cols:
                stationized_data[col] = stationized_data[col].diff()
                n_diffs += 1
                pval = adfuller(stationized_data[col].dropna())[1]
            else:
                print(f"{col} not in either list, skipping")
                break

        print(f"{col}: p={pval:.3f} {'Stationary' if pval < 0.05 else 'NOT stationary'} {n_diffs}")
    return stationized_data.dropna()


# Building pipeline for elasticnet and fitting

def splits(stationized_data, target_var):
    features = stationized_data.columns.drop(target_var)

    X = stationized_data[features].values
    y = stationized_data[target_var].values

    return X, y, features

def elasticnet(X, y, features, l1_ratio = 0.5, n_alphas = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0], 
               cvfold = 5, test_size = 0.2, random_state=42):

    split = int(len(X)*(1-test_size))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    tscv = TimeSeriesSplit(n_splits=cvfold)

    pipeline = Pipeline([('scaler', StandardScaler()),
                         ('ElasticNet', ElasticNetCV(
                             l1_ratio=l1_ratio,
                             n_alphas=n_alphas,
                             cv = tscv,
                             random_state=random_state
                         )
                         )])
    pipeline.fit(X_train, y_train)
    metrics = {
        'train_R2': pipeline.score(X_train, y_train),
        'test_R2': pipeline.score(X_test, y_test),
        'optimal_alpha': pipeline.named_steps['ElasticNet'].alpha_,
        'n_features_total': X.shape[1]
    }
    model = pipeline.named_steps['ElasticNet']
    coefs = model.coef_

    mask = np.abs(coefs) > 1e-5
    selected_features = features[mask]
    selected_coefs = coefs[mask]

    sorted_idx = np.argsort(np.abs(selected_coefs))[::-1]

    results = {
        'selected_features': selected_features[sorted_idx],
        'coefficients': selected_coefs[sorted_idx],
        'n_selected': mask.sum()
    }
    return pipeline, metrics, results


