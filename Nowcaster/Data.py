import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from fredapi import Fred
from dotenv import load_dotenv
from statsmodels.tsa.stattools import adfuller
import numpy as np
from .config import max_diffs, adf_pval_threshold

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