import pandas as pd
import numpy as np
from .config import gdp_lag_days

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
    feat_df = feat_df.dropna(axis=1, how='all')  # drop features with no data at all
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
