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