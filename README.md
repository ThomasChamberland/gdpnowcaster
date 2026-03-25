# gdpnowcaster
# GDP Nowcasting with ElasticNet

Predicting quarterly GDP growth using monthly economic indicators.

## What This Does

Builds a nowcasting model that predicts current-quarter GDP using:
- 14 monthly indicators (employment, inflation, sentiment, etc.)
- ElasticNet for automatic feature selection
- Publication lag modeling (v2) for realistic backtesting

## Two Versions

### v1: Basic Model
- Simple ElasticNetCV on quarterly-aggregated data
- **Problem:** Uses future data (not realistic), does not implement nowcasting logic in the process
- Test R²: ~0.XX

### v2: Publication-Lag Aware
- Better structure than v1
- Models when data is actually released
- Only uses data available at prediction time

### V3, V4, V5
- Add random forest (V3), XGboost (V4), compare to Lasso
- Scale up, add PCA factors and have 50+ indicators (V5)
