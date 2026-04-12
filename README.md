# GDP Nowcasting
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
- **Problem:** Uses future data (not realistic), does not implement nowcasting logic in the process, 0 annotations and lacks clarity
- Test R²: ~0.XX

### v2: Publication-Lag Aware
- Better structure than v1
- Models when data is actually released
- Only uses data available at prediction time

### V3: Implementation of DGP program prior to implementing nonlinear ML methods
- AR process used to generate synthetic data to simulate how different models would fare on low observation, high dimensional samples
- Results show that Linear processes (Elasticnet and Lasso) are by far the best
- Will still implement nonlinear processes, might interact differently on empirical data set
#### - Based on IMF Working Paper WP/25/252

### V4
- Add random forest, GBM, SVR, ensemble methods
- Scale up number of features

### V5
- PCA factors
