import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

def splits(data: pd.DataFrame, target_col: str) -> tuple:
    features = data.columns.drop(target_col)

    X = data[features].values
    y = data[target_col].values

    return X, y, features

def nowcast(pipeline,
            X_now: pd.Series,
            features_names: pd.Index) -> float:
    
    X_now_aligned = X_now.reindex(features_names).fillna(0.0).values.reshape(1,-1)
    prediction = pipeline.predict(X_now_aligned)[0]
    return prediction

def extract_importances(pipeline: Pipeline, feature_names: pd.Index) -> pd.Series:
    model = pipeline.named_steps["model"]

    if hasattr(model, "coef_"):                  # ElasticNet, SVR with linear kernel
        values = np.abs(model.coef_)
    elif hasattr(model, "feature_importances_"):  # RF, GBM
        values = model.feature_importances_
    else:                                         # SVR with RBF, stacking
        return pd.Series(dtype=float)             # empty — use permutation importance instead

    return pd.Series(values, index=feature_names).sort_values(ascending=False)