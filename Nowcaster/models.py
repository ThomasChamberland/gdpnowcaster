from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import (RandomForestRegressor,
                              GradientBoostingRegressor,
                              StackingRegressor)
import numpy as np
import pandas as pd
from .config import zero_threshold


def build_elasticnet(l1_ratio: list = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
               n_alphas: int = 100,
               cvfold: int = 5) -> Pipeline:

    tscv = TimeSeriesSplit(n_splits=cvfold)

    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNetCV(
            l1_ratio=l1_ratio,
            n_alphas=n_alphas,
            cv=tscv,
        ))
    ])
def build_random_forest(n_estimators: int = 500,
                        max_depth: int | None = 6,
                        min_samples_leaf: int = 4,
                        random_state: int = 42) -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            max_features='sqrt'
        ))
    ])
def build_gradient_boosting(n_estimators: int = 300,
                            max_depth: int = 3,
                            learning_rate: float = 0.01,
                            subsample: float = 0.8,
                            random_state: int = 42):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=random_state
        ))
    ])

REGISTRY = {
    "elasticnet": build_elasticnet,
    "random_forest": build_random_forest,
    "gradient_boosting": build_gradient_boosting
}

def build_model(name: str) -> Pipeline:
    return REGISTRY[name]()