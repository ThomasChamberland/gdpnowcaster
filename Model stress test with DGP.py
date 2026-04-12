"""
DGP Simulation Module for GDP Nowcasting
Implements the Monte Carlo simulation from the IMF Working Paper WP/25/252.

Factor structure:
    F factors (n_F) -> drive both GDP-relevant indicators (X_u) and observed indicators (X_o)
    G factors (n_G) -> drive only observed indicators (X_o), irrelevant to nowcast target (GDP)

Indicator structure:
    X_u (K indicators)   = beta * F + noise          (enter GDP equation)
    X_o (N-K indicators) = beta_u * F + beta_o * G + noise  (correlated but noisy)

GDP is generated from X_u via one of three DGPs:
    DGP 1: Linear
    DGP 2: Quadratic (adds squared terms)
    DGP 3: Quadratic + Interaction terms

The models never see which indicators are X_u vs X_o.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DGP CONFIGURATION
# ============================================================================

# Default simulation parameters (matching the paper)
DEFAULT_SIM_CONFIG = {
    # Factor structure
    'n_F': 5,              # Number of GDP-relevant factors
    'n_G': 3,              # Number of irrelevant factors
    'lambda_lo': 0.4,      # AR persistence lower bound
    'lambda_hi': 0.6,      # AR persistence upper bound

    # Indicator structure
    'K': 5,                # Number of unobservable (GDP-generating) indicators
    'N': 105,              # Total number of monthly indicators (K + observed)
    'beta_lo': 0.0,        # Loading lower bound
    'beta_hi': 0.9,        # Loading upper bound

    # Simulation settings
    'n_monte_carlo': 500,  # Number of Monte Carlo iterations
    'train_ratio': 0.7,    # Train/test split ratio
    'T_values': [40, 50, 60, 70, 80, 100, 120, 150, 200],  # Quarterly sample sizes
}

# Small-scale config for quick testing / debugging
SMALL_SIM_CONFIG = {
    'n_F': 3,
    'n_G': 2,
    'lambda_lo': 0.4,
    'lambda_hi': 0.6,
    'K': 5,
    'N': 20,
    'beta_lo': 0.0,
    'beta_hi': 0.9,
    'n_monte_carlo': 50,
    'train_ratio': 0.7,
    'T_values': [40, 60, 80, 100],
}

# GDP equation coefficients (fixed across all iterations, as in paper)
# DGP 1 uses only the linear terms
# DGP 2 adds quadratic terms
# DGP 3 adds quadratic + interaction terms
GDP_COEFFICIENTS = {
    'intercept': 1.2,
    # X1 loadings: homogeneous across months in quarter
    'x1': [0.5, 0.5, 0.5],        # [month_0, month_-1, month_-2]
    # X2 loadings: heterogeneous
    'x2': [0.3, 0.6, 0.4],
    # X3 loadings: heterogeneous, negative
    'x3': [-0.25, -0.2, -0.3],
    # Quadratic terms on X2 (DGP 2 and 3)
    'x2_sq': [0.05, 0.02, 0.03],
    # Interaction X3*X4 (DGP 3 only)
    'x3_x4': [0.15, 0.15, 0.15],
    # Interaction X4*X5 (DGP 3 only)
    'x4_x5': [-0.2, -0.3, -0.1],
}


# ============================================================================
# FACTOR AND DATA GENERATION
# ============================================================================

def generate_factors(n_factors: int, T_monthly: int,
                     lambda_lo: float, lambda_hi: float,
                     rng: np.random.Generator) -> np.ndarray:
    """
    Generate latent AR(1) factors.

    Each factor: F_t = lambda_i * F_{t-1} + eta_t,  eta ~ N(0,1)

    Returns:
        factors: (n_factors, T_monthly) array
    """
    lambdas = rng.uniform(lambda_lo, lambda_hi, size=n_factors)
    factors = np.zeros((n_factors, T_monthly))
    shocks = rng.standard_normal((n_factors, T_monthly))

    for t in range(1, T_monthly):
        factors[:, t] = lambdas * factors[:, t - 1] + shocks[:, t]

    return factors


def generate_indicators(F: np.ndarray, G: np.ndarray,
                        K: int, N: int,
                        beta_lo: float, beta_hi: float,
                        rng: np.random.Generator) -> np.ndarray:
    """
    Generate monthly indicators from factors.

    X_u (first K):  X_u[i,t] = beta[i] @ F[:,t] + noise
    X_o (rest):     X_o[i,t] = beta_u[i] @ F[:,t] + beta_o[i] @ G[:,t] + noise

    Returns:
        X: (N, T_monthly) array — all indicators combined, X_u first then X_o
    """
    n_F, T_m = F.shape
    n_G = G.shape[0]
    N_obs = N - K  # number of observed (noisy) indicators

    X = np.zeros((N, T_m))

    # X_u: unobservable indicators (GDP-generating), load only on F
    beta_u_on_F = rng.uniform(beta_lo, beta_hi, size=(K, n_F))
    noise_u = rng.standard_normal((K, T_m))
    X[:K, :] = beta_u_on_F @ F + noise_u

    # X_o: observed indicators, load on both F and G
    beta_o_on_F = rng.uniform(beta_lo, beta_hi, size=(N_obs, n_F))
    beta_o_on_G = rng.uniform(beta_lo, beta_hi, size=(N_obs, n_G))
    noise_o = rng.standard_normal((N_obs, T_m))
    X[K:, :] = beta_o_on_F @ F + beta_o_on_G @ G + noise_o

    return X


def monthly_to_quarterly_blocks(X: np.ndarray, K: int) -> np.ndarray:
    """
    Convert monthly indicators to quarterly by splitting each month in a quarter
    into a separate variable (blocking/split-sampling).

    For each indicator i and quarter q:
        x_i_0  = value in last month of quarter    (month 0)
        x_i_-1 = value in second month of quarter  (month -1)
        x_i_-2 = value in first month of quarter   (month -2)

    Only uses complete quarters (trims leading months if T_m not divisible by 3).

    Returns:
        X_q: (T_quarterly, N * 3) array — blocked quarterly features
        X_u_quarterly: dict with keys 0..K-1, each containing (T_q, 3) array
                       for the three monthly values of each unobservable indicator
    """
    N, T_m = X.shape
    T_q = T_m // 3
    trim = T_m - T_q * 3  # months to drop from start

    X_blocked = np.zeros((T_q, N * 3))
    X_u_quarterly = {}

    for q in range(T_q):
        m_start = trim + q * 3  # first month of this quarter
        # month_-2 (first), month_-1 (second), month_0 (last)
        for lag, offset in enumerate([2, 1, 0]):
            col_start = lag * N
            X_blocked[q, col_start:col_start + N] = X[:, m_start + (2 - lag)]

    # Also extract the unobservable indicators separately for GDP generation
    for i in range(K):
        X_u_quarterly[i] = np.zeros((T_q, 3))
        for q in range(T_q):
            m_start = trim + q * 3
            X_u_quarterly[i][q, 0] = X[i, m_start + 2]  # month 0 (last)
            X_u_quarterly[i][q, 1] = X[i, m_start + 1]  # month -1
            X_u_quarterly[i][q, 2] = X[i, m_start]       # month -2

    return X_blocked, X_u_quarterly


def generate_gdp(X_u_q: dict, dgp: int, K: int) -> np.ndarray:
    """
    Generate quarterly GDP from unobservable indicators using specified DGP.

    Args:
        X_u_q: dict of (T_q, 3) arrays for each unobservable indicator
        dgp: 1 (linear), 2 (quadratic), or 3 (quadratic + interactions)
        K: number of unobservable indicators

    Returns:
        y: (T_q,) array of quarterly GDP
    """
    c = GDP_COEFFICIENTS
    T_q = X_u_q[0].shape[0]
    y = np.full(T_q, c['intercept'])

    # Linear terms — X1, X2, X3 (indices 0, 1, 2)
    # Only uses first 3 of K unobservable indicators for linear terms
    for month in range(3):  # 0, -1, -2
        if K > 0:
            y += c['x1'][month] * X_u_q[0][:, month]
        if K > 1:
            y += c['x2'][month] * X_u_q[1][:, month]
        if K > 2:
            y += c['x3'][month] * X_u_q[2][:, month]

    # DGP 2 and 3: add quadratic terms on X2
    if dgp >= 2 and K > 1:
        for month in range(3):
            y += c['x2_sq'][month] * X_u_q[1][:, month] ** 2

    # DGP 3: add interaction terms X3*X4 and X4*X5
    if dgp >= 3:
        if K > 3:
            for month in range(3):
                y += c['x3_x4'][month] * X_u_q[2][:, month] * X_u_q[3][:, month]
        if K > 4:
            for month in range(3):
                y += c['x4_x5'][month] * X_u_q[3][:, month] * X_u_q[4][:, month]

    return y


# ============================================================================
# FULL DATA GENERATION PIPELINE
# ============================================================================

def generate_dataset(T_quarterly: int, dgp: int, config: dict,
                     rng: np.random.Generator) -> tuple:
    """
    Generate one complete synthetic dataset.

    Returns:
        X_blocked: (T_q, N*3) quarterly blocked features (what models see)
        y: (T_q,) quarterly GDP target
        X_averaged: (T_q, N) quarterly averaged features (alternative aggregation)
    """
    T_monthly = T_quarterly * 3

    # Generate factors
    F = generate_factors(config['n_F'], T_monthly,
                         config['lambda_lo'], config['lambda_hi'], rng)
    G = generate_factors(config['n_G'], T_monthly,
                         config['lambda_lo'], config['lambda_hi'], rng)

    # Generate monthly indicators
    X = generate_indicators(F, G, config['K'], config['N'],
                            config['beta_lo'], config['beta_hi'], rng)

    # Convert to quarterly
    X_blocked, X_u_q = monthly_to_quarterly_blocks(X, config['K'])

    # Also create quarterly averages (alternative to blocking)
    N, T_m = X.shape
    T_q = T_m // 3
    trim = T_m - T_q * 3
    X_averaged = np.zeros((T_q, N))
    for q in range(T_q):
        m_start = trim + q * 3
        X_averaged[q, :] = X[:, m_start:m_start + 3].mean(axis=1)

    # Generate GDP
    y = generate_gdp(X_u_q, dgp, config['K'])

    return X_blocked, y, X_averaged


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def build_models(cv_folds: int = 5) -> dict:
    """
    Build dictionary of models to evaluate.
    Returns dict of {name: sklearn Pipeline}.

    Starts with the models you already have (ElasticNet) plus a few others.
    Add more as needed.
    """
    cv = KFold(n_splits=cv_folds, shuffle=False)

    models = {}

    # --- Linear ML (paper's best ML performers) ---
    models['ElasticNet'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
            n_alphas=100,
            cv=cv,
            max_iter=10000,
        ))
    ])

    models['Lasso'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LassoCV(
            n_alphas=100,
            cv=cv,
            max_iter=10000,
        ))
    ])

    # --- Non-linear ML ---
    models['RandomForest'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(
            n_estimators=100,
            max_depth=4,
            random_state=42,
        ))
    ])

    models['SVR_Linear'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVR(kernel='linear', C=1.0))
    ])

    models['SVR_Radial'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVR(kernel='rbf', C=1.0))
    ])

    return models


def ar1_benchmark(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """
    AR(1) benchmark: y_t = phi * y_{t-1} + c
    Estimated on training set, applied to test set.
    """
    y_lag = y_train[:-1]
    y_curr = y_train[1:]
    # OLS for AR(1)
    X_ar = np.column_stack([np.ones_like(y_lag), y_lag])
    beta = np.linalg.lstsq(X_ar, y_curr, rcond=None)[0]

    # For test set, predict using lagged values
    # First test prediction uses last training observation
    preds = np.zeros(len(y_test))
    prev = y_train[-1]
    for i in range(len(y_test)):
        preds[i] = beta[0] + beta[1] * prev
        prev = y_test[i]  # use actual for next step (recursive)
    return preds


# ============================================================================
# MONTE CARLO EVALUATION
# ============================================================================

def run_single_iteration(T_quarterly: int, dgp: int, config: dict,
                         models: dict, rng: np.random.Generator) -> dict:
    """
    Run one Monte Carlo iteration:
    1. Generate data
    2. Split train/test
    3. Fit all models + AR(1) benchmark
    4. Return relative RMSEs
    """
    # Generate data — use blocked features (paper reports lower of blocked vs averaged)
    X_blocked, y, X_averaged = generate_dataset(T_quarterly, dgp, config, rng)

    # Train/test split
    n_train = int(len(y) * config['train_ratio'])
    if n_train < 10 or len(y) - n_train < 5:
        return None  # skip if too few observations

    X_train_b, X_test_b = X_blocked[:n_train], X_blocked[n_train:]
    X_train_a, X_test_a = X_averaged[:n_train], X_averaged[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # AR(1) benchmark
    ar_preds = ar1_benchmark(y_train, y_test)
    rmse_ar = np.sqrt(mean_squared_error(y_test, ar_preds))

    if rmse_ar < 1e-10:
        return None  # degenerate case

    # Evaluate each model on both blocked and averaged, take lower RMSE
    results = {}
    for name, model_template in models.items():
        try:
            best_rmse = np.inf

            for X_tr, X_te, label in [(X_train_b, X_test_b, 'blocked'),
                                       (X_train_a, X_test_a, 'averaged')]:
                from sklearn.base import clone
                model = clone(model_template)
                model.fit(X_tr, y_train)
                preds = model.predict(X_te)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                if rmse < best_rmse:
                    best_rmse = rmse

            results[name] = best_rmse / rmse_ar  # relative RMSE

        except Exception as e:
            results[name] = np.nan

    return results


def run_monte_carlo(T_quarterly: int, dgp: int, config: dict,
                    models: dict, seed: int = 42) -> pd.DataFrame:
    """
    Run full Monte Carlo simulation for a given T and DGP.

    Returns:
        DataFrame with columns = model names, rows = iterations,
        values = relative RMSE vs AR(1)
    """
    rng = np.random.default_rng(seed)
    n_mc = config['n_monte_carlo']
    all_results = []

    print(f"  Running {n_mc} iterations for T={T_quarterly}, DGP={dgp}...")

    for i in range(n_mc):
        result = run_single_iteration(T_quarterly, dgp, config, models, rng)
        if result is not None:
            all_results.append(result)

        if (i + 1) % max(1, n_mc // 5) == 0:
            print(f"    iteration {i + 1}/{n_mc}")

    df = pd.DataFrame(all_results)
    return df


def run_full_simulation(config: dict = None, dgps: list = None,
                        seed: int = 42) -> dict:
    """
    Run the complete simulation across all T values and DGPs.

    Returns:
        nested dict: results[dgp][T] = DataFrame of relative RMSEs
    """
    if config is None:
        config = SMALL_SIM_CONFIG  # default to small for safety
    if dgps is None:
        dgps = [1]  # start with linear DGP

    models = build_models()
    results = {}

    for dgp in dgps:
        print(f"\n{'='*60}")
        print(f"DGP {dgp}")
        print(f"{'='*60}")
        results[dgp] = {}

        for T_q in config['T_values']:
            df = run_monte_carlo(T_q, dgp, config, models, seed)
            results[dgp][T_q] = df

            # Print summary
            print(f"\n  T={T_q}: Mean relative RMSE (vs AR(1)):")
            means = df.mean()
            stds = df.std()
            for name in means.index:
                print(f"    {name:<20} {means[name]:.3f}  (±{stds[name]:.3f})")
            print(f"    {'Best:':<20} {means.idxmin()} ({means.min():.3f})")

    return results


# ============================================================================
# RESULTS SUMMARY
# ============================================================================

def summarize_results(results: dict) -> pd.DataFrame:
    """
    Create a summary table like Tables 3-5 in the paper.

    Returns:
        DataFrame with MultiIndex (DGP, T) and columns = model names,
        values = mean relative RMSE
    """
    rows = []
    for dgp, t_dict in results.items():
        for T_q, df in t_dict.items():
            row = {'DGP': dgp, 'T': T_q}
            for col in df.columns:
                row[f'{col}_mean'] = df[col].mean()
                row[f'{col}_std'] = df[col].std()
            rows.append(row)

    summary = pd.DataFrame(rows).set_index(['DGP', 'T'])
    return summary


def print_summary_table(results: dict):
    """Print a clean summary table to console."""
    for dgp, t_dict in results.items():
        dgp_names = {1: 'Linear', 2: 'Quadratic', 3: 'Quadratic + Interactions'}
        print(f"\n{'='*80}")
        print(f"DGP {dgp}: {dgp_names.get(dgp, 'Unknown')}")
        print(f"{'='*80}")

        # Collect model names from first result
        first_df = list(t_dict.values())[0]
        model_names = list(first_df.columns)

        # Header
        header = f"{'T':>6}"
        for name in model_names:
            header += f" {name:>14}"
        header += f" {'Best':>14}"
        print(header)
        print("-" * len(header))

        for T_q, df in sorted(t_dict.items()):
            means = df.mean()
            row_str = f"{T_q:>6}"
            best_val = means.min()
            for name in model_names:
                val = means[name]
                marker = " *" if val == best_val else "  "
                row_str += f" {val:>12.3f}{marker}"
            row_str += f" {means.idxmin():>14}"
            print(row_str)


# ============================================================================
# DIAGNOSTIC CHECKS
# ============================================================================

def diagnose_dgp(T_quarterly: int = 80, dgp: int = 1,
                 config: dict = None, seed: int = 42):
    """
    Generate one dataset and print diagnostic statistics.
    Useful for verifying the DGP is working correctly.
    """
    if config is None:
        config = SMALL_SIM_CONFIG

    rng = np.random.default_rng(seed)
    X_blocked, y, X_averaged = generate_dataset(T_quarterly, dgp, config, rng)

    print(f"DGP {dgp} Diagnostics (T_q={T_quarterly})")
    print(f"  GDP (y):      mean={y.mean():.3f}, std={y.std():.3f}")
    print(f"  X_blocked:    shape={X_blocked.shape}")
    print(f"  X_averaged:   shape={X_averaged.shape}")

    # Correlations between averaged indicators and GDP
    corrs = np.array([np.corrcoef(X_averaged[:, i], y)[0, 1]
                      for i in range(X_averaged.shape[1])])

    K = config['K']
    print(f"\n  Correlations with GDP:")
    print(f"    X_u (first {K}):  mean={corrs[:K].mean():.3f}, "
          f"range=[{corrs[:K].min():.3f}, {corrs[:K].max():.3f}]")
    print(f"    X_o (rest):       mean={corrs[K:].mean():.3f}, "
          f"range=[{corrs[K:].min():.3f}, {corrs[K:].max():.3f}]")
    print(f"    All:              mean={corrs.mean():.3f}, "
          f"range=[{corrs.min():.3f}, {corrs.max():.3f}]")

    # Paper says correlations should be 0.2-0.6
    in_range = np.sum((np.abs(corrs) >= 0.1) & (np.abs(corrs) <= 0.7))
    print(f"    In [0.1, 0.7]:    {in_range}/{len(corrs)} indicators")

    return X_blocked, y, X_averaged


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':

    # Step 1: Diagnose — verify DGP produces reasonable data
    print("STEP 1: Diagnosing DGP...")
    print("-" * 40)
    diagnose_dgp(T_quarterly=80, dgp=1, config=SMALL_SIM_CONFIG)
    print()
    diagnose_dgp(T_quarterly=80, dgp=2, config=SMALL_SIM_CONFIG)
    print()
    diagnose_dgp(T_quarterly=80, dgp=3, config=SMALL_SIM_CONFIG)

    # Step 2: Run small simulation
    print("\n\nSTEP 2: Running simulation (small scale)...")
    print("-" * 40)
    results = run_full_simulation(
        config=SMALL_SIM_CONFIG,
        dgps=[1, 2, 3],
        seed=42
    )

    # Step 3: Print results
    print("\n\nSTEP 3: Results Summary")
    print("-" * 40)
    print_summary_table(results)
