"""
Leakage-safe feature pipeline for the IEEE-CIS fraud detection project.

Design mirrors scikit-learn's fit/transform split:
    - `fit_feature_pipeline(train_df)` learns all statistics from TRAIN ONLY
      and returns a `state` dict (per-card amount stats, global fallbacks,
      and the frozen list of feature columns).
    - `transform_features(df, state)` applies every feature step to any split
      using the train-learned `state`, so validation/test can never leak
      information into the fitted statistics.
    - `make_Xy(df, state)` returns the model-ready feature matrix and target.

Typical use:
    state = fit_feature_pipeline(train_df)
    X_train, y_train = make_Xy(transform_features(train_df, state), state)
    X_val,   y_val   = make_Xy(transform_features(val_df,   state), state)
    X_test,  y_test  = make_Xy(transform_features(test_df,  state), state)
"""

import numpy as np
import pandas as pd

# Low-cardinality categoricals to one-hot encode
CAT_COLS = ["ProductCD", "card4", "card6"]

# Missing-indicators kept because they separate fraud in EDA
# (maps the new indicator column name -> the source column it checks)
KEEP_INDICATORS = {
    "addr1_missing": "addr1",
    "D6_missing": "D6",
    "D8_missing": "D8",
    "D12_missing": "D12",
    "dist2_missing": "dist2",
    "id_31_missing": "id_31",
}

# Columns that must never be used as features
EXCLUDE = ["isFraud", "TransactionID", "TransactionDT"]


def fit_feature_pipeline(train_df):
    """Learn all train-only statistics needed for stateful features.

    Returns a `state` dict consumed by `transform_features` and `make_Xy`.
    """
    state = {}

    # Per-card amount statistics — LEARNED ON TRAIN ONLY (leakage-critical)
    stats = (
        train_df.groupby("card1")["TransactionAmt"]
        .agg(["mean", "std"])
        .reset_index()
    )
    stats.columns = ["card1", "card_amt_mean", "card_amt_std"]
    state["card_stats"] = stats
    state["global_mean"] = train_df["TransactionAmt"].mean()
    state["global_std"] = train_df["TransactionAmt"].std()

    # Freeze the feature column schema from a transformed train sample
    transformed = transform_features(train_df, state, _defining_schema=True)
    numeric = transformed.select_dtypes(include=[np.number, "bool"]).columns
    state["feature_cols"] = [c for c in numeric if c not in EXCLUDE]

    return state


def transform_features(df, state, _defining_schema=False):
    """Apply all feature steps to a split, using train-learned `state`.

    `_defining_schema` is used internally by `fit_feature_pipeline` before the
    final feature column list exists; leave it False in normal use.
    """
    df = df.copy()

    # 1. Time feature (hour of the DT-reference day)
    df["hour"] = ((df["TransactionDT"] / 3600) % 24).astype(int)

    # 2. Missing-indicators (stateless — pure null checks)
    for ind_col, src_col in KEEP_INDICATORS.items():
        df[ind_col] = df[src_col].isnull().astype(int)

    # 3. Card-level amount z-score (stateful — uses TRAIN stats from `state`)
    df = df.merge(state["card_stats"], on="card1", how="left")
    df["card_amt_mean"] = df["card_amt_mean"].fillna(state["global_mean"])
    df["card_amt_std"] = (
        df["card_amt_std"].replace(0, state["global_std"]).fillna(state["global_std"])
    )
    df["amt_z_for_card"] = (df["TransactionAmt"] - df["card_amt_mean"]) / df["card_amt_std"]

    # 4. One-hot encode low-cardinality categoricals (NaN as its own category)
    df = pd.get_dummies(df, columns=CAT_COLS, dummy_na=True, dtype=int)

    if _defining_schema:
        return df

    # 5. Align to the frozen train schema (add missing dummy cols as 0)
    for col in state["feature_cols"]:
        if col not in df.columns:
            df[col] = 0

    return df


def make_Xy(df, state):
    """Return model-ready X (frozen feature columns) and y (target, or None)."""
    X = df[state["feature_cols"]]
    y = df["isFraud"] if "isFraud" in df.columns else None
    return X, y
