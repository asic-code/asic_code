import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ks_statistic(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0.0
    a = np.sort(a)
    b = np.sort(b)
    all_vals = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, all_vals, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_vals, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _mutual_info_from_crosstab(ct):
    total = ct.values.sum()
    if total == 0:
        return 0.0
    pxy = ct / total
    px = pxy.sum(axis=1).values.reshape(-1, 1)
    py = pxy.sum(axis=0).values.reshape(1, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pxy.values / (px * py)
        log_term = np.log(ratio, where=ratio > 0)
        mi = (pxy.values * log_term).sum()
    return float(mi)


def compute_feature_separability(
    df,
    label_col="cluster",
    exclude_cols=None,
    sample_size=None,
    top_n=20,
    max_unique=200,
):
    """
    Compute per-feature separability using:
      - KS statistic for numeric features
      - Mutual information for categorical-like features
    """
    if df is None or df.empty:
        return []

    if label_col not in df.columns:
        logger.warning(f"[Separability] Label column '{label_col}' missing; skip.")
        return []

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    y = df[label_col]
    if y.nunique(dropna=True) < 2:
        logger.info(f"[Separability] Not enough label variation for '{label_col}'.")
        return []

    exclude = set(exclude_cols or [])
    exclude.add(label_col)

    results = []
    for col in df.columns:
        if col in exclude:
            continue

        series = df[col]
        non_na = series.dropna()
        if non_na.empty:
            continue

        is_numeric = pd.api.types.is_numeric_dtype(non_na.dtype)
        if is_numeric:
            group0 = non_na[y.loc[non_na.index] == 0].astype(float).values
            group1 = non_na[y.loc[non_na.index] == 1].astype(float).values
            ks = _ks_statistic(group0, group1)
            results.append(
                {
                    "feature": col,
                    "metric": "ks",
                    "score": ks,
                    "n0": int(len(group0)),
                    "n1": int(len(group1)),
                    "nunique": int(non_na.nunique(dropna=True)),
                }
            )
        else:
            nunique = int(non_na.nunique(dropna=True))
            if nunique > max_unique:
                continue
            ct = pd.crosstab(non_na, y.loc[non_na.index])
            mi = _mutual_info_from_crosstab(ct)
            results.append(
                {
                    "feature": col,
                    "metric": "mi",
                    "score": mi,
                    "n0": int((y.loc[non_na.index] == 0).sum()),
                    "n1": int((y.loc[non_na.index] == 1).sum()),
                    "nunique": nunique,
                }
            )

    if not results:
        return []

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


def log_separability_summary(results, turn_counter=None):
    if not results:
        logger.info(f"[Separability] Turn {turn_counter}: no results.")
        return
    logger.info(f"[Separability] Turn {turn_counter}: top features by separability:")
    for rec in results:
        logger.info(
            f"  - {rec['feature']} ({rec['metric']}={rec['score']:.4f}, "
            f"n0={rec['n0']}, n1={rec['n1']}, nunique={rec['nunique']})"
        )
