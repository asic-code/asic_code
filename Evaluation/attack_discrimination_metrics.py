"""
Attack discrimination metrics for association-rule-based signatures.

Given alerts (rows matched by each signature) with attack-type labels (e.g. CICIDS2017 Label),
computes how well each signature distinguishes attack types (purity, entropy)
and summary statistics across signatures.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

ATTACK_LABEL_COL = "Label"  # CICIDS2017; override via argument where needed


def _entropy(probs: np.ndarray) -> float:
    """Entropy of a probability distribution (0 for empty or single class)."""
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def per_signature_attack_distribution(
    alerts_df: pd.DataFrame,
    attack_col: str = ATTACK_LABEL_COL,
    signature_id_col: str = "signature_id",
) -> pd.DataFrame:
    """
    For each signature_id, compute the count and proportion of each attack type in matched rows.

    alerts_df must have at least: signature_id_col, attack_col.
    Returns a DataFrame with columns: signature_id, attack_type, count, total, proportion.
    """
    if attack_col not in alerts_df.columns or signature_id_col not in alerts_df.columns:
        return pd.DataFrame()
    g = alerts_df.groupby(signature_id_col)[attack_col]
    counts = g.value_counts().unstack(fill_value=0)
    totals = counts.sum(axis=1)
    # long form: signature_id, attack_type, count, total, proportion
    rows = []
    for sig_id in counts.index:
        total = totals.loc[sig_id]
        for attack_type in counts.columns:
            cnt = counts.loc[sig_id, attack_type]
            rows.append({
                "signature_id": sig_id,
                "attack_type": attack_type,
                "count": int(cnt),
                "total": int(total),
                "proportion": cnt / total if total > 0 else 0.0,
            })
    return pd.DataFrame(rows)


def per_signature_purity_and_entropy(
    alerts_df: pd.DataFrame,
    attack_col: str = ATTACK_LABEL_COL,
    signature_id_col: str = "signature_id",
    binary_attack_col: str | None = None,
) -> pd.DataFrame:
    """
    For each signature, compute:
    - purity: proportion of the dominant attack type (max proportion)
    - entropy: entropy of the attack-type distribution (higher = less discriminative)
    - dominant_attack: the attack type with highest count
    - n_attack_types: number of distinct attack types matched
    - precision: if binary_attack_col is set, TP/(TP+FP) where positive = (binary_attack_col == 1)
                 (cluster: 0=benign, !=0=attack; or label: 0=normal, 1=anomaly)

    Returns DataFrame with columns: signature_id, purity, entropy, dominant_attack, n_attack_types, total_matched[, precision].
    """
    if attack_col not in alerts_df.columns or signature_id_col not in alerts_df.columns:
        return pd.DataFrame()
    rows = []
    for sig_id, grp in alerts_df.groupby(signature_id_col):
        vc = grp[attack_col].value_counts()
        total = len(grp)
        precision_val = None
        if total > 0 and binary_attack_col and binary_attack_col in grp.columns:
            tp = (grp[binary_attack_col] == 1).sum()
            precision_val = float(tp / total) if total > 0 else 0.0
        if total == 0:
            row = {
                "signature_id": sig_id,
                "purity": 0.0,
                "entropy": 0.0,
                "dominant_attack": None,
                "n_attack_types": 0,
                "total_matched": 0,
                "attack_type_counts": "",
            }
            if binary_attack_col:
                row["precision"] = 0.0
            rows.append(row)
            continue
        probs = vc.values.astype(float) / total
        purity = float(np.max(probs))
        dominant = vc.index[0]
        attack_type_counts_str = ", ".join(f"{k}:{int(v)}" for k, v in vc.items())
        row = {
            "signature_id": sig_id,
            "purity": purity,
            "entropy": _entropy(probs),
            "dominant_attack": dominant,
            "n_attack_types": len(vc),
            "total_matched": total,
            "attack_type_counts": attack_type_counts_str,
        }
        if binary_attack_col:
            row["precision"] = precision_val if precision_val is not None else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def per_attack_specificity(
    alerts_df: pd.DataFrame,
    attack_col: str = ATTACK_LABEL_COL,
    signature_id_col: str = "signature_id",
) -> pd.DataFrame:
    """
    For each attack type: how many signatures fire on it, and what share of their matches
    are this attack type (avg purity for that attack).

    Returns DataFrame: attack_type, n_signatures_firing, total_alerts, avg_purity_for_this_attack.
    """
    if attack_col not in alerts_df.columns or signature_id_col not in alerts_df.columns:
        return pd.DataFrame()
    dist_df = per_signature_attack_distribution(alerts_df, attack_col=attack_col, signature_id_col=signature_id_col)
    if dist_df.empty:
        return pd.DataFrame()
    rows = []
    for attack_type in dist_df["attack_type"].unique():
        sub = dist_df[(dist_df["attack_type"] == attack_type) & (dist_df["count"] > 0)]
        n_sigs = sub["signature_id"].nunique()
        total_alerts = sub["count"].sum()
        # For signatures that fire on this attack, average proportion of their matches that are this attack
        avg_proportion_this = float(sub["proportion"].mean()) if len(sub) else 0.0
        rows.append({
            "attack_type": attack_type,
            "n_signatures_firing": int(n_sigs),
            "total_alerts": int(total_alerts),
            "avg_proportion_this_attack": avg_proportion_this,
        })
    return pd.DataFrame(rows)


def discrimination_summary(
    alerts_df: pd.DataFrame,
    attack_col: str = ATTACK_LABEL_COL,
    signature_id_col: str = "signature_id",
    binary_attack_col: str | None = None,
) -> dict:
    """
    Aggregate discrimination metrics across all signatures and attack types.

    Returns dict with:
    - mean_purity, median_purity (per-signature purity)
    - mean_entropy, median_entropy
    - n_signatures, n_alerts, n_attack_types
    - per_attack_specificity_df (summary table)
    - per_signature_purity_df (per-signature table)
    """
    purity_df = per_signature_purity_and_entropy(
        alerts_df, attack_col=attack_col, signature_id_col=signature_id_col, binary_attack_col=binary_attack_col
    )
    specificity_df = per_attack_specificity(alerts_df, attack_col=attack_col, signature_id_col=signature_id_col)
    n_sigs = purity_df["signature_id"].nunique() if not purity_df.empty else 0
    n_alerts = len(alerts_df)
    n_attack_types = alerts_df[attack_col].nunique() if attack_col in alerts_df.columns else 0
    summary = {
        "n_signatures": n_sigs,
        "n_alerts": n_alerts,
        "n_attack_types": n_attack_types,
        "mean_purity": float(purity_df["purity"].mean()) if not purity_df.empty else 0.0,
        "median_purity": float(purity_df["purity"].median()) if not purity_df.empty else 0.0,
        "mean_entropy": float(purity_df["entropy"].mean()) if not purity_df.empty else 0.0,
        "median_entropy": float(purity_df["entropy"].median()) if not purity_df.empty else 0.0,
        "per_signature_purity_df": purity_df,
        "per_attack_specificity_df": specificity_df,
    }
    return summary


def print_discrimination_report(
    summary: dict,
    attack_col: str = ATTACK_LABEL_COL,
    top_n_signatures: int = 20,
    top_n_attacks: int = 20,
) -> None:
    """Print a human-readable report to logger (INFO)."""
    logger.info("=== Attack discrimination report ===")
    logger.info(f"  Signatures: {summary['n_signatures']}, Alerts: {summary['n_alerts']}, Attack types: {summary['n_attack_types']}")
    logger.info(f"  Mean/median purity (per signature): {summary['mean_purity']:.4f} / {summary['median_purity']:.4f}")
    logger.info(f"  Mean/median entropy (per signature): {summary['mean_entropy']:.4f} / {summary['median_entropy']:.4f}")
    purity_df = summary["per_signature_purity_df"]
    if not purity_df.empty:
        # Worst discriminators (low purity, high entropy)
        worst = purity_df.nlargest(top_n_signatures, "entropy").head(top_n_signatures)
        logger.info(f"  Top-{top_n_signatures} worst discriminators (by entropy):")
        for _, row in worst.iterrows():
            logger.info(f"    sig_id={row['signature_id']} purity={row['purity']:.3f} entropy={row['entropy']:.3f} dominant={row['dominant_attack']} n_types={row['n_attack_types']} matched={row['total_matched']}")
    spec_df = summary["per_attack_specificity_df"]
    if not spec_df.empty:
        logger.info("  Per-attack specificity (n_signatures firing, total_alerts, avg_proportion_this_attack):")
        for _, row in spec_df.head(top_n_attacks).iterrows():
            logger.info(f"    {row['attack_type']}: n_sigs={row['n_signatures_firing']} alerts={row['total_alerts']} avg_prop={row['avg_proportion_this_attack']:.4f}")
    logger.info("=== End report ===")
