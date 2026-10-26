import logging
import os
import re

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def _default_separation_base_dir(file_type: str) -> str:
    ft = str(file_type or "CICIDS2017")
    return os.path.join("..", "Seperate_Attack_Ex", ft, "var_benign_ack")


def _default_separation_paths(file_type: str):
    base = _default_separation_base_dir(file_type)
    return {
        "full": os.path.join(base, "feature_distribution_separation_full.csv"),
        "turn": os.path.join(base, "feature_distribution_separation_turn.csv"),
        "stability": os.path.join(base, "feature_variance_turn_stability.csv"),
    }


def _norm_feature_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def load_separation_feature_pool(args):
    """
    Optional feature pool loader for association mining:
    - Reads full separation metrics CSV (required when enabled)
    - Optionally intersects with turn-stability CSV
    Returns: set(feature_names) or None
    """
    if not getattr(args, "use_separation_feature_filter", False):
        return None
    allow_missing = bool(getattr(args, "separation_allow_missing_files", False))
    dynamic_turn = bool(getattr(args, "use_separation_turn_dynamic", True))
    full_csv = os.path.expanduser(args.separation_full_csv)
    if not os.path.isfile(full_csv):
        msg = f"[FeatFilter] separation_full_csv not found: {full_csv}."
        if dynamic_turn:
            logger.warning(msg + " Global fallback pool disabled; will rely on per-turn pools.")
            return None
        if allow_missing:
            logger.warning(msg + " Feature filter disabled.")
            return None
        raise FileNotFoundError(msg + " Stop because separability filter is enabled.")
    try:
        df = pd.read_csv(full_csv, low_memory=False)
    except Exception as e:
        msg = f"[FeatFilter] Failed reading separation_full_csv: {e}."
        if dynamic_turn:
            logger.warning(msg + " Global fallback pool disabled; will rely on per-turn pools.")
            return None
        if allow_missing:
            logger.warning(msg + " Feature filter disabled.")
            return None
        raise RuntimeError(msg + " Stop because separability filter is enabled.")
    if "feature" not in df.columns:
        msg = "[FeatFilter] separation_full_csv has no 'feature' column."
        if dynamic_turn:
            logger.warning(msg + " Global fallback pool disabled; will rely on per-turn pools.")
            return None
        if allow_missing:
            logger.warning(msg + " Feature filter disabled.")
            return None
        raise ValueError(msg + " Stop because separability filter is enabled.")

    work = df.copy()
    if "ks_statistic" in work.columns:
        work = work[work["ks_statistic"].fillna(-1) >= args.separation_min_ks]
    if "q05_q95_overlap_ratio" in work.columns:
        work = work[work["q05_q95_overlap_ratio"].fillna(1.0) <= args.separation_max_overlap]
    if "cohens_d_attack_vs_benign" in work.columns:
        work = work[work["cohens_d_attack_vs_benign"].abs().fillna(0.0) >= args.separation_min_abs_cohens_d]

    if getattr(args, "use_separation_stability_filter", False):
        stab_csv = os.path.expanduser(args.separation_stability_csv)
        if os.path.isfile(stab_csv):
            try:
                sdf = pd.read_csv(stab_csv, low_memory=False)
                if "feature" in sdf.columns:
                    if "attack_gt_benign_ratio" in sdf.columns:
                        sdf = sdf[sdf["attack_gt_benign_ratio"].fillna(0.0) >= args.separation_min_attack_gt_ratio]
                    if "sign_flip_rate" in sdf.columns:
                        sdf = sdf[sdf["sign_flip_rate"].fillna(1.0) <= args.separation_max_flip_rate]
                    keep_features = set(sdf["feature"].astype(str))
                    work = work[work["feature"].astype(str).isin(keep_features)]
                else:
                    logger.warning("[FeatFilter] stability csv has no 'feature' column; skipping stability filter.")
            except Exception as e:
                logger.warning(f"[FeatFilter] Failed reading stability csv: {e}. skipping stability filter.")
        else:
            msg = f"[FeatFilter] stability csv not found: {stab_csv}."
            if allow_missing:
                logger.warning(msg + " skipping stability filter.")
            else:
                raise FileNotFoundError(msg + " Stop because stability filter is enabled.")

    rank_cols = []
    ascending = []
    if "ks_statistic" in work.columns:
        rank_cols.append("ks_statistic")
        ascending.append(False)
    if "separation_score_1_minus_overlap" in work.columns:
        rank_cols.append("separation_score_1_minus_overlap")
        ascending.append(False)
    if "cohens_d_attack_vs_benign" in work.columns:
        work["abs_cohens_d_tmp"] = work["cohens_d_attack_vs_benign"].abs()
        rank_cols.append("abs_cohens_d_tmp")
        ascending.append(False)
    if rank_cols:
        work = work.sort_values(rank_cols, ascending=ascending)
    top_k = max(0, int(args.separation_top_k_features))
    if top_k > 0:
        work = work.head(top_k)

    feature_pool = set(work["feature"].astype(str))
    if len(feature_pool) < max(1, int(args.separation_min_selected_features)):
        logger.warning(
            f"[FeatFilter] Selected features too few ({len(feature_pool)}). "
            f"Need >= {args.separation_min_selected_features}. Feature filter disabled."
        )
        return None
    logger.info(
        f"[FeatFilter] Enabled: selected {len(feature_pool)} features "
        f"(top_k={top_k}, min_ks={args.separation_min_ks}, max_overlap={args.separation_max_overlap}, "
        f"min_abs_d={args.separation_min_abs_cohens_d})."
    )
    logger.info(f"[FeatFilter] Sample: {list(sorted(feature_pool))[:20]}")
    return feature_pool


def load_turn_separation_feature_pools(args):
    """
    Optional per-turn feature pools from turn CSV (feature, turn, abs_variance_diff, ...).
    Returns dict: turn(int) -> set(features)
    """
    if not (getattr(args, "use_separation_feature_filter", False) and getattr(args, "use_separation_turn_dynamic", True)):
        return {}
    allow_missing = bool(getattr(args, "separation_allow_missing_files", False))
    turn_csv = os.path.expanduser(args.separation_turn_csv)
    if not os.path.isfile(turn_csv):
        msg = f"[FeatFilter] separation_turn_csv not found: {turn_csv}."
        if allow_missing:
            logger.warning(msg + " turn-dynamic filter disabled.")
            return {}
        raise FileNotFoundError(msg + " Stop because turn-dynamic filter is enabled.")
    try:
        df = pd.read_csv(turn_csv, low_memory=False)
    except Exception as e:
        msg = f"[FeatFilter] Failed reading separation_turn_csv: {e}."
        if allow_missing:
            logger.warning(msg + " turn-dynamic filter disabled.")
            return {}
        raise RuntimeError(msg + " Stop because turn-dynamic filter is enabled.")
    if "feature" not in df.columns or "turn" not in df.columns:
        msg = "[FeatFilter] separation_turn_csv requires columns: feature, turn."
        if allow_missing:
            logger.warning(msg + " turn-dynamic filter disabled.")
            return {}
        raise ValueError(msg + " Stop because turn-dynamic filter is enabled.")

    work = df.copy()
    if "abs_variance_diff" in work.columns:
        work = work[work["abs_variance_diff"].fillna(0.0) >= args.separation_turn_min_abs_var_diff]
    if "variance_ratio_attack_over_benign" in work.columns:
        r = max(1.0, float(args.separation_turn_min_var_ratio))
        rr = work["variance_ratio_attack_over_benign"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
        work = work[(rr >= r) | (rr <= (1.0 / r))]
    if "ks_statistic" in work.columns:
        work = work[work["ks_statistic"].fillna(-1.0) >= args.separation_turn_min_ks]
    if "q05_q95_overlap_ratio" in work.columns:
        work = work[work["q05_q95_overlap_ratio"].fillna(1.0) <= args.separation_turn_max_overlap]
    if "cohens_d_attack_vs_benign" in work.columns:
        work = work[work["cohens_d_attack_vs_benign"].abs().fillna(0.0) >= args.separation_turn_min_abs_cohens_d]

    pools = {}
    top_k_turn = max(0, int(args.separation_top_k_features_turn))
    for t, g in work.groupby("turn"):
        gg = g.copy()
        rank_cols = []
        ascending = []
        if "ks_statistic" in gg.columns:
            rank_cols.append("ks_statistic")
            ascending.append(False)
        if "separation_score_1_minus_overlap" in gg.columns:
            rank_cols.append("separation_score_1_minus_overlap")
            ascending.append(False)
        if "cohens_d_attack_vs_benign" in gg.columns:
            gg["abs_cohens_d_tmp"] = gg["cohens_d_attack_vs_benign"].abs()
            rank_cols.append("abs_cohens_d_tmp")
            ascending.append(False)
        if "abs_variance_diff" in gg.columns:
            rank_cols.append("abs_variance_diff")
            ascending.append(False)
        if rank_cols:
            gg = gg.sort_values(rank_cols, ascending=ascending)
        if top_k_turn > 0:
            gg = gg.head(top_k_turn)
        pools[int(t)] = set(gg["feature"].astype(str))
    logger.info(f"[FeatFilter] Turn-dynamic pools loaded for {len(pools)} turns from {turn_csv}")
    return pools


def add_separation_filter_args(parser):
    """Register separability-filter related parser arguments."""
    parser.add_argument('--use_separation_feature_filter', action='store_true', help="Use precomputed benign<->attack separation scores to select features for association mining.")
    parser.add_argument('--separation_full_csv', type=str, default=None, help="Full separation score CSV path. Default: ../Seperate_Attack_Ex/<file_type>/var_benign_ack/feature_distribution_separation_full.csv")
    parser.add_argument('--use_separation_stability_filter', action='store_true', help="Also use turn-stability CSV constraints when selecting features.")
    parser.add_argument('--separation_stability_csv', type=str, default=None, help="Turn stability CSV path. Default: ../Seperate_Attack_Ex/<file_type>/var_benign_ack/feature_variance_turn_stability.csv")
    parser.add_argument('--separation_top_k_features', type=int, default=20, help="Top-K separability-ranked features to keep (0 = all passing thresholds).")
    parser.add_argument('--separation_min_ks', type=float, default=0.15, help="Minimum KS statistic for feature selection.")
    parser.add_argument('--separation_max_overlap', type=float, default=0.90, help="Maximum q05-q95 overlap ratio for feature selection.")
    parser.add_argument('--separation_min_abs_cohens_d', type=float, default=0.10, help="Minimum |Cohen's d| for feature selection.")
    parser.add_argument('--separation_min_attack_gt_ratio', type=float, default=0.50, help="(Stability filter) minimum attack_gt_benign_ratio.")
    parser.add_argument('--separation_max_flip_rate', type=float, default=0.45, help="(Stability filter) maximum sign_flip_rate.")
    parser.add_argument('--use_separation_turn_dynamic', action='store_true', default=True, help="Use per-turn separability CSV so selected mining features can differ by turn (default: ON).")
    parser.add_argument('--no_separation_turn_dynamic', action='store_true', help="Disable per-turn separability selection and use global separability pool only.")
    parser.add_argument('--separation_turn_csv', type=str, default=None, help="Per-turn separation CSV path (feature, turn, ks/overlap/d/variance metrics). Default: ../Seperate_Attack_Ex/<file_type>/var_benign_ack/feature_distribution_separation_turn.csv")
    parser.add_argument('--separation_top_k_features_turn', type=int, default=20, help="Top-K features per turn from separation_turn_csv (0 = all passing thresholds).")
    parser.add_argument('--separation_turn_min_abs_var_diff', type=float, default=0.0, help="Minimum abs_variance_diff for per-turn feature selection.")
    parser.add_argument('--separation_turn_min_var_ratio', type=float, default=1.0, help="Minimum variance ratio distance from 1.0 for per-turn feature selection.")
    parser.add_argument('--separation_turn_min_ks', type=float, default=0.15, help="(Turn-dynamic) minimum KS statistic for feature selection.")
    parser.add_argument('--separation_turn_max_overlap', type=float, default=0.90, help="(Turn-dynamic) maximum q05-q95 overlap ratio for feature selection.")
    parser.add_argument('--separation_turn_min_abs_cohens_d', type=float, default=0.10, help="(Turn-dynamic) minimum |Cohen's d| for feature selection.")
    parser.add_argument('--separation_allow_missing_files', action='store_true', help="Allow missing/invalid separation CSVs and disable filter instead of stopping.")
    parser.add_argument('--separation_min_selected_features', type=int, default=5, help="Minimum features required globally; otherwise disable feature filter.")
    parser.add_argument('--separation_min_cols_per_turn', type=int, default=3, help="Minimum selected columns required in a turn to apply feature filter.")


def apply_separation_cli_postprocess(cli_args):
    """Apply CLI post-processing for separability args."""
    if getattr(cli_args, "no_separation_turn_dynamic", False):
        cli_args.use_separation_turn_dynamic = False
    defaults = _default_separation_paths(getattr(cli_args, "file_type", None))
    if not getattr(cli_args, "separation_full_csv", None):
        cli_args.separation_full_csv = defaults["full"]
    if not getattr(cli_args, "separation_turn_csv", None):
        cli_args.separation_turn_csv = defaults["turn"]
    if not getattr(cli_args, "separation_stability_csv", None):
        cli_args.separation_stability_csv = defaults["stability"]

