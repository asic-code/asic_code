import re
from collections import defaultdict

import numpy as np
import pandas as pd

from utils.chunk_cache import _yield_batches_from_source


def _sanitize_attack_name_for_column(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "unknown"


def _is_benign_like_text(text: str) -> bool:
    t = str(text).strip().lower()
    return t in {"benign", "normal", "normal.", "0", "none", "nan", ""}


def _default_attack_columns_for_report(file_type: str):
    if file_type in ["CICIDS2017"]:
        return ["Label"]
    if file_type in ["CICIoT2023", "CICIoT"]:
        return ["attack_name"]
    if file_type in ["DARPA98", "DARPA"]:
        return ["Class"]
    if file_type in ["MiraiBotnet"]:
        return ["reconnaissance", "infection", "action"]
    if file_type in ["netML"]:
        return ["Label"]
    if file_type in ["NSL-KDD", "NSL_KDD"]:
        return ["class"]
    return []


def build_attack_index_map_full_dataset(data_source, file_type: str, batch_size: int):
    """
    Build attack-type -> global row index set for whole dataset, reporting-only.
    This function is strictly for post-hoc evaluation and must not affect rule generation.
    """
    attack_cols = _default_attack_columns_for_report(file_type)
    if not attack_cols:
        return {}, {}

    attack_to_indices = defaultdict(set)
    for batch_df in _yield_batches_from_source(data_source, batch_size=batch_size):
        if batch_df is None or batch_df.empty:
            continue
        # MiraiBotnet: multi-hot attack columns
        if file_type in ["MiraiBotnet"]:
            for col in attack_cols:
                if col not in batch_df.columns:
                    continue
                vals = pd.to_numeric(batch_df[col], errors="coerce").fillna(0)
                idx = batch_df.index[vals > 0]
                if len(idx) > 0:
                    attack_to_indices[col].update(idx.tolist())
            continue

        # Single attack name column family
        col = next((c for c in attack_cols if c in batch_df.columns), None)
        if col is None:
            continue
        series = batch_df[col].astype(str).str.strip()
        if series.empty:
            continue
        unique_vals = series.unique().tolist()
        for val in unique_vals:
            if _is_benign_like_text(val):
                continue
            idx = batch_df.index[series == val]
            if len(idx) > 0:
                attack_to_indices[str(val)].update(idx.tolist())

    attack_totals = {k: len(v) for k, v in attack_to_indices.items() if len(v) > 0}
    return dict(attack_to_indices), attack_totals


def build_attack_index_map_by_turn_full_dataset(data_source, file_type: str, batch_size: int):
    """
    Build attack-type -> turn -> global row index set.
    Used for turn-locked cumulative attackwise recall reporting.
    """
    attack_cols = _default_attack_columns_for_report(file_type)
    if not attack_cols:
        return {}

    attack_to_indices_by_turn = defaultdict(lambda: defaultdict(set))

    # Prefer exact turn metadata when available (ChunkCache.iter_chunks_with_turn).
    # This avoids incorrect turn inference from batch slicing boundaries.
    if hasattr(data_source, "iter_chunks_with_turn"):
        for turn_id, batch_df in data_source.iter_chunks_with_turn():
            if batch_df is None or batch_df.empty:
                continue
            turn_int = int(turn_id)

            if file_type in ["MiraiBotnet"]:
                for col in attack_cols:
                    if col not in batch_df.columns:
                        continue
                    vals = pd.to_numeric(batch_df[col], errors="coerce").fillna(0)
                    idx = batch_df.index[vals > 0]
                    if len(idx) > 0:
                        attack_to_indices_by_turn[col][turn_int].update(idx.tolist())
                continue

            col = next((c for c in attack_cols if c in batch_df.columns), None)
            if col is None:
                continue
            series = batch_df[col].astype(str).str.strip()
            if series.empty:
                continue
            for val in series.unique().tolist():
                if _is_benign_like_text(val):
                    continue
                idx = batch_df.index[series == val]
                if len(idx) > 0:
                    attack_to_indices_by_turn[str(val)][turn_int].update(idx.tolist())

    else:
        for batch_df in _yield_batches_from_source(data_source, batch_size=batch_size):
            if batch_df is None or batch_df.empty:
                continue

            # Prefer explicit per-row turn id if present; otherwise infer from chunk_size boundary.
            # IMPORTANT: keep as pandas Series indexed by batch_df.index so `.loc[...]` is always valid.
            if "turn" in batch_df.columns:
                turn_series = pd.to_numeric(batch_df["turn"], errors="coerce")
                if not isinstance(turn_series, pd.Series):
                    turn_series = pd.Series(turn_series, index=batch_df.index)
                elif not turn_series.index.equals(batch_df.index):
                    turn_series = pd.Series(turn_series.to_numpy(), index=batch_df.index)
            else:
                inferred_turns = pd.to_numeric(
                    np.floor(pd.to_numeric(batch_df.index, errors="coerce") / float(max(1, batch_size))) + 1,
                    errors="coerce",
                )
                turn_series = pd.Series(np.asarray(inferred_turns), index=batch_df.index)

            if file_type in ["MiraiBotnet"]:
                for col in attack_cols:
                    if col not in batch_df.columns:
                        continue
                    vals = pd.to_numeric(batch_df[col], errors="coerce").fillna(0)
                    mask_attack = vals > 0
                    if not mask_attack.any():
                        continue
                    attacked = batch_df.loc[mask_attack]
                    attacked_turns = turn_series.loc[attacked.index]
                    for t in attacked_turns.dropna().astype(int).unique().tolist():
                        idx_t = attacked.index[attacked_turns == t]
                        if len(idx_t) > 0:
                            attack_to_indices_by_turn[col][int(t)].update(idx_t.tolist())
                continue

            col = next((c for c in attack_cols if c in batch_df.columns), None)
            if col is None:
                continue
            series = batch_df[col].astype(str).str.strip()
            if series.empty:
                continue
            for val in series.unique().tolist():
                if _is_benign_like_text(val):
                    continue
                mask_val = series == val
                selected = batch_df.loc[mask_val]
                selected_turns = turn_series.loc[selected.index]
                for t in selected_turns.dropna().astype(int).unique().tolist():
                    idx_t = selected.index[selected_turns == t]
                    if len(idx_t) > 0:
                        attack_to_indices_by_turn[str(val)][int(t)].update(idx_t.tolist())

    out = {}
    for attack_name, by_turn in attack_to_indices_by_turn.items():
        clean = {int(t): set(v) for t, v in by_turn.items() if len(v) > 0}
        if clean:
            out[attack_name] = clean
    return out


def compute_attackwise_recall_from_alerts(alerts_df: pd.DataFrame, attack_to_indices: dict, attack_totals: dict):
    """
    Compute recall per attack type over the FULL dataset index space.
    Recall(type) = alerted_rows_of_type / total_rows_of_type
    """
    if not attack_totals:
        return {}
    if alerts_df is None or alerts_df.empty or "alert_index" not in alerts_df.columns:
        alerted = set()
    else:
        alerted = set(pd.to_numeric(alerts_df["alert_index"], errors="coerce").dropna().astype(int).tolist())

    out = {}
    for attack_name, total in attack_totals.items():
        if total <= 0:
            out[attack_name] = np.nan
            continue
        hit = len(alerted.intersection(attack_to_indices.get(attack_name, set())))
        out[attack_name] = float(hit / float(total))
    return out


def compute_attackwise_recall_turn_locked_cumulative(
    alerts_df: pd.DataFrame,
    attack_to_indices_by_turn: dict,
    attack_totals: dict,
    turn_counter: int,
    captured_state: dict,
):
    """
    Turn-locked cumulative recall:
    - At turn t, only attack samples from turn t can be newly captured.
    - Captured samples persist in recall for later turns.
    - Missed samples from earlier turns cannot be captured retroactively.
    Returns: (recall_dict, updated_captured_state)
    """
    st = captured_state if isinstance(captured_state, dict) else {}
    if alerts_df is None or alerts_df.empty or "alert_index" not in alerts_df.columns:
        alerted = set()
    else:
        alerted = set(pd.to_numeric(alerts_df["alert_index"], errors="coerce").dropna().astype(int).tolist())

    out = {}
    turn_int = int(turn_counter)
    for attack_name, total in attack_totals.items():
        if total <= 0:
            out[attack_name] = np.nan
            continue
        if attack_name not in st:
            st[attack_name] = set()
        turn_pool = set(attack_to_indices_by_turn.get(attack_name, {}).get(turn_int, set()))
        if turn_pool:
            st[attack_name].update(alerted.intersection(turn_pool))
        out[attack_name] = float(len(st[attack_name]) / float(total))
    return out, st
