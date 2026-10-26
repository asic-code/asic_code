import numpy as np
import pandas as pd

try:
    import rule_eval_c
    HAVE_RULE_EVAL_C = True
except Exception:
    HAVE_RULE_EVAL_C = False
    rule_eval_c = None


# PCI Filter (in Main_Association_Rule) Exclusive
def _build_condition_mask(series, cond):
    if isinstance(cond, tuple) and len(cond) == 4:
        left_val, right_val, left_inclusive, right_inclusive = cond
        numeric_series = pd.to_numeric(series, errors='coerce')
        if left_inclusive:
            left_mask = numeric_series >= left_val
        else:
            left_mask = numeric_series > left_val
        if right_inclusive:
            right_mask = numeric_series <= right_val
        else:
            right_mask = numeric_series < right_val
        return left_mask & right_mask
    if pd.api.types.is_numeric_dtype(series.dtype) and isinstance(cond, (int, float)):
        return series.astype(float) == float(cond)
    return series.astype(str) == str(cond)


def filter_rule_batch_c_ext(rule_batch, normal_df, negative_filtering_enabled,
                            negative_filter_threshold, normal_min_support,
                            max_cond_bytes=200_000_000):
    if not HAVE_RULE_EVAL_C or rule_eval_c is None:
        return None
    if not rule_batch:
        return []
    if normal_df is None or normal_df.empty:
        return list(rule_batch)

    total_rows = len(normal_df)
    n_bytes = (total_rows + 7) // 8

    cond_key_to_id = {}
    cond_id_to_pair = []
    rule_lengths = []
    rule_cond_ids_list = []
    for r in rule_batch:
        if not isinstance(r, dict) or not r:
            rule_lengths.append(0)
            continue
        cond_ids = []
        for col, cond in r.items():
            key = (col, str(cond))
            cond_id = cond_key_to_id.get(key)
            if cond_id is None:
                cond_id = len(cond_id_to_pair)
                cond_key_to_id[key] = cond_id
                cond_id_to_pair.append((col, cond))
            cond_ids.append(cond_id)
        rule_lengths.append(len(cond_ids))
        rule_cond_ids_list.extend(cond_ids)

    n_rules = len(rule_lengths)
    n_conds = len(cond_id_to_pair)
    if n_rules == 0 or n_conds == 0:
        return list(rule_batch)
    if n_conds * n_bytes > max_cond_bytes:
        return None

    rule_ptr = np.zeros(n_rules + 1, dtype=np.int64)
    if rule_lengths:
        rule_ptr[1:] = np.cumsum(rule_lengths)
    rule_cond_ids = np.array(rule_cond_ids_list, dtype=np.int64)

    cond_bitsets = np.empty((n_conds, n_bytes), dtype=np.uint8)
    for cond_id, (col, cond) in enumerate(cond_id_to_pair):
        if col not in normal_df.columns:
            cond_bitsets[cond_id, :] = 0
            continue
        mask = _build_condition_mask(normal_df[col], cond)
        cond_bitsets[cond_id, :] = np.packbits(mask.to_numpy(dtype=np.uint8))

    try:
        scores = rule_eval_c.eval_rules_c(
            np.ascontiguousarray(rule_ptr),
            np.ascontiguousarray(rule_cond_ids),
            np.ascontiguousarray(cond_bitsets),
            n_conds,
            n_bytes
        )
    except Exception:
        return None

    filtered = []
    for idx, r in enumerate(rule_batch):
        matches = scores[idx]
        support = (matches / total_rows) if total_rows > 0 else 0
        if negative_filtering_enabled and support > negative_filter_threshold:
            continue
        if support > normal_min_support:
            continue
        filtered.append(r)
    return filtered
