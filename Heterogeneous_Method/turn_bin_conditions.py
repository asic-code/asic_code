import re
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from Modules.Heterogeneous_module import choose_heterogeneous_method
from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups
from utils.class_row import get_label_columns_to_exclude

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False
    njit = None
    prange = None

try:
    import rule_eval_c
    #HAVE_RULE_EVAL_C = True
    HAVE_RULE_EVAL_C = getattr(rule_eval_c, "eval_rules_c", None) is not None
    if not HAVE_RULE_EVAL_C:
        rule_eval_c = None
except Exception:
    HAVE_RULE_EVAL_C = False
    rule_eval_c = None

RANGE_MATCH_ENGINE = "bool"  # "bool" or "bitset"
_POPCOUNT_LUT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)
RANGE_MATCH_JIT = "none"  # "none" or "numba"
USE_RULE_EVAL_C = True  # use C extension for bitset kernel when available (disable via set_use_rule_eval_c(False))


def set_range_match_engine(engine: str):
    global RANGE_MATCH_ENGINE
    if engine in ("bool", "bitset"):
        RANGE_MATCH_ENGINE = engine


def set_range_match_jit(engine: str):
    global RANGE_MATCH_JIT
    if engine in ("none", "numba"):
        RANGE_MATCH_JIT = engine


def set_use_rule_eval_c(use: bool):
    """Enable/disable the pure C rule evaluation kernel (when installed)."""
    global USE_RULE_EVAL_C
    USE_RULE_EVAL_C = bool(use)


if HAVE_NUMBA:
    @njit(cache=True)
    def _and_reduce_packed(mask_matrix):
        out = mask_matrix[0].copy()
        for i in range(1, mask_matrix.shape[0]):
            out &= mask_matrix[i]
        return out

    @njit(parallel=True, cache=True)
    def _and_reduce_packed_blocked(mask_matrix, block_size):
        n_masks = mask_matrix.shape[0]
        n_bytes = mask_matrix.shape[1]
        n_blocks = (n_masks + block_size - 1) // block_size
        block_out = np.empty((n_blocks, n_bytes), dtype=np.uint8)
        for b in prange(n_blocks):
            start = b * block_size
            end = min(start + block_size, n_masks)
            acc = mask_matrix[start].copy()
            for i in range(start + 1, end):
                acc &= mask_matrix[i]
            block_out[b] = acc
        return _and_reduce_packed(block_out)

    @njit(cache=True)
    def _popcount_packed_numba(packed, lut):
        total = 0
        for i in range(packed.shape[0]):
            total += lut[packed[i]]
        return total


def get_attack_columns_for_report(file_type: str):
    if file_type in ['CICIDS2017']:
        return ['Label']
    if file_type in ['CICIoT2023', 'CICIoT']:
        return ['attack_name']
    if file_type in ['DARPA98', 'DARPA']:
        return ['Class']
    if file_type in ['MiraiBotnet']:
        return ['reconnaissance', 'infection', 'action']
    if file_type in ['netML']:
        return ['Label']
    if file_type in ['NSL-KDD', 'NSL_KDD']:
        return ['class']
    return []

def _parse_interval_string(interval_str):
    if not isinstance(interval_str, str):
        return None
    interval_str = interval_str.strip()
    if not interval_str:
        return None
    left_bracket = interval_str[0]
    right_bracket = interval_str[-1]
    # Only treat canonical interval literals as intervals.
    # This prevents categorical strings like IPs ("172.016.112.050")
    # from being mis-parsed into numeric ranges.
    if left_bracket not in ("[", "(") or right_bracket not in ("]", ")"):
        return None
    nums = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', interval_str)
    if len(nums) != 2:
        return None
    try:
        left_val = float(nums[0])
        right_val = float(nums[1])
    except ValueError:
        return None
    left_inclusive = left_bracket == '['
    right_inclusive = right_bracket == ']'
    return (left_val, right_val, left_inclusive, right_inclusive)


def _normalize_rule_conditions(rule):
    normalized = {}
    for key, val in rule.items():
        if isinstance(val, tuple) and len(val) == 4 and all(isinstance(v, (int, float, bool)) for v in val):
            normalized[key] = val
            continue
        if isinstance(val, (list, tuple)) and len(val) == 2:
            try:
                normalized[key] = (float(val[0]), float(val[1]), True, True)
                continue
            except (TypeError, ValueError):
                pass
        parsed = _parse_interval_string(val) if isinstance(val, str) else None
        if parsed is not None:
            normalized[key] = parsed
        else:
            normalized[key] = val
    return normalized


def _rule_looks_range(rule):
    if not isinstance(rule, dict):
        return False
    for val in rule.values():
        if isinstance(val, tuple) and len(val) == 4:
            return True
        if isinstance(val, str) and _parse_interval_string(val):
            return True
    return False


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
    # String cond that parses as number: compare numerically so 0.0 matches '0'
    if isinstance(cond, str):
        try:
            cond_num = float(cond)
            numeric_series = pd.to_numeric(series, errors='coerce')
            return numeric_series == cond_num
        except (ValueError, TypeError):
            pass
    return series.astype(str) == str(cond)


def try_extract_attack_types_c_ext(sorted_sigs, chunk_cache_obj, attack_cols, batch_size,
                                   max_out_masks_bytes=200_000_000, cond_limit=5000):
    if not attack_cols:
        return {}
    if not USE_RULE_EVAL_C or not HAVE_RULE_EVAL_C:
        return None

    # Build condition-id mapping and CSR rule matrix once
    cond_key_to_id = {}
    cond_id_to_pair = []
    rule_lengths = []
    rule_cond_ids_list = []
    sig_ids = []
    for sig_id, sig in sorted_sigs:
        sig_ids.append(sig_id)
        if not isinstance(sig, dict) or not sig:
            rule_lengths.append(0)
            continue
        cond_ids = []
        for col, cond in sig.items():
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
        return {}
    if n_conds > cond_limit:
        return None

    rule_ptr = np.zeros(n_rules + 1, dtype=np.int64)
    if rule_lengths:
        rule_ptr[1:] = np.cumsum(rule_lengths)
    rule_cond_ids = np.array(rule_cond_ids_list, dtype=np.int64)

    attacks_by_sig = {sid: set() for sid in sig_ids}

    for batch in chunk_cache_obj.iter_batches(batch_size=batch_size):
        if batch is None or batch.empty:
            continue
        n_rows = len(batch)
        n_bytes = (n_rows + 7) // 8
        block_rules = max(1, max_out_masks_bytes // max(1, n_bytes))
        block_rules = min(block_rules, n_rules)

        # Build condition bitsets for this batch
        cond_bitsets = np.empty((n_conds, n_bytes), dtype=np.uint8)
        for cond_id, (col, cond) in enumerate(cond_id_to_pair):
            if col not in batch.columns:
                cond_bitsets[cond_id, :] = 0
                continue
            mask = _build_condition_mask(batch[col], cond)
            cond_bitsets[cond_id, :] = np.packbits(mask.to_numpy(dtype=np.uint8))

        for block_start in range(0, n_rules, block_rules):
            block_end = min(n_rules, block_start + block_rules)
            ptr_start = rule_ptr[block_start]
            ptr_end = rule_ptr[block_end]
            rule_ptr_sub = (rule_ptr[block_start:block_end + 1] - ptr_start).astype(np.int64, copy=False)
            rule_cond_ids_sub = rule_cond_ids[ptr_start:ptr_end].astype(np.int64, copy=False)

            out_masks = np.empty((block_end - block_start, n_bytes), dtype=np.uint8)
            try:
                scores = rule_eval_c.eval_rules_c(
                    np.ascontiguousarray(rule_ptr_sub),
                    np.ascontiguousarray(rule_cond_ids_sub),
                    np.ascontiguousarray(cond_bitsets),
                    n_conds,
                    n_bytes,
                    out_masks
                )
            except Exception:
                return None

            for local_idx, score in enumerate(scores):
                if score == 0:
                    continue
                rule_idx = block_start + local_idx
                sig_id = sig_ids[rule_idx]
                mask_bool = np.unpackbits(out_masks[local_idx])[:n_rows].astype(bool)
                if not mask_bool.any():
                    continue
                matched = batch.loc[mask_bool]

                if attack_cols == ['reconnaissance', 'infection', 'action']:
                    for col in attack_cols:
                        if col in matched.columns:
                            vals = matched[col]
                            if (vals == 1).any():
                                attacks_by_sig[sig_id].add(col)
                    continue

                for col in attack_cols:
                    if col in matched.columns:
                        vals = matched[col].astype(str).unique().tolist()
                        attacks_by_sig[sig_id].update(vals)

    for sid, attacks in attacks_by_sig.items():
        if not attacks:
            attacks.add('-')
    return attacks_by_sig


def apply_range_signatures_to_dataset(
    df,
    signatures,
    base_time=datetime(2025, 4, 14, 12, 0, 0),
    show_progress=True,
    log_info=True,
):
    if df is None or df.empty or not signatures:
        return pd.DataFrame()

    df = df.copy()
    label_cols_present = [c for c in ['label', 'class', 'Class', 'cluster'] if c in df.columns]
    df['_original_index'] = df.index
    if not df.index.is_unique:
        df = df.reset_index(drop=True)

    temp_df = pd.DataFrame(index=df.index)
    temp_df['_match_sig_id'] = pd.NA
    temp_df['_row_index'] = np.arange(len(df))

    row_count = len(df)
    max_cache_masks = max(16, min(512, int(100_000_000 / max(1, row_count))))
    condition_cache = OrderedDict()
    condition_count_cache = {}

    def _build_bool_mask(col, cond):
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        series = df[col]
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
        # String cond that parses as number: compare numerically so 0.0 matches '0'
        if isinstance(cond, str):
            try:
                cond_num = float(cond)
                numeric_series = pd.to_numeric(series, errors='coerce')
                return numeric_series == cond_num
            except (ValueError, TypeError):
                pass
        return series.astype(str) == str(cond)

    def _pack_mask(mask_bool):
        return np.packbits(mask_bool.to_numpy(dtype=np.uint8))

    def _popcount(packed):
        return int(_POPCOUNT_LUT[packed].sum())

    def _get_condition_mask(col, cond, store=True):
        key = (col, str(cond))
        if key in condition_cache:
            return condition_cache[key]

        mask_bool = _build_bool_mask(col, cond)
        if RANGE_MATCH_ENGINE == "bitset":
            mask = _pack_mask(mask_bool)
        else:
            mask = mask_bool

        if store:
            if len(condition_cache) >= max_cache_masks:
                condition_cache.popitem(last=False)
            condition_cache[key] = mask
        return mask

    # Build sparse rule matrix (CSR): rule_ptr + rule_cond_ids
    cond_key_to_id = {}
    cond_id_to_pair = []
    sig_ids = []
    rule_lengths = []
    rule_cond_ids_list = []
    for sig_info in signatures:
        sig_id = sig_info.get('id', 'UNKNOWN_ID') if isinstance(sig_info, dict) else 'UNKNOWN_ID'
        rule = sig_info.get('rule_dict') if isinstance(sig_info, dict) else sig_info
        sig_ids.append(sig_id)
        if not isinstance(rule, dict) or not rule:
            rule_lengths.append(0)
            continue
        rule = _normalize_rule_conditions(rule)
        cond_ids = []
        for col, cond in rule.items():
            key = (col, str(cond))
            cond_id = cond_key_to_id.get(key)
            if cond_id is None:
                cond_id = len(cond_id_to_pair)
                cond_key_to_id[key] = cond_id
                cond_id_to_pair.append((col, cond))
            cond_ids.append(cond_id)
        rule_lengths.append(len(cond_ids))
        rule_cond_ids_list.extend(cond_ids)

    rule_ptr = np.zeros(len(rule_lengths) + 1, dtype=np.int64)
    if rule_lengths:
        rule_ptr[1:] = np.cumsum(rule_lengths)
    rule_cond_ids = np.array(rule_cond_ids_list, dtype=np.int64)

    # Optional: precompute all condition masks/counts when reasonable (rule batch eval).
    precompute_all = len(cond_id_to_pair) <= 2000
    if precompute_all:
        for cond_id, (col, cond) in enumerate(cond_id_to_pair):
            cond_mask = _get_condition_mask(col, cond, store=len(condition_cache) < max_cache_masks)
            if RANGE_MATCH_ENGINE == "bitset":
                condition_count_cache[cond_id] = _popcount(cond_mask)
            else:
                condition_count_cache[cond_id] = int(cond_mask.sum())

    use_numba = HAVE_NUMBA and RANGE_MATCH_JIT == "numba" and RANGE_MATCH_ENGINE == "bitset"
    block_size = 64

    c_ext_scores = None
    c_ext_masks = None
    if RANGE_MATCH_ENGINE == "bitset" and (not USE_RULE_EVAL_C or not HAVE_RULE_EVAL_C) and log_info:
        print("Info: C extension not available (rule_eval_c); using Numba/NumPy fallback for bitset AND.")
    if (
        RANGE_MATCH_ENGINE == "bitset"
        and USE_RULE_EVAL_C
        and HAVE_RULE_EVAL_C
        and precompute_all
        and len(sig_ids) > 0
        and len(cond_id_to_pair) > 0
    ):
        n_conds = len(cond_id_to_pair)
        n_bytes = (row_count + 7) // 8
        n_rules = len(sig_ids)
        cond_bitsets = np.empty((n_conds, n_bytes), dtype=np.uint8)
        for cond_id, (col, cond) in enumerate(cond_id_to_pair):
            key = (col, str(cond))
            if key in condition_cache:
                cond_bitsets[cond_id, :] = condition_cache[key]
            else:
                m = _get_condition_mask(col, cond, store=len(condition_cache) < max_cache_masks)
                cond_bitsets[cond_id, :] = np.asarray(m, dtype=np.uint8).reshape(-1)[:n_bytes]
        rule_ptr_sorted_list = [0]
        rule_cond_ids_sorted_list = []
        for ri in range(n_rules):
            s, e = int(rule_ptr[ri]), int(rule_ptr[ri + 1])
            if s == e:
                rule_ptr_sorted_list.append(rule_ptr_sorted_list[-1])
                continue
            ids = list(rule_cond_ids[s:e])
            ids.sort(key=lambda c: condition_count_cache.get(int(c), 0))
            rule_cond_ids_sorted_list.extend(ids)
            rule_ptr_sorted_list.append(rule_ptr_sorted_list[-1] + len(ids))
        rule_ptr_sorted = np.ascontiguousarray(np.array(rule_ptr_sorted_list, dtype=np.int64))
        rule_cond_ids_sorted = np.ascontiguousarray(np.array(rule_cond_ids_sorted_list, dtype=np.int64))
        cond_bitsets = np.ascontiguousarray(cond_bitsets)
        out_masks = np.empty((n_rules, n_bytes), dtype=np.uint8)
        try:
            scores_list = rule_eval_c.eval_rules_c(
                rule_ptr_sorted, rule_cond_ids_sorted, cond_bitsets, n_conds, n_bytes, out_masks
            )
            c_ext_scores = np.array(scores_list, dtype=np.int64)
            c_ext_masks = out_masks
            if log_info:
                print("Info: Using rule_eval_c kernel for bitset evaluation (scores + row masks).")
        except Exception as e:
            c_ext_scores = None
            c_ext_masks = None
            if log_info:
                print("Info: rule_eval_c.eval_rules_c failed (using Numba/NumPy fallback):", type(e).__name__, str(e))

    pbar = tqdm(range(len(sig_ids)), desc="[EVAL] Applying Signatures", disable=not show_progress)
    for rule_idx in pbar:
        sig_id = sig_ids[rule_idx]
        start = rule_ptr[rule_idx]
        end = rule_ptr[rule_idx + 1]
        if start == end:
            continue
        if c_ext_scores is not None and c_ext_scores[rule_idx] == 0:
            continue
        if c_ext_scores is not None and c_ext_masks is not None:
            # Full C path: row-level mask already in c_ext_masks; Python only applies result.
            mask_bool = np.unpackbits(c_ext_masks[rule_idx])[:row_count].astype(bool)
            unmatched = df.index[mask_bool & temp_df["_match_sig_id"].isna()]
            if not unmatched.empty:
                temp_df.loc[unmatched, "_match_sig_id"] = sig_id
            continue
        cond_infos = []
        for cond_id in rule_cond_ids[start:end]:
            if cond_id in condition_count_cache:
                count = condition_count_cache[cond_id]
            else:
                col, cond = cond_id_to_pair[cond_id]
                cond_mask = _get_condition_mask(col, cond, store=len(condition_cache) < max_cache_masks)
                if RANGE_MATCH_ENGINE == "bitset":
                    count = _popcount(cond_mask)
                else:
                    count = int(cond_mask.sum())
                condition_count_cache[cond_id] = count
            cond_infos.append((count, cond_id))
        cond_infos.sort(key=lambda x: x[0])

        if RANGE_MATCH_ENGINE == "bitset":
            # When C extension was used (c_ext_scores is not None), use NumPy path only;
            # Numba is not used so C ON ⇒ no Numba dependency.
            if use_numba and cond_infos and (c_ext_scores is None):
                mask_stack = []
                for _, cond_id in cond_infos:
                    col, cond = cond_id_to_pair[cond_id]
                    cond_mask = _get_condition_mask(col, cond, store=len(condition_cache) < max_cache_masks)
                    mask_stack.append(cond_mask)
                mask_matrix = np.stack(mask_stack, axis=0)
                if mask_matrix.shape[0] >= block_size:
                    rule_mask = _and_reduce_packed_blocked(mask_matrix, block_size)
                else:
                    rule_mask = _and_reduce_packed(mask_matrix)
                if _popcount_packed_numba(rule_mask, _POPCOUNT_LUT) > 0:
                    mask_bool = np.unpackbits(rule_mask)[:row_count].astype(bool)
                    unmatched = df.index[mask_bool & temp_df['_match_sig_id'].isna()]
                    if not unmatched.empty:
                        temp_df.loc[unmatched, '_match_sig_id'] = sig_id
            else:
                rule_mask = None
                for _, cond_id in cond_infos:
                    col, cond = cond_id_to_pair[cond_id]
                    cond_mask = _get_condition_mask(col, cond, store=len(condition_cache) < max_cache_masks)
                    rule_mask = cond_mask if rule_mask is None else np.bitwise_and(rule_mask, cond_mask)
                    if _popcount(rule_mask) == 0:
                        break
                if rule_mask is not None and _popcount(rule_mask) > 0:
                    mask_bool = np.unpackbits(rule_mask)[:row_count].astype(bool)
                    unmatched = df.index[mask_bool & temp_df['_match_sig_id'].isna()]
                    if not unmatched.empty:
                        temp_df.loc[unmatched, '_match_sig_id'] = sig_id
        else:
            mask = pd.Series(True, index=df.index)
            for _, cond_id in cond_infos:
                col, cond = cond_id_to_pair[cond_id]
                cond_mask = _get_condition_mask(col, cond, store=len(condition_cache) < max_cache_masks)
                mask &= cond_mask
                if not mask.any():
                    break
            if mask.any():
                unmatched = df.index[mask & temp_df['_match_sig_id'].isna()]
                if not unmatched.empty:
                    temp_df.loc[unmatched, '_match_sig_id'] = sig_id

    alerts_df_raw = temp_df[temp_df['_match_sig_id'].notna()].copy()
    if alerts_df_raw.empty:
        if log_info:
            print("Info: No alerts generated after applying all signatures.")
        return pd.DataFrame()

    alerts_df_raw = alerts_df_raw.join(df[label_cols_present + ['_original_index']], lsuffix='_left', rsuffix='_right')
    alerts_final = pd.DataFrame({
        'alert_index': alerts_df_raw['_original_index'].values,
        'timestamp': alerts_df_raw['_row_index'].apply(lambda i: base_time + pd.Timedelta(seconds=i * 2)),
        'src_ip': [f"192.168.1.{np.random.randint(1, 255)}" for _ in range(len(alerts_df_raw))],
        'dst_ip': [f"10.0.0.{np.random.randint(1, 255)}" for _ in range(len(alerts_df_raw))],
        'signature_id': alerts_df_raw['_match_sig_id'],
        'signature_name': alerts_df_raw['_match_sig_id'].map({s.get('id'): s.get('name', 'UNKNOWN_NAME') for s in signatures if isinstance(s, dict)})
    })
    for col in label_cols_present:
        if col in alerts_df_raw.columns:
            alerts_final[col] = alerts_df_raw[col].values

    if log_info:
        print(f"Info: Generated {len(alerts_final)} raw alerts.")
    return alerts_final


def _patch_range_matchers():
    try:
        import Rebuild_Method.FalsePositive_Check as _fp_check
        _fp_check.apply_signatures_to_dataset = apply_range_signatures_to_dataset
    except Exception:
        pass
    try:
        import Rebuild_Method.fp_diagnostics as _fp_diag
        _fp_diag.apply_signatures_to_dataset = apply_range_signatures_to_dataset
    except Exception:
        pass


def preprocess_and_map_chunk_turn(chunk_df, file_type, n_splits=None, debug=False, debug_context=None):
    """
    Per-turn mapping: build mapping from current chunk only and map this chunk.
    Returns mapped_chunk, category_mapping, data_list, mapping_features_original.
    """
    label_cols_to_exclude = get_label_columns_to_exclude(file_type)
    mapping_features = chunk_df.drop(columns=label_cols_to_exclude, errors='ignore')
    mapping_features_original = mapping_features.copy()

    embedded_dataframe, _, category_mapping, data_list = choose_heterogeneous_method(
        mapping_features,
        file_type,
        'Interval_inverse',
        'N',
        n_splits_override=n_splits,
    )

    # Use existing_mapping from this turn to ensure consistent mapping within the chunk
    chunk_embedded, _, _, chunk_data_list = choose_heterogeneous_method(
        chunk_df,
        file_type,
        'Interval_inverse',
        'N',
        n_splits_override=n_splits,
        existing_mapping=category_mapping,
    )

    # Ensure raw numeric values are used for interval columns
    if isinstance(category_mapping, dict) and isinstance(category_mapping.get('interval'), pd.DataFrame):
        interval_cols = [c for c in category_mapping['interval'].columns if c in chunk_df.columns]
    else:
        interval_cols = []
    chunk_for_mapping = chunk_embedded.copy()
    if interval_cols:
        for c in interval_cols:
            if c in chunk_df.columns:
                chunk_for_mapping[c] = chunk_df[c].values

    mapped_chunk, _ = map_intervals_to_groups(
        chunk_for_mapping,
        category_mapping,
        chunk_data_list,
        'N',
        debug=debug,
        debug_context=debug_context,
    )

    # Preserve label/cluster/adjusted_cluster columns from original chunk
    if 'label' in chunk_df.columns:
        mapped_chunk['label'] = chunk_df['label'].values
    if 'cluster' in chunk_df.columns:
        mapped_chunk['cluster'] = chunk_df['cluster'].values
    if 'adjusted_cluster' in chunk_df.columns:
        mapped_chunk['adjusted_cluster'] = chunk_df['adjusted_cluster'].values

    # Preserve attack columns for reporting only
    attack_cols = get_attack_columns_for_report(file_type)
    for col in attack_cols:
        if col in chunk_df.columns and col not in mapped_chunk.columns:
            mapped_chunk[col] = chunk_df[col].values

    return mapped_chunk, category_mapping, data_list, mapping_features, mapping_features_original
