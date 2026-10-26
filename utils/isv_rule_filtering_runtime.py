import logging
import multiprocessing

import pandas as pd

from utils.isv_filtering import init_worker_filter_batch, is_rule_valid_for_filtering_batch


logger = logging.getLogger(__name__)


def rule_passes_global(
    rule,
    *,
    is_fallback,
    normal_data_empty,
    normal_data_chunks,
    strict_normal_zero,
    anomalous_data_in_chunk,
    negative_filtering_enabled,
    negative_filter_threshold,
    normal_min_support_effective,
):
    """Evaluate a rule with global support over all normal chunks."""
    if normal_data_empty:
        return True

    normal_total_rows = 0
    normal_total_matches = 0
    for chunk in normal_data_chunks:
        if chunk is None or chunk.empty:
            continue
        normal_total_rows += len(chunk)

        rule_keys_in_chunk = [k for k in rule.keys() if k in chunk.columns]
        if not rule_keys_in_chunk:
            continue

        rule_series = pd.Series({k: rule[k] for k in rule_keys_in_chunk})
        match_mask = chunk[rule_keys_in_chunk].eq(rule_series).all(axis=1)
        normal_total_matches += match_mask.sum()

    if normal_total_rows == 0:
        return True

    normal_support = normal_total_matches / normal_total_rows

    if strict_normal_zero and normal_total_matches > 0:
        return False

    if is_fallback:
        if not anomalous_data_in_chunk.empty:
            anomaly_total_rows = len(anomalous_data_in_chunk)
            rule_keys_in_anomaly = [k for k in rule.keys() if k in anomalous_data_in_chunk.columns]
            if rule_keys_in_anomaly:
                rule_series_anomaly = pd.Series({k: rule[k] for k in rule_keys_in_anomaly})
                anomaly_match_mask = anomalous_data_in_chunk[rule_keys_in_anomaly].eq(rule_series_anomaly).all(axis=1)
                anomaly_total_matches = anomaly_match_mask.sum()
                anomaly_support = anomaly_total_matches / anomaly_total_rows if anomaly_total_rows > 0 else 0.0
            else:
                anomaly_support = 0.0
        else:
            anomaly_support = 0.0

        if normal_support >= 0.95 and anomaly_support >= 0.95:
            logger.warning(
                f"[FallbackRule] Rejected: Cannot distinguish normal from anomaly. "
                f"Normal support: {normal_support:.4f}, Anomaly support: {anomaly_support:.4f}. "
                f"This rule matches almost all data and is not useful for detection."
            )
            return False

        logger.info(
            f"[FallbackRule] Accepted: Normal support: {normal_support:.4f}, "
            f"Anomaly support: {anomaly_support:.4f}. Rule can distinguish between normal and anomaly."
        )
        return True

    if negative_filtering_enabled and normal_support > negative_filter_threshold:
        return False
    if normal_support > normal_min_support_effective:
        return False

    return True


def filter_rule_batch_runtime(
    rule_batch,
    *,
    check_fallback,
    dominant_cols,
    normal_data_empty,
    num_processes,
    normal_data_chunks,
    normal_min_support_effective,
    negative_filtering_enabled,
    negative_filter_threshold,
    strict_normal_zero,
    anomalous_data_in_chunk,
):
    """Filter one batch of rules while preserving existing fallback semantics."""
    if not rule_batch:
        return []
    if normal_data_empty:
        return list(rule_batch)

    filtered = []
    try:
        pool_processes = min(len(rule_batch), num_processes) if num_processes else len(rule_batch)
        if pool_processes > 1:
            chunksize = max(1, len(rule_batch) // (pool_processes * 4))
            with multiprocessing.Pool(
                processes=pool_processes,
                initializer=init_worker_filter_batch,
                initargs=(normal_data_chunks, normal_min_support_effective, negative_filtering_enabled, negative_filter_threshold),
            ) as pool:
                results_iter = pool.imap_unordered(is_rule_valid_for_filtering_batch, rule_batch, chunksize=chunksize)
                filtered = [r for r in results_iter if r is not None]
        else:
            init_worker_filter_batch(normal_data_chunks, normal_min_support_effective, negative_filtering_enabled, negative_filter_threshold)
            filtered = [r for r in (is_rule_valid_for_filtering_batch(x) for x in rule_batch) if r is not None]
    except Exception as e:
        logger.warning(f"[Filter] Parallel filtering failed; falling back to sequential. Error: {e}")
        filtered = []
        for r in rule_batch:
            is_fallback = False
            if check_fallback and dominant_cols:
                rule_cols = set(r.keys())
                dominant_cols_set = set(dominant_cols.keys())
                if rule_cols.issubset(dominant_cols_set) and all(r.get(col) == dominant_cols.get(col) for col in rule_cols):
                    is_fallback = True
            if rule_passes_global(
                r,
                is_fallback=is_fallback,
                normal_data_empty=normal_data_empty,
                normal_data_chunks=normal_data_chunks,
                strict_normal_zero=strict_normal_zero,
                anomalous_data_in_chunk=anomalous_data_in_chunk,
                negative_filtering_enabled=negative_filtering_enabled,
                negative_filter_threshold=negative_filter_threshold,
                normal_min_support_effective=normal_min_support_effective,
            ):
                filtered.append(r)

    if check_fallback and dominant_cols and filtered:
        filtered_checked = []
        for r in filtered:
            rule_cols = set(r.keys())
            dominant_cols_set = set(dominant_cols.keys())
            is_fallback = False
            if rule_cols.issubset(dominant_cols_set) and all(r.get(col) == dominant_cols.get(col) for col in rule_cols):
                is_fallback = True
            if rule_passes_global(
                r,
                is_fallback=is_fallback,
                normal_data_empty=normal_data_empty,
                normal_data_chunks=normal_data_chunks,
                strict_normal_zero=strict_normal_zero,
                anomalous_data_in_chunk=anomalous_data_in_chunk,
                negative_filtering_enabled=negative_filtering_enabled,
                negative_filter_threshold=negative_filter_threshold,
                normal_min_support_effective=normal_min_support_effective,
            ):
                filtered_checked.append(r)
        filtered = filtered_checked

    return filtered

