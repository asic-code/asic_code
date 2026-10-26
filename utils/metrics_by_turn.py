from collections import defaultdict
from bisect import bisect_left, bisect_right


def calculate_metrics_from_alerts(alerts_df, eval_df, label_col):
    """
    Recompute metrics from precomputed alerts without rerunning matcher.
    """
    if eval_df is None or eval_df.empty or label_col not in eval_df.columns:
        return 0.0, 0.0, 0.0, 0.0

    alerted_indices = set()
    if alerts_df is not None and not alerts_df.empty and "alert_index" in alerts_df.columns:
        alerted_indices = set(alerts_df["alert_index"].dropna().tolist())

    pos_idx = set(eval_df.index[eval_df[label_col] == 1])
    neg_idx = set(eval_df.index[eval_df[label_col] == 0])
    total_rows = len(eval_df)

    tp = len(alerted_indices.intersection(pos_idx))
    fp = len(alerted_indices.intersection(neg_idx))
    tn = len(neg_idx) - fp

    total_pos = len(pos_idx)
    total_alerts = len(alerted_indices)

    recall = tp / total_pos if total_pos > 0 else 0.0
    precision = tp / total_alerts if total_alerts > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total_rows if total_rows > 0 else 0.0
    return recall, precision, f1, accuracy


def compute_metrics_by_turn_range(chunk_cache_obj, sig_items, created_map, removed_map, eval_label_col, calculate_signature):
    if chunk_cache_obj is None or chunk_cache_obj.is_empty() or not sig_items:
        return {}
    turns = chunk_cache_obj.get_turns()
    if not turns:
        return {}
    first_turn = turns[0]
    last_turn = turns[-1]
    sig_rules = {sid: rule for sid, rule in sig_items}
    created_map = created_map or {}
    removed_map = removed_map or {}

    start_map = defaultdict(list)
    end_map = defaultdict(list)
    for sid, _ in sig_items:
        start = created_map.get(sid, first_turn)
        end = removed_map.get(sid, last_turn)
        # Align to existing turns
        si = bisect_left(turns, start)
        ei = bisect_right(turns, end) - 1
        if si >= len(turns) or ei < 0:
            continue
        start_aligned = turns[si]
        end_aligned = turns[ei]
        if end_aligned < start_aligned:
            continue
        start_map[start_aligned].append(sid)
        end_map[end_aligned].append(sid)

    metrics = {sid: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for sid, _ in sig_items}
    active_ids = set()
    for turn, df in chunk_cache_obj.iter_chunks_with_turn():
        if turn in start_map:
            active_ids.update(start_map[turn])
        if not active_ids:
            if turn in end_map:
                for sid in end_map[turn]:
                    active_ids.discard(sid)
            continue

        eval_df = df
        if eval_label_col == "cluster" and "cluster" in eval_df.columns:
            eval_df = eval_df.copy()
            eval_df["label"] = eval_df["cluster"]

        active_list = sorted(active_ids)
        sig_list = [sig_rules[sid] for sid in active_list]
        sig_metrics_results = calculate_signature(eval_df, sig_list)
        for idx, sid in enumerate(active_list):
            if idx >= len(sig_metrics_results):
                break
            result = sig_metrics_results[idx]
            metrics[sid]['tp'] += result.get('TP', 0)
            metrics[sid]['fp'] += result.get('FP', 0)
            metrics[sid]['tn'] += result.get('TN', 0)
            metrics[sid]['fn'] += result.get('FN', 0)

        if turn in end_map:
            for sid in end_map[turn]:
                active_ids.discard(sid)

    for sid, m in metrics.items():
        tp = m['tp']
        fp = m['fp']
        fn = m['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        m['precision'] = precision
        m['recall'] = recall
        m['f1_score'] = f1_score
    return metrics
