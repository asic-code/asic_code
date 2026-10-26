import logging

import pandas as pd

from Rebuild_Method.FalsePositive_Check import apply_signatures_to_dataset

logger = logging.getLogger(__name__)


def run_fp_diagnostics(
    *,
    turn_counter,
    fp_alerts,
    current_signatures_map,
    turn_eval_data,
    normal_data_in_chunk,
    anomalous_data_in_chunk,
    rules_at_turn_start,
    args,
    signatures_to_remove,
):
    """
    Run FP diagnostics and optional FP-heavy rule management.

    Returns the updated signatures_to_remove set.
    """
    fp_events = []
    if fp_alerts is None or fp_alerts.empty:
        return signatures_to_remove, fp_events

    signatures_to_test = list(current_signatures_map.values())

    # 1) Cluster distribution for FP alerts (based on alert indices)
    if "alert_index" in fp_alerts.columns and turn_eval_data is not None and "cluster" in turn_eval_data.columns:
        fp_indices = fp_alerts["alert_index"].dropna().unique().tolist()
        if fp_indices:
            fp_clusters = turn_eval_data.loc[fp_indices, "cluster"]
            cluster_ratio = fp_clusters.value_counts(normalize=True).to_dict()
            logger.info(f"[FP-Diag] Turn {turn_counter}: FP cluster ratio={cluster_ratio}")

    # 2) Top FP-causing rules
    fp_counts = fp_alerts["signature_id"].value_counts().head(5)
    if not fp_counts.empty:
        logger.info(f"[FP-Diag] Turn {turn_counter}: Top FP rules (id -> fp_count)={fp_counts.to_dict()}")

        # Precompute FP counts for all signatures from normal alerts
        fp_counts_all = fp_alerts["signature_id"].value_counts().to_dict()

        # Precompute TP alert indices for all signatures on anomaly data
        tp_indices_by_sig = {}
        if anomalous_data_in_chunk is not None and not anomalous_data_in_chunk.empty:
            tp_alerts_all = apply_signatures_to_dataset(
                anomalous_data_in_chunk,
                signatures_to_test,
                show_progress=False,
                log_info=False,
            )
            if not tp_alerts_all.empty:
                grouped = tp_alerts_all.groupby("signature_id")["alert_index"]
                for sid, idxs in grouped:
                    tp_indices_by_sig[sid] = set(idxs.dropna().unique().tolist())

        # Per-rule precision (cluster & label) + reduction/inactive hints
        for sig_id, fp_cnt in fp_counts.items():
            sig_entry = current_signatures_map.get(sig_id)
            if not sig_entry:
                continue
            rule = sig_entry["rule_dict"]

            # Cluster-based precision: TP on anomalous, FP on normal
            tp_cnt = 0
            if anomalous_data_in_chunk is not None and not anomalous_data_in_chunk.empty:
                tp_alerts = apply_signatures_to_dataset(
                    anomalous_data_in_chunk,
                    [sig_entry],
                    show_progress=False,
                    log_info=False,
                )
                tp_cnt = int(tp_alerts["alert_index"].nunique()) if not tp_alerts.empty else 0
            fp_cnt = int(fp_cnt)
            cluster_prec = tp_cnt / (tp_cnt + fp_cnt) if (tp_cnt + fp_cnt) > 0 else 0.0

            # Label-based precision (if label available in turn_eval_data)
            label_prec = None
            if turn_eval_data is not None and "label" in turn_eval_data.columns:
                label_alerts = apply_signatures_to_dataset(
                    turn_eval_data,
                    [sig_entry],
                    show_progress=False,
                    log_info=False,
                )
                if not label_alerts.empty:
                    idxs = label_alerts["alert_index"].dropna().unique().tolist()
                    if idxs:
                        lbl = turn_eval_data.loc[idxs, "label"]
                        tp_label = int((lbl == 1).sum())
                        fp_label = int((lbl == 0).sum())
                        label_prec = tp_label / (tp_label + fp_label) if (tp_label + fp_label) > 0 else 0.0

            # Inactive hint: no alerts in turn_eval_data
            inactive_hint = False
            if turn_eval_data is not None:
                alerts_any = apply_signatures_to_dataset(
                    turn_eval_data,
                    [sig_entry],
                    show_progress=False,
                    log_info=False,
                )
                inactive_hint = alerts_any.empty

            # Reduction hint: if rule is a superset of another rule in current set
            reduction_hint = False
            subset_rule_id = None
            try:
                for other_id, other_rule in rules_at_turn_start.items():
                    if other_id == sig_id or not isinstance(other_rule, dict):
                        continue
                    if all(k in rule and rule[k] == v for k, v in other_rule.items()):
                        reduction_hint = True
                        subset_rule_id = other_id
                        break
            except Exception:
                pass

            logger.info(
                f"[FP-Diag] Turn {turn_counter}: sig_id={sig_id}, fp={fp_cnt}, tp={tp_cnt}, "
                f"cluster_prec={cluster_prec:.4f}, "
                f"label_prec={label_prec if label_prec is not None else 'NA'}, "
                f"inactive_hint={inactive_hint}, reduction_hint={reduction_hint}, rule={rule}"
            )

            # Optional: targeted reduction for FP-heavy rules that are supersets (run first)
            try:
                if getattr(args, "fp_reduce_supersets", False) and reduction_hint and subset_rule_id is not None:
                    signatures_to_remove.add(sig_id)
                    fp_events.append(
                        {
                            "turn": int(turn_counter) if turn_counter is not None else None,
                            "sig_id": sig_id,
                            "action": "reduce_superset",
                            "fp_count": int(fp_cnt),
                            "tp_count": int(tp_cnt),
                            "coverage_ratio": None,
                            "subset_rule_id": subset_rule_id,
                        }
                    )
                    logger.warning(
                        f"[FP-Reduce] Turn {turn_counter}: removed superset sig_id={sig_id} "
                        f"(subset={subset_rule_id})."
                    )
                    # Skip replacement search if already removed as superset
                    continue
            except Exception as e:
                logger.warning(f"[FP-Reduce] Turn {turn_counter}: reduction check failed: {e}")

            try:
                tp_indices = tp_indices_by_sig.get(sig_id, set())
                if tp_indices:
                    replacements = []
                    for cand_id, cand_tp_idx in tp_indices_by_sig.items():
                        if cand_id == sig_id:
                            continue
                        overlap = len(tp_indices.intersection(cand_tp_idx))
                        if overlap == 0:
                            continue
                        cand_fp_cnt = int(fp_counts_all.get(cand_id, 0))
                        if cand_fp_cnt < fp_cnt:
                            replacements.append((cand_id, cand_fp_cnt, overlap))
                    if replacements:
                        replacements.sort(key=lambda x: (x[1], -x[2]))
                        logger.info(
                            f"[FP-Diag Replace] Turn {turn_counter}: sig_id={sig_id} "
                            f"(id, fp_cnt, overlap_tp) top5={replacements[:5]}"
                        )

                        # Collective coverage: can lower-FP rules cover this rule's TP?
                        top_candidates = replacements[:10]
                        covered = set()
                        for cand_id, _cand_fp, _overlap in top_candidates:
                            covered.update(tp_indices_by_sig.get(cand_id, set()))
                        coverage_ratio = (len(covered.intersection(tp_indices)) / len(tp_indices)) if tp_indices else 0.0
                        logger.info(
                            f"[FP-Diag Collective] Turn {turn_counter}: sig_id={sig_id} "
                            f"coverage(top10)={coverage_ratio:.2%}"
                        )

                        # Optional: replace high-FP rules if TP coverage is high enough
                        if getattr(args, "fp_replace_by_coverage", False) and coverage_ratio >= args.fp_replace_tp_coverage_threshold:
                            signatures_to_remove.add(sig_id)
                            fp_events.append(
                                {
                                    "turn": int(turn_counter) if turn_counter is not None else None,
                                    "sig_id": sig_id,
                                    "action": "replace_by_coverage",
                                    "fp_count": int(fp_cnt),
                                    "tp_count": int(tp_cnt),
                                    "coverage_ratio": float(coverage_ratio),
                                    "subset_rule_id": None,
                                }
                            )
                            logger.warning(
                                f"[FP-Replace] Turn {turn_counter}: removed sig_id={sig_id} "
                                f"(fp={fp_cnt}, coverage={coverage_ratio:.2%})."
                            )
            except Exception as e:
                logger.warning(f"[FP-Diag] Turn {turn_counter}: replacement search failed: {e}")

        # 3) Low-FP rules that still alert true attacks (sample scan)
        if turn_eval_data is not None and "cluster" in turn_eval_data.columns:
            sample_rules = list(current_signatures_map.values())[:200]
            low_fp_candidates = []
            for sig_entry in sample_rules:
                rule = sig_entry["rule_dict"]
                fp_cnt = 0
                tp_cnt = 0
                if normal_data_in_chunk is not None and not normal_data_in_chunk.empty:
                    fp_alerts_s = apply_signatures_to_dataset(
                        normal_data_in_chunk,
                        [sig_entry],
                        show_progress=False,
                        log_info=False,
                    )
                    fp_cnt = int(fp_alerts_s["alert_index"].nunique()) if not fp_alerts_s.empty else 0
                if anomalous_data_in_chunk is not None and not anomalous_data_in_chunk.empty:
                    tp_alerts_s = apply_signatures_to_dataset(
                        anomalous_data_in_chunk,
                        [sig_entry],
                        show_progress=False,
                        log_info=False,
                    )
                    tp_cnt = int(tp_alerts_s["alert_index"].nunique()) if not tp_alerts_s.empty else 0
                if tp_cnt > 0 and fp_cnt <= 1:
                    low_fp_candidates.append((sig_entry["id"], tp_cnt, fp_cnt, rule))
            if low_fp_candidates:
                low_fp_candidates.sort(key=lambda x: x[1], reverse=True)
                logger.info(f"[FP-Diag] Turn {turn_counter}: Low-FP rules (top 5 by TP): {low_fp_candidates[:5]}")

    return signatures_to_remove, fp_events
