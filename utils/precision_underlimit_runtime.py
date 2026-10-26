import copy
import numpy as np


def init_precision_underlimit_state():
    return {
        "precision_recent_history": {},
        "precision_cumulative_stats": {},
        "no_alert_streak_by_sig": {},
    }


def build_cluster_based_sig_metrics(turn_eval_data, sorted_signatures, alerts_df):
    """
    Build per-signature metrics using cluster-based TP/FP only.
    No label column is used here.
    """
    sig_metrics = []
    if alerts_df is not None and not alerts_df.empty and "signature_id" in alerts_df.columns:
        for sig_id, rule in sorted_signatures:
            sig_alerts = alerts_df[alerts_df["signature_id"] == sig_id]
            if sig_alerts.empty:
                precision = 0.0
                tp = 0
                fp = 0
                no_alert = True
            else:
                matched_indices = sig_alerts["alert_index"].dropna().unique().tolist()
                matched_rows = turn_eval_data.loc[matched_indices] if matched_indices else None
                if matched_rows is None or matched_rows.empty:
                    precision = 0.0
                    tp = 0
                    fp = 0
                    no_alert = True
                else:
                    tp = int((matched_rows["cluster"] == 1).sum())
                    fp = int((matched_rows["cluster"] == 0).sum())
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    no_alert = False
            sig_metrics.append(
                {
                    "Signature_dict": rule,
                    "Precision": float(precision),
                    "signature_id": sig_id,
                    "tp": int(tp),
                    "fp": int(fp),
                    "no_alert": bool(no_alert),
                }
            )
    else:
        # No alerts -> all no_alert=True
        sig_metrics = [
            {
                "Signature_dict": rule,
                "Precision": 0.0,
                "signature_id": sig_id,
                "tp": 0,
                "fp": 0,
                "no_alert": True,
            }
            for sig_id, rule in sorted_signatures
        ]
    return sig_metrics


def select_signatures_by_precision_underlimit(
    sig_metrics,
    *,
    precision_threshold,
    signature_ea,
    temporal_enabled=True,
    temporal_mode="rolling",
    temporal_window=3,
    keep_no_alert=1,
    no_alert_max_streak=3,
    state=None,
):
    """
    Temporal precision-underlimit selector.
    Returns: (selected_metrics, filter_desc, updated_state)
    """
    st = state if isinstance(state, dict) else init_precision_underlimit_state()
    precision_recent_history = st.setdefault("precision_recent_history", {})
    precision_cumulative_stats = st.setdefault("precision_cumulative_stats", {})
    no_alert_streak_by_sig = st.setdefault("no_alert_streak_by_sig", {})

    temporal_window = max(1, int(temporal_window))
    no_alert_max_streak = max(0, int(no_alert_max_streak))
    keep_mode = int(keep_no_alert)
    if keep_mode not in (0, 1, 2):
        keep_mode = 0
    precision_thr = float(precision_threshold)

    selected_metrics = []
    dropped_by_precision = 0
    dropped_by_no_alert = 0
    no_alert_kept = 0

    for m in sig_metrics:
        sig_id = m.get("signature_id")
        cur_p = float(m.get("Precision", 0.0))
        cur_tp = int(m.get("tp", 0))
        cur_fp = int(m.get("fp", 0))
        no_alert = bool(m.get("no_alert", False))

        if sig_id not in precision_recent_history:
            precision_recent_history[sig_id] = []
        if sig_id not in precision_cumulative_stats:
            precision_cumulative_stats[sig_id] = {"tp": 0, "fp": 0}
        if sig_id not in no_alert_streak_by_sig:
            no_alert_streak_by_sig[sig_id] = 0

        if no_alert:
            no_alert_streak_by_sig[sig_id] += 1
        else:
            no_alert_streak_by_sig[sig_id] = 0

        # Update temporal evidences only when alerts exist
        if not no_alert:
            precision_recent_history[sig_id].append(cur_p)
            if len(precision_recent_history[sig_id]) > temporal_window:
                precision_recent_history[sig_id] = precision_recent_history[sig_id][-temporal_window:]
            precision_cumulative_stats[sig_id]["tp"] += cur_tp
            precision_cumulative_stats[sig_id]["fp"] += cur_fp

        fail_no_alert = (
            no_alert
            and keep_mode == 1  #and bool(keep_no_alert)
            and no_alert_max_streak > 0
            and no_alert_streak_by_sig[sig_id] >= no_alert_max_streak
        )
        force_drop_no_alert = no_alert and (keep_mode in (0, 2))

        if no_alert and keep_mode == 1 and not fail_no_alert:
            pass_precision = True
            rank_score = -1.0
            no_alert_kept += 1
        elif force_drop_no_alert:
            # keep_no_alert in {0,2}: no-alert signatures are not selected
            # by precision threshold filtering (including thr=0.0).
            pass_precision = False
            rank_score = -1.0
        else:
            if bool(temporal_enabled):
                if str(temporal_mode) == "cumulative":
                    ctp = int(precision_cumulative_stats[sig_id]["tp"])
                    cfp = int(precision_cumulative_stats[sig_id]["fp"])
                    denom = ctp + cfp
                    if denom <= 0:
                        pass_precision = True
                        rank_score = -1.0
                    else:
                        p_eval = float(ctp / denom)
                        pass_precision = p_eval >= precision_thr
                        rank_score = p_eval
                else:
                    recent = precision_recent_history[sig_id]
                    if len(recent) < temporal_window:
                        pass_precision = True
                        rank_score = float(np.mean(recent)) if recent else -1.0
                    else:
                        window_vals = recent[-temporal_window:]
                        pass_precision = not all(float(v) < precision_thr for v in window_vals)
                        rank_score = float(np.mean(window_vals))
            else:
                pass_precision = cur_p >= precision_thr
                rank_score = cur_p

        m["_rank_score"] = float(rank_score)
        if fail_no_alert or force_drop_no_alert:
            dropped_by_no_alert += 1
            continue
        if not pass_precision:
            dropped_by_precision += 1
            continue
        selected_metrics.append(m)

    if signature_ea is not None:
        selected_metrics = sorted(
            selected_metrics,
            key=lambda x: float(x.get("_rank_score", -1.0)),
            reverse=True,
        )[: int(signature_ea)]
        topn_desc = f", top {int(signature_ea)}"
    else:
        topn_desc = ", all-signatures mode"

    mode_desc = (
        f"{'temporal' if bool(temporal_enabled) else 'current'}"
        + (f"/{str(temporal_mode)}[w={int(temporal_window)}]" if bool(temporal_enabled) else "")
    )
    filter_desc = (
        f"precision>= {precision_thr} ({mode_desc}){topn_desc}, "
        f"keep_no_alert={keep_mode}, no_alert_max_streak={no_alert_max_streak}, "
        f"drop_precision={dropped_by_precision}, drop_no_alert={dropped_by_no_alert}, "
        f"no_alert_kept={no_alert_kept}"
    )

    return selected_metrics, filter_desc, st


def cleanup_precision_underlimit_state(state, active_ids):
    st = state if isinstance(state, dict) else {}
    active = set(active_ids) if active_ids is not None else set()
    for key in ("precision_recent_history", "precision_cumulative_stats", "no_alert_streak_by_sig"):
        sub = st.get(key, {})
        if isinstance(sub, dict):
            for sid in list(sub.keys()):
                if sid not in active:
                    sub.pop(sid, None)
    return st


def run_precision_underlimit_auto_adjust(
    *,
    sig_metrics,
    args,
    precision_underlimit_state,
    all_valid_signatures,
    turn_eval_data,
    calculate_performance_in_batches_fn,
    precision_thr,
    temporal_enabled,
    temporal_mode,
    temporal_window,
    keep_no_alert,
    no_alert_max_streak,
    auto_factor,
    auto_max_retries,
    auto_start_threshold,
    auto_recall_floor,
    auto_precision_floor,
    auto_min_thr,
    turn_counter=None,
    logger=None,
):
    """
    Auto-adjust only (bidirectional):
    - If recall is below floor, lower precision threshold to recover recall,
      stopping when precision falls below precision floor.
    - Else if precision is below floor, raise precision threshold to recover
      precision, stopping when recall falls below recall floor.
    """
    precision_thr = float(precision_thr)
    temporal_enabled = bool(temporal_enabled)
    temporal_mode = str(temporal_mode)
    temporal_window = max(1, int(temporal_window))
    keep_no_alert = int(keep_no_alert)
    no_alert_max_streak = max(0, int(no_alert_max_streak))
    auto_factor = float(auto_factor)
    auto_max_retries = max(0, int(auto_max_retries))
    auto_start_threshold = float(auto_start_threshold)
    auto_recall_floor = float(auto_recall_floor)
    auto_precision_floor = float(auto_precision_floor)
    auto_min_thr = max(0.0, float(auto_min_thr))

    def _evaluate_trial(trial_thr, seed_state):
        trial_metrics, trial_desc, trial_state_after = select_signatures_by_precision_underlimit(
            sig_metrics,
            precision_threshold=float(trial_thr),
            signature_ea=args.signature_ea,
            temporal_enabled=temporal_enabled,
            temporal_mode=temporal_mode,
            temporal_window=temporal_window,
            keep_no_alert=keep_no_alert,
            no_alert_max_streak=no_alert_max_streak,
            state=seed_state,
        )
        trial_ids = {m.get("signature_id") for m in trial_metrics if m.get("signature_id") is not None}
        if trial_ids:
            trial_sigs_sorted = sorted(
                [(sid, all_valid_signatures[sid]) for sid in trial_ids if sid in all_valid_signatures],
                key=lambda x: x[0],
            )
            trial_formatted = [{"id": sid, "name": f"Sig_{sid}", "rule_dict": r} for sid, r in trial_sigs_sorted]
            _, _, trial_recall_cluster, trial_precision_cluster, _, _, _ = calculate_performance_in_batches_fn(
                turn_eval_data,
                trial_formatted,
                args.evaluation_batch_size,
                label_col="cluster",  # cluster-only evaluation to avoid label leakage
                show_apply_progress=False,
                show_apply_info=False,
                show_batch_progress=False,
            )
        else:
            trial_recall_cluster = 0.0
            trial_precision_cluster = 0.0
        return {
            "thr": float(trial_thr),
            "metrics": trial_metrics,
            "desc": trial_desc,
            "state": trial_state_after,
            "recall_cluster": float(trial_recall_cluster),
            "precision_cluster": float(trial_precision_cluster),
        }

    current_thr = float(precision_thr)
    mode = None  # raise_recall | raise_precision | stable
    first_bundle = None
    best_bundle = None
    evaluated_thresholds = []
    reached_tiny_step_stop_in_raise_recall = False
    total_trials = auto_max_retries + 1
    for trial_idx in range(1, total_trials + 1):
        bundle = _evaluate_trial(current_thr, copy.deepcopy(precision_underlimit_state))
        evaluated_thresholds.append(float(current_thr))
        if first_bundle is None:
            first_bundle = bundle
            if float(bundle["recall_cluster"]) < auto_start_threshold:
                mode = "raise_recall"
            elif float(bundle["precision_cluster"]) < auto_start_threshold:
                mode = "raise_precision"
            else:
                mode = "stable"
            if logger is not None:
                logger.info(
                    f"[UnderLimitAuto] Turn {turn_counter}: mode={mode}, "
                    f"base_thr={current_thr:.6f}, base_cluster_precision={bundle['precision_cluster']:.4f}, "
                    f"base_cluster_recall={bundle['recall_cluster']:.4f}, "
                    f"start_threshold={auto_start_threshold:.4f}, "
                    f"precision_floor={auto_precision_floor:.4f}, recall_floor={auto_recall_floor:.4f}"
                )

        if logger is not None:
            logger.info(
                f"[UnderLimitAuto] Turn {turn_counter} trial {trial_idx}/{total_trials}: "
                f"thr={current_thr:.6f}, cluster_precision={bundle['precision_cluster']:.4f}, "
                f"cluster_recall={bundle['recall_cluster']:.4f}, mode={mode}"
            )

        if mode == "raise_recall":
            if float(bundle["precision_cluster"]) < auto_precision_floor:
                if logger is not None:
                    logger.info(
                        f"[UnderLimitAuto] Turn {turn_counter}: stop at thr={current_thr:.6f} "
                        f"(cluster_precision {bundle['precision_cluster']:.4f} < floor {auto_precision_floor:.4f})."
                    )
                break
            if (
                best_bundle is None
                or float(bundle["recall_cluster"]) > float(best_bundle["recall_cluster"])
                or (
                    abs(float(bundle["recall_cluster"]) - float(best_bundle["recall_cluster"])) <= 1e-12
                    and float(bundle["thr"]) < float(best_bundle["thr"])
                )
            ):
                best_bundle = bundle
            next_thr = max(auto_min_thr, current_thr * auto_factor)
            if next_thr >= current_thr - 1e-12:
                reached_tiny_step_stop_in_raise_recall = True
                break
            current_thr = float(next_thr)
            continue

        if mode == "raise_precision":
            if float(bundle["recall_cluster"]) < auto_recall_floor:
                if logger is not None:
                    logger.info(
                        f"[UnderLimitAuto] Turn {turn_counter}: stop at thr={current_thr:.6f} "
                        f"(cluster_recall {bundle['recall_cluster']:.4f} < floor {auto_recall_floor:.4f})."
                    )
                break
            if (
                best_bundle is None
                or float(bundle["precision_cluster"]) > float(best_bundle["precision_cluster"])
                or (
                    abs(float(bundle["precision_cluster"]) - float(best_bundle["precision_cluster"])) <= 1e-12
                    and float(bundle["thr"]) > float(best_bundle["thr"])
                )
            ):
                best_bundle = bundle
            next_thr = min(1.0, current_thr / auto_factor) if auto_factor > 0 else current_thr
            if next_thr <= current_thr + 1e-12:
                break
            current_thr = float(next_thr)
            continue

        # stable
        best_bundle = bundle
        break

    chosen = best_bundle if best_bundle is not None else first_bundle

    # Optional safety-net: if recall-improvement mode made no real progress,
    # force-test exact 0.0 threshold once (when min allows it).
    if (
        mode == "raise_recall"
        and reached_tiny_step_stop_in_raise_recall
        and auto_min_thr <= 0.0
        and all(abs(t) > 1e-15 for t in evaluated_thresholds)
        and first_bundle is not None
        and chosen is not None
        and float(chosen["recall_cluster"]) <= float(first_bundle["recall_cluster"]) + 1e-12
    ):
        zero_bundle = _evaluate_trial(0.0, copy.deepcopy(precision_underlimit_state))
        if logger is not None:
            logger.info(
                f"[UnderLimitAuto] Turn {turn_counter}: fallback zero-threshold trial "
                f"(thr=0.000000, p={zero_bundle['precision_cluster']:.4f}, r={zero_bundle['recall_cluster']:.4f})."
            )
        if float(zero_bundle["precision_cluster"]) >= auto_precision_floor:
            if (
                chosen is None
                or float(zero_bundle["recall_cluster"]) > float(chosen["recall_cluster"])
                or (
                    abs(float(zero_bundle["recall_cluster"]) - float(chosen["recall_cluster"])) <= 1e-12
                    and float(zero_bundle["thr"]) < float(chosen["thr"])
                )
            ):
                chosen = zero_bundle

    if chosen is None:
        return [], f"auto_adjust=1 but no valid trial (base_thr={precision_thr})", precision_underlimit_state

    if logger is not None and first_bundle is not None:
        logger.info(
            f"[UnderLimitAuto] Turn {turn_counter}: selected thr={chosen['thr']:.6f} "
            f"(base_thr={first_bundle['thr']:.6f}, mode={mode}, "
            f"base_p={first_bundle['precision_cluster']:.4f}, base_r={first_bundle['recall_cluster']:.4f}, "
            f"chosen_p={chosen['precision_cluster']:.4f}, chosen_r={chosen['recall_cluster']:.4f})."
        )

    filter_desc = (
        f"{chosen['desc']}, auto_adjust=1, "
        f"mode={mode}, "
        f"chosen_thr={chosen['thr']:.6f}, "
        f"chosen_cluster_precision={chosen['precision_cluster']:.4f}, "
        f"chosen_cluster_recall={chosen['recall_cluster']:.4f}, "
        f"precision_floor={auto_precision_floor:.4f}, recall_floor={auto_recall_floor:.4f}"
    )
    return chosen["metrics"], filter_desc, chosen["state"]
