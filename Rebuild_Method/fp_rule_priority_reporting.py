import pandas as pd


def _resolve_signature_created_turn(sig_id, signature_turn_created):
    if sig_id in signature_turn_created:
        return signature_turn_created.get(sig_id)
    try:
        sig_id_int = int(sig_id)
        if sig_id_int in signature_turn_created:
            return signature_turn_created.get(sig_id_int)
    except Exception:
        pass
    return None


def build_rule_phase_stats_by_signature(
    *,
    turn_counter,
    phase_tag,
    alerts_df,
    turn_eval_data,
    eval_label_col,
    signatures_map,
    signature_turn_created,
):
    """
    Build per-signature FP/TP stats for one turn and one phase.
    Includes an exact single-rule recall drop estimate:
      recall_drop_if_removed = exclusive_tp_rows / total_anomalies
    where exclusive_tp_rows are anomaly rows alerted only by that signature.
    """
    base_cols = [
        "turn",
        "phase",
        "signature_id",
        "signature_created_turn",
        "rule_content",
        "total_alert_events",
        "fp_alert_events",
        "tp_alert_events",
        "unique_normal_rows",
        "normal_coverage",
        "unique_anomaly_rows",
        "exclusive_tp_rows",
        "recall_drop_if_removed",
        "total_normals",
        "total_anomalies",
    ]
    if alerts_df is None or alerts_df.empty or turn_eval_data is None or turn_eval_data.empty:
        return pd.DataFrame(columns=base_cols)
    if "alert_index" not in alerts_df.columns or "signature_id" not in alerts_df.columns:
        return pd.DataFrame(columns=base_cols)
    if eval_label_col not in turn_eval_data.columns:
        return pd.DataFrame(columns=base_cols)

    total_normals = int((turn_eval_data[eval_label_col] == 0).sum())
    total_anomalies = int((turn_eval_data[eval_label_col] == 1).sum())
    normal_idx = set(turn_eval_data.index[turn_eval_data[eval_label_col] == 0])
    anomaly_idx = set(turn_eval_data.index[turn_eval_data[eval_label_col] == 1])

    work = alerts_df[["signature_id", "alert_index"]].copy()
    work["is_normal"] = work["alert_index"].isin(normal_idx)
    work["is_anomaly"] = work["alert_index"].isin(anomaly_idx)

    # Per-signature event and unique-row counts
    grp = work.groupby("signature_id", dropna=False)
    stats = grp.agg(
        total_alert_events=("alert_index", "size"),
        fp_alert_events=("is_normal", "sum"),
        tp_alert_events=("is_anomaly", "sum"),
    ).reset_index()

    unique_norm = (
        work[work["is_normal"]]
        .groupby("signature_id", dropna=False)["alert_index"]
        .nunique()
        .rename("unique_normal_rows")
        .reset_index()
    )
    unique_anom = (
        work[work["is_anomaly"]]
        .groupby("signature_id", dropna=False)["alert_index"]
        .nunique()
        .rename("unique_anomaly_rows")
        .reset_index()
    )
    stats = stats.merge(unique_norm, on="signature_id", how="left")
    stats = stats.merge(unique_anom, on="signature_id", how="left")
    stats["unique_normal_rows"] = stats["unique_normal_rows"].fillna(0).astype(int)
    stats["unique_anomaly_rows"] = stats["unique_anomaly_rows"].fillna(0).astype(int)

    # Exclusive anomaly coverage per signature (single-rule removal recall impact)
    if total_anomalies > 0:
        anom_alerts = work[work["is_anomaly"]][["signature_id", "alert_index"]].drop_duplicates()
        if not anom_alerts.empty:
            row_counts = anom_alerts.groupby("alert_index")["signature_id"].nunique().rename("sig_cnt").reset_index()
            anom_alerts = anom_alerts.merge(row_counts, on="alert_index", how="left")
            exclusive = (
                anom_alerts[anom_alerts["sig_cnt"] == 1]
                .groupby("signature_id", dropna=False)["alert_index"]
                .nunique()
                .rename("exclusive_tp_rows")
                .reset_index()
            )
        else:
            exclusive = pd.DataFrame(columns=["signature_id", "exclusive_tp_rows"])
    else:
        exclusive = pd.DataFrame(columns=["signature_id", "exclusive_tp_rows"])
    stats = stats.merge(exclusive, on="signature_id", how="left")
    stats["exclusive_tp_rows"] = stats["exclusive_tp_rows"].fillna(0).astype(int)

    stats["turn"] = int(turn_counter)
    stats["phase"] = str(phase_tag)
    stats["signature_created_turn"] = stats["signature_id"].map(
        lambda sid: _resolve_signature_created_turn(sid, signature_turn_created)
    )
    stats["signature_created_turn"] = pd.to_numeric(stats["signature_created_turn"], errors="coerce").fillna(-1).astype(int)
    stats["rule_content"] = stats["signature_id"].map(
        lambda sid: str(signatures_map.get(sid, signatures_map.get(int(sid), {}))) if signatures_map is not None else "{}"
    )
    stats["total_normals"] = total_normals
    stats["total_anomalies"] = total_anomalies
    stats["normal_coverage"] = (
        stats["unique_normal_rows"].astype(float) / float(total_normals)
        if total_normals > 0 else 0.0
    )
    stats["recall_drop_if_removed"] = (
        stats["exclusive_tp_rows"].astype(float) / float(total_anomalies)
        if total_anomalies > 0 else 0.0
    )
    return stats[base_cols].sort_values(["fp_alert_events", "signature_id"], ascending=[False, True]).reset_index(drop=True)

