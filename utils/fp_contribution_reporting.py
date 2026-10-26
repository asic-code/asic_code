import pandas as pd


def _resolve_signature_created_turn(sig_id, signature_turn_created):
    """Best-effort lookup for signature creation turn with mixed id types."""
    if sig_id in signature_turn_created:
        return signature_turn_created.get(sig_id)
    try:
        sig_id_int = int(sig_id)
        if sig_id_int in signature_turn_created:
            return signature_turn_created.get(sig_id_int)
    except Exception:
        pass
    return None


def build_fp_contribution_rows_by_creation_turn(
    *,
    turn_counter,
    phase_tag,
    alerts_df,
    turn_eval_data,
    eval_label_col,
    signature_turn_created,
):
    """
    Build per-creation-turn FP contribution rows.
    Duplicate FP alerts are intentionally counted (one row per alert event).
    """
    base_cols = [
        "turn",
        "phase",
        "signature_created_turn",
        "fp_alert_events",
        "fp_alert_share",
        "total_fp_alert_events",
    ]
    if alerts_df is None or alerts_df.empty or turn_eval_data is None or turn_eval_data.empty:
        return pd.DataFrame([{
            "turn": int(turn_counter),
            "phase": str(phase_tag),
            "signature_created_turn": "none",
            "fp_alert_events": 0,
            "fp_alert_share": 0.0,
            "total_fp_alert_events": 0,
        }], columns=base_cols)
    if "alert_index" not in alerts_df.columns or "signature_id" not in alerts_df.columns:
        return pd.DataFrame([{
            "turn": int(turn_counter),
            "phase": str(phase_tag),
            "signature_created_turn": "unavailable",
            "fp_alert_events": 0,
            "fp_alert_share": 0.0,
            "total_fp_alert_events": 0,
        }], columns=base_cols)
    if eval_label_col not in turn_eval_data.columns:
        return pd.DataFrame([{
            "turn": int(turn_counter),
            "phase": str(phase_tag),
            "signature_created_turn": "label_missing",
            "fp_alert_events": 0,
            "fp_alert_share": 0.0,
            "total_fp_alert_events": 0,
        }], columns=base_cols)

    normal_idx = set(turn_eval_data.index[turn_eval_data[eval_label_col] == 0])
    if not normal_idx:
        return pd.DataFrame([{
            "turn": int(turn_counter),
            "phase": str(phase_tag),
            "signature_created_turn": "no_normals",
            "fp_alert_events": 0,
            "fp_alert_share": 0.0,
            "total_fp_alert_events": 0,
        }], columns=base_cols)

    fp_alerts = alerts_df[alerts_df["alert_index"].isin(normal_idx)].copy()
    total_fp_events = int(len(fp_alerts))
    if total_fp_events == 0:
        return pd.DataFrame([{
            "turn": int(turn_counter),
            "phase": str(phase_tag),
            "signature_created_turn": "none",
            "fp_alert_events": 0,
            "fp_alert_share": 0.0,
            "total_fp_alert_events": 0,
        }], columns=base_cols)

    fp_alerts["signature_created_turn"] = fp_alerts["signature_id"].map(
        lambda sid: _resolve_signature_created_turn(sid, signature_turn_created)
    )
    fp_alerts["signature_created_turn"] = fp_alerts["signature_created_turn"].fillna(-1).astype(int)

    grouped = fp_alerts.groupby("signature_created_turn", dropna=False).size().reset_index(name="fp_alert_events")
    grouped["turn"] = int(turn_counter)
    grouped["phase"] = str(phase_tag)
    grouped["total_fp_alert_events"] = total_fp_events
    grouped["fp_alert_share"] = grouped["fp_alert_events"].astype(float) / float(total_fp_events)
    return grouped[base_cols].sort_values(["signature_created_turn"]).reset_index(drop=True)

