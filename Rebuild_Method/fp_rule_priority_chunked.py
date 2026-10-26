import pandas as pd


def _to_int_safe(x, default=0):
    try:
        return int(x)
    except Exception:
        return int(default)


def _to_float_safe(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def generate_fp_priority_reports_from_csv(
    *,
    fp_rule_phase_csv_path,
    fp_top_cohort_rule_details_csv,
    fp_rule_delete_priority_csv,
    fp_rule_delete_turn_impact_csv,
    cohort_top_n=5,
    delete_top_k=50,
    chunksize=200000,
):
    """
    Build FP-priority reports from turn_fp_rule_phase_stats.csv using chunked reads.
    Semantics are aligned with the in-memory aggregation:
      - analysis target is phase == 'exit'
      - high FP first, lower recall-drop first for delete priority
    """
    if not fp_rule_phase_csv_path:
        return {"ok": False, "reason": "missing_path"}

    # 1) Aggregate exit-phase rows by rule and by creation turn
    cohort_fp_sum = {}
    rule_agg = {}

    usecols = [
        "turn",
        "phase",
        "signature_id",
        "signature_created_turn",
        "rule_content",
        "fp_alert_events",
        "tp_alert_events",
        "normal_coverage",
        "recall_drop_if_removed",
        "total_anomalies",
        "exclusive_tp_rows",
    ]

    for chunk in pd.read_csv(fp_rule_phase_csv_path, chunksize=chunksize, usecols=usecols):
        exit_chunk = chunk[chunk["phase"] == "exit"]
        if exit_chunk.empty:
            continue

        # Cohort FP sums
        cohort_grp = exit_chunk.groupby("signature_created_turn", dropna=False)["fp_alert_events"].sum()
        for created_turn, fp_sum in cohort_grp.items():
            c_key = _to_int_safe(created_turn, default=-1)
            cohort_fp_sum[c_key] = cohort_fp_sum.get(c_key, 0.0) + _to_float_safe(fp_sum, 0.0)

        # Rule-level aggregation
        for _, row in exit_chunk.iterrows():
            sig_raw = row.get("signature_id")
            sig_key = str(sig_raw)
            created_turn = _to_int_safe(row.get("signature_created_turn"), default=-1)
            key = (sig_key, created_turn)
            rec = rule_agg.get(key)
            if rec is None:
                rec = {
                    "signature_id": sig_raw,
                    "signature_created_turn": created_turn,
                    "rule_content": row.get("rule_content", ""),
                    "fp_alert_events_total": 0.0,
                    "tp_alert_events_total": 0.0,
                    "normal_coverage_sum": 0.0,
                    "normal_coverage_cnt": 0,
                    "last_active_turn": -1,
                    "recall_drop_sum": 0.0,
                    "recall_drop_cnt": 0,
                    "max_recall_drop_if_removed": 0.0,
                }
                rule_agg[key] = rec

            fp_ev = _to_float_safe(row.get("fp_alert_events"), 0.0)
            tp_ev = _to_float_safe(row.get("tp_alert_events"), 0.0)
            nc = _to_float_safe(row.get("normal_coverage"), 0.0)
            rd = _to_float_safe(row.get("recall_drop_if_removed"), 0.0)
            turn_v = _to_int_safe(row.get("turn"), default=-1)

            rec["fp_alert_events_total"] += fp_ev
            rec["tp_alert_events_total"] += tp_ev
            rec["normal_coverage_sum"] += nc
            rec["normal_coverage_cnt"] += 1
            rec["last_active_turn"] = max(rec["last_active_turn"], turn_v)
            rec["recall_drop_sum"] += rd
            rec["recall_drop_cnt"] += 1
            rec["max_recall_drop_if_removed"] = max(rec["max_recall_drop_if_removed"], rd)
            if (not rec.get("rule_content")) and row.get("rule_content"):
                rec["rule_content"] = row.get("rule_content")

    if not rule_agg:
        return {"ok": False, "reason": "no_exit_rows"}

    # 2) Determine top FP-heavy creation turns
    cohort_df = (
        pd.DataFrame(
            [{"signature_created_turn": k, "fp_alert_events": v} for k, v in cohort_fp_sum.items()]
        )
        .sort_values("fp_alert_events", ascending=False)
        .reset_index(drop=True)
    )
    top_creation_turns = set(
        cohort_df.head(max(1, int(cohort_top_n)))["signature_created_turn"].tolist()
    )

    # 3) Build rule priority frame
    rows = []
    for rec in rule_agg.values():
        cnt_nc = max(1, int(rec["normal_coverage_cnt"]))
        cnt_rd = max(1, int(rec["recall_drop_cnt"]))
        rows.append(
            {
                "signature_id": rec["signature_id"],
                "signature_created_turn": rec["signature_created_turn"],
                "rule_content": rec["rule_content"],
                "fp_alert_events_total": int(rec["fp_alert_events_total"]),
                "tp_alert_events_total": int(rec["tp_alert_events_total"]),
                "normal_coverage_mean": rec["normal_coverage_sum"] / float(cnt_nc),
                "last_active_turn": int(rec["last_active_turn"]),
                "avg_recall_drop_if_removed": rec["recall_drop_sum"] / float(cnt_rd),
                "max_recall_drop_if_removed": float(rec["max_recall_drop_if_removed"]),
            }
        )
    rule_priority = pd.DataFrame(rows).sort_values(
        ["fp_alert_events_total", "avg_recall_drop_if_removed", "max_recall_drop_if_removed"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    rule_priority["delete_priority_rank"] = rule_priority.index + 1
    rule_priority.to_csv(fp_rule_delete_priority_csv, index=False, encoding="utf-8-sig")

    # 4) Top-cohort details
    top_cohort_detail = rule_priority[
        rule_priority["signature_created_turn"].isin(top_creation_turns)
    ][
        [
            "signature_id",
            "signature_created_turn",
            "rule_content",
            "fp_alert_events_total",
            "tp_alert_events_total",
            "normal_coverage_mean",
            "last_active_turn",
            "avg_recall_drop_if_removed",
            "max_recall_drop_if_removed",
        ]
    ].rename(
        columns={
            "fp_alert_events_total": "fp_alert_events",
            "tp_alert_events_total": "tp_alert_events",
        }
    ).sort_values(["signature_created_turn", "fp_alert_events"], ascending=[True, False])
    top_cohort_detail.to_csv(fp_top_cohort_rule_details_csv, index=False, encoding="utf-8-sig")

    # 5) Per-turn impact for top-K delete candidates (second chunked pass)
    top_k = max(1, int(delete_top_k))
    top_sig_ids = set(str(v) for v in rule_priority.head(top_k)["signature_id"].tolist())
    impact_rows = []
    impact_cols = [
        "turn",
        "signature_id",
        "signature_created_turn",
        "fp_alert_events",
        "normal_coverage",
        "exclusive_tp_rows",
        "recall_drop_if_removed",
        "total_anomalies",
    ]
    for chunk in pd.read_csv(fp_rule_phase_csv_path, chunksize=chunksize, usecols=["phase"] + impact_cols):
        exit_chunk = chunk[chunk["phase"] == "exit"]
        if exit_chunk.empty:
            continue
        sel = exit_chunk[exit_chunk["signature_id"].astype(str).isin(top_sig_ids)]
        if not sel.empty:
            impact_rows.append(sel[impact_cols].copy())

    if impact_rows:
        turn_impact = pd.concat(impact_rows, ignore_index=True).sort_values(["signature_id", "turn"])
    else:
        turn_impact = pd.DataFrame(columns=impact_cols)
    turn_impact.to_csv(fp_rule_delete_turn_impact_csv, index=False, encoding="utf-8-sig")

    return {
        "ok": True,
        "top_creation_turns": sorted(list(top_creation_turns)),
        "rule_priority_count": int(len(rule_priority)),
        "turn_impact_rows": int(len(turn_impact)),
    }

