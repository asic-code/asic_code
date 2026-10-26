"""
Shared logic for attack-wise performance history CSV signature bars.

Same semantics as create_attackwise_history_graph.compute_signature_bar_plot_series
(documented there in Korean).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_signature_bar_plot_series(
    df: pd.DataFrame, gen_col: str, rem_col: str
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    actual_only plot 컬럼 쌍에 대해, 가능하면 턴 경계 유입/유출(carry_in/carry_out)까지 반영한 2막대로 맞춘다.

    턴 t(행)에서 prev_exit = actual_exit_signature_count[t-1], t가 시간상 첫 행이면 prev_exit = entry_signature_count[t].
      carry_in = max(0, entry[t] - prev_exit)
      carry_out = max(0, prev_exit - entry[t])
      generated_plot_final = plot_generated + carry_in
      removed_plot_final = plot_removed + carry_out
    그러면 prev_exit + generated_plot_final - removed_plot_final = actual_exit[t].

    위에 필요한 컬럼이 없으면 actual_net_change(또는 exit−entry) + removed 로 보조한다.

    그 외 컬럼 조합은 원본(gen_col/rem_col) 값을 그대로 사용한다.
    """
    gen_raw = pd.to_numeric(df[gen_col], errors="coerce").fillna(0).to_numpy(dtype=float)
    rem_raw = pd.to_numeric(df[rem_col], errors="coerce").fillna(0).to_numpy(dtype=float)

    def _is_actual_only_pair(g: str, r: str) -> bool:
        return g.startswith("plot_generated_survived_actual_only") and r.startswith(
            "plot_removed_not_created_actual_only"
        )

    if _is_actual_only_pair(gen_col, rem_col):
        if "entry_signature_count" in df.columns and "actual_exit_signature_count" in df.columns:
            work = df.assign(_orig_pos=np.arange(len(df)))
            if "turn" in work.columns:
                work = work.sort_values("turn", kind="mergesort")
            entry_sig = pd.to_numeric(work["entry_signature_count"], errors="coerce").fillna(0).to_numpy(dtype=float)
            actual_exit = pd.to_numeric(work["actual_exit_signature_count"], errors="coerce").fillna(0).to_numpy(
                dtype=float
            )
            gen_w = pd.to_numeric(work[gen_col], errors="coerce").fillna(0).to_numpy(dtype=float)
            rem_w = pd.to_numeric(work[rem_col], errors="coerce").fillna(0).to_numpy(dtype=float)
            if "turn" in work.columns:
                turn_vals = pd.to_numeric(work["turn"], errors="coerce").fillna(0).to_numpy(dtype=int)
            else:
                turn_vals = None
            n = len(work)
            gen_f = np.zeros(n, dtype=float)
            rem_f = np.zeros(n, dtype=float)
            for i in range(n):
                if i > 0 and (turn_vals is None or turn_vals[i] == turn_vals[i - 1] + 1):
                    prev_exit = float(actual_exit[i - 1])
                else:
                    prev_exit = float(entry_sig[i])
                carry_in = max(0.0, float(entry_sig[i] - prev_exit))
                carry_out = max(0.0, float(prev_exit - entry_sig[i]))
                gen_f[i] = gen_w[i] + carry_in
                rem_f[i] = rem_w[i] + carry_out
            gen_out = np.zeros(len(df), dtype=float)
            rem_out = np.zeros(len(df), dtype=float)
            orig = work["_orig_pos"].to_numpy(dtype=int)
            for i in range(n):
                o = orig[i]
                gen_out[o] = gen_f[i]
                rem_out[o] = rem_f[i]
            return gen_out, rem_out, True

        if "actual_net_change" in df.columns:
            net_change = pd.to_numeric(df["actual_net_change"], errors="coerce").fillna(0).to_numpy(dtype=float)
        elif "entry_signature_count" in df.columns and "exit_signature_count" in df.columns:
            entry_sig = pd.to_numeric(df["entry_signature_count"], errors="coerce").fillna(0).to_numpy(dtype=float)
            exit_sig = pd.to_numeric(df["exit_signature_count"], errors="coerce").fillna(0).to_numpy(dtype=float)
            net_change = exit_sig - entry_sig
        else:
            net_change = None

        if net_change is not None:
            removed_eff = rem_raw
            generated_eff = net_change + removed_eff
            return generated_eff, removed_eff, True

    return gen_raw, rem_raw, False


def pick_signature_bar_columns(df: pd.DataFrame, keep_no_alert: int) -> tuple[str | None, str | None]:
    """create_attackwise_history_graph.py 와 동일한 우선순위로 Generated/Removed 소스 열을 고른다."""
    cols = df.columns
    if keep_no_alert == 1 and (
        "plot_generated_survived_actual_only_no_alert_excluded" in cols
        and "plot_removed_not_created_actual_only_no_alert_excluded" in cols
    ):
        return (
            "plot_generated_survived_actual_only_no_alert_excluded",
            "plot_removed_not_created_actual_only_no_alert_excluded",
        )
    if "plot_generated_survived_actual_only" in cols and "plot_removed_not_created_actual_only" in cols:
        return "plot_generated_survived_actual_only", "plot_removed_not_created_actual_only"
    if "plot_generated_survived_same_turn" in cols and "plot_removed_not_created_this_turn" in cols:
        return "plot_generated_survived_same_turn", "plot_removed_not_created_this_turn"
    if "plot_generated_excl_inactive_reduction" in cols and "plot_removed_excl_inactive_reduction" in cols:
        return "plot_generated_excl_inactive_reduction", "plot_removed_excl_inactive_reduction"
    if "generated" in cols and "removed" in cols:
        return "generated", "removed"
    return None, None
