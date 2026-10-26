import logging
import os

import pandas as pd


logger = logging.getLogger(__name__)


def validate_fp_subset_runtime(
    *,
    signatures_subset,
    turn_counter,
    phase_tag,
    rules_at_turn_start_ref,
    normal_data_raw,
    anomalous_data_raw,
    turn_eval_data,
    args,
    run_dir,
    signatures_to_remove,
    fp_event_records,
    turn_fp_events_csv,
    apply_signatures_to_dataset_fn,
    evaluate_false_positives_fn,
    summarize_fp_results_fn,
    run_fp_diagnostics_fn,
):
    """
    Validate a signature subset against normal rows for FP removal.
    Returns: (newly_removed_count, newly_flagged_ids)
    """
    newly_removed_count_local = 0
    newly_flagged_local = set()
    if args.disable_fp_removal or normal_data_raw.empty or not signatures_subset:
        return newly_removed_count_local, newly_flagged_local

    signatures_to_test = [
        {'id': sig_id, 'name': f'Sig_{sig_id}', 'rule_dict': rule}
        for sig_id, rule in signatures_subset.items()
        if sig_id not in signatures_to_remove
    ]
    if not signatures_to_test:
        return newly_removed_count_local, newly_flagged_local

    if args.use_fp_metrics:
        logger.info(
            f"[FP Metrics:{phase_tag}] Validating {len(signatures_to_test)} signatures "
            f"against {len(normal_data_raw)} normal rows..."
        )
        fp_alerts_tmp_path = os.path.join(run_dir, f"fp_alerts_{phase_tag}_turn_{turn_counter}.csv")
        if os.path.exists(fp_alerts_tmp_path):
            try:
                os.remove(fp_alerts_tmp_path)
            except OSError:
                logger.warning(f"[FP Metrics:{phase_tag}] Could not remove existing temp file: {fp_alerts_tmp_path}")
        fp_alerts_written = False
        batch_size = args.signature_batch_size
        num_batches = (len(signatures_to_test) + batch_size - 1) // batch_size
        logger.info(f"[FP Metrics:{phase_tag}] Processing {num_batches} batches (size={batch_size}).")
        for i in range(0, len(signatures_to_test), batch_size):
            batch_signatures = signatures_to_test[i:i + batch_size]
            fp_alerts_batch = apply_signatures_to_dataset_fn(normal_data_raw, batch_signatures)
            if not fp_alerts_batch.empty:
                fp_alerts_batch.to_csv(
                    fp_alerts_tmp_path,
                    mode="a",
                    index=False,
                    header=not fp_alerts_written,
                    encoding='utf-8-sig',
                )
                fp_alerts_written = True

        if fp_alerts_written:
            fp_alerts = pd.read_csv(fp_alerts_tmp_path)
            current_signatures_map = {sig['id']: sig for sig in signatures_to_test}
            try:
                signatures_to_remove_updated, fp_events = run_fp_diagnostics_fn(
                    turn_counter=turn_counter,
                    fp_alerts=fp_alerts,
                    current_signatures_map=current_signatures_map,
                    turn_eval_data=turn_eval_data,
                    normal_data_in_chunk=normal_data_raw,
                    anomalous_data_in_chunk=anomalous_data_raw,
                    rules_at_turn_start=rules_at_turn_start_ref,
                    args=args,
                    signatures_to_remove=signatures_to_remove,
                )
                signatures_to_remove.update(signatures_to_remove_updated)
                if fp_events:
                    fp_event_records.extend(fp_events)
                    fp_df = pd.DataFrame(fp_events)
                    fp_df.to_csv(
                        turn_fp_events_csv,
                        mode="a",
                        index=False,
                        header=not os.path.exists(turn_fp_events_csv),
                        encoding='utf-8-sig',
                    )
            except Exception as e:
                logger.warning(f"[FP-Diag:{phase_tag}] Turn {turn_counter}: diagnostics failed: {e}")

            detailed_fp_results = evaluate_false_positives_fn(
                alerts_df=fp_alerts,
                current_signatures_map=current_signatures_map,
                known_fp_sig_dicts=None,
                attack_free_df=normal_data_raw,
                t0_nra=args.fp_t0_nra,
                n0_nra=args.fp_n0_nra,
                lambda_haf=args.fp_lambda_haf,
                lambda_ufp=args.fp_lambda_ufp,
                combine_method=args.fp_combine_method,
                belief_threshold=args.fp_belief_threshold,
                superset_strictness=args.fp_superset_strictness,
                file_type=args.file_type
            )
            fp_summary = summarize_fp_results_fn(detailed_fp_results)
            if not fp_summary.empty:
                high_fp_signatures = fp_summary[fp_summary['final_likely_fp'] == True]
                flagged_for_removal = set(high_fp_signatures['signature_id'].unique())
                newly_flagged_local = flagged_for_removal - signatures_to_remove
                newly_removed_count_local = len(newly_flagged_local)
                if newly_removed_count_local > 0:
                    signatures_to_remove.update(newly_flagged_local)
                    logger.warning(
                        f"[FP Metrics:{phase_tag}] Removed {newly_removed_count_local} FP-like signatures "
                        f"({len(high_fp_signatures)}/{len(fp_summary)} flagged)."
                    )
                else:
                    logger.info(f"[FP Metrics:{phase_tag}] No new FP signatures detected.")
            try:
                if os.path.exists(fp_alerts_tmp_path):
                    os.remove(fp_alerts_tmp_path)
            except OSError:
                logger.warning(f"[FP Metrics:{phase_tag}] Could not remove temp file: {fp_alerts_tmp_path}")
        else:
            logger.info(f"[FP Metrics:{phase_tag}] No alerts on normal data.")
    else:
        logger.info(
            f"[FP:{phase_tag}] Validating {len(signatures_to_test)} signatures "
            f"against {len(normal_data_raw)} normal rows (simple)."
        )
        all_fp_alerts = []
        batch_size = args.signature_batch_size
        for i in range(0, len(signatures_to_test), batch_size):
            batch_signatures = signatures_to_test[i:i + batch_size]
            fp_alerts_batch = apply_signatures_to_dataset_fn(normal_data_raw, batch_signatures)
            if not fp_alerts_batch.empty:
                all_fp_alerts.append(fp_alerts_batch)
        if all_fp_alerts:
            fp_alerts = pd.concat(all_fp_alerts, ignore_index=True)
            flagged_for_removal = set(fp_alerts['signature_id'].unique())
            newly_flagged_local = flagged_for_removal - signatures_to_remove
            newly_removed_count_local = len(newly_flagged_local)
            if newly_removed_count_local > 0:
                signatures_to_remove.update(newly_flagged_local)
                logger.warning(f"[FP:{phase_tag}] Removed {newly_removed_count_local} FP signatures.")
    return newly_removed_count_local, newly_flagged_local

