"""
Utility functions for saving detailed signature reports with metrics, attack types, and reverse mapping.
for Main_Association_Rule_ex_Batch.py
"""

import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def get_attack_columns_for_report(file_type):
    """Return attack/label columns used ONLY for reporting attack types."""
    if file_type in ['CICIDS2017', 'CICIDS']:
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


def extract_attack_types_for_signature(sig, eval_df, attack_cols, file_type):
    """Extract attack types for a signature by matching against evaluation data."""
    attacks = set()
    if not attack_cols or eval_df is None or eval_df.empty:
        attacks.add('-')
        return attacks
    
    rule_keys = [k for k in sig.keys() if k in eval_df.columns]
    if not rule_keys:
        attacks.add('-')
        return attacks
    
    match_mask = eval_df[rule_keys].eq(pd.Series({k: sig[k] for k in rule_keys})).all(axis=1)
    if not match_mask.any():
        attacks.add('-')
        return attacks
    
    matched = eval_df[match_mask]
    
    # For binary attack columns (MiraiBotnet), check if value is 1
    if file_type in ['MiraiBotnet']:
        for col in attack_cols:
            if col in matched.columns:
                if (matched[col] == 1).any():
                    attacks.add(col)
    else:
        # For categorical attack columns, collect unique values
        for col in attack_cols:
            if col in matched.columns:
                vals = matched[col].astype(str).unique().tolist()
                attacks.update(vals)
    
    if not attacks:
        attacks.add('-')
    return attacks


def _build_condition_mask(series, cond):
    """Build mask for scalar or interval condition tuple."""
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
    return series.astype(str) == str(cond)


def extract_attack_types_for_signature_from_cache(sig, chunk_cache_obj, attack_cols, batch_size):
    """
    Stream cached chunks and extract attack labels for one signature.
    Supports interval tuple conditions: (left, right, left_inclusive, right_inclusive).
    """
    attacks = set()
    if not attack_cols:
        attacks.add('-')
        return attacks

    for batch in chunk_cache_obj.iter_batches(batch_size=batch_size):
        if batch is None or batch.empty:
            continue
        rule_keys = [k for k in sig.keys() if k in batch.columns]
        if not rule_keys:
            continue

        mask = pd.Series(True, index=batch.index)
        for k in rule_keys:
            mask &= _build_condition_mask(batch[k], sig[k])
            if not mask.any():
                break
        if not mask.any():
            continue
        matched = batch.loc[mask]

        # Mirai one-hot style attack columns.
        if attack_cols == ['reconnaissance', 'infection', 'action']:
            for col in attack_cols:
                if col in matched.columns and (matched[col] == 1).any():
                    attacks.add(col)
            continue

        for col in attack_cols:
            if col in matched.columns:
                vals = matched[col].astype(str).unique().tolist()
                attacks.update(vals)

    if not attacks:
        attacks.add('-')
    return attacks


def save_detailed_signatures(signature_sets, eval_df, category_mapping, file_type, file_number, 
                             association_method, association_metric, signature_ea, batch_size=20000):
    """
    Save detailed signature information including metrics, attack types, and reverse-mapped versions.
    Saves to the same directory as csv_association: ../Dataset_Paral/signature/{file_type}/
    
    Args:
        signature_sets: List of signature dictionaries in format [{'signature_name': {'Signature_dict': {...}, ...}, ...}]
        eval_df: DataFrame for evaluation (must contain 'label' column)
        category_mapping: Category mapping from heterogeneous feature encoding
        file_type: Dataset file type
        file_number: Dataset file number
        association_method: Association rule method name
        association_metric: Association metric name
        signature_ea: Signature EA parameter
        batch_size: Batch size for processing (default: 20000)
    """
    from Heterogeneous_Method.reverse_mapping import build_reverse_mapping, reverse_map_rule
    from Evaluation.calculate_signature import calculate_signature
    
    if not signature_sets:
        logger.warning("No signatures to save detailed information for.")
        return
    
    logger.info("--- Saving detailed signature information ---")
    
    # Extract rules from signature_sets format
    # signature_sets format: [{'signature_name': {'Signature_dict': {...}, ...}, ...}]
    signatures = []
    for sig_set in signature_sets:
        if 'signature_name' in sig_set and 'Signature_dict' in sig_set['signature_name']:
            signatures.append(sig_set['signature_name']['Signature_dict'])
    
    if not signatures:
        logger.warning("No valid signatures found in signature_sets.")
        return
    
    total_sigs = len(signatures)
    logger.info(f"Processing {total_sigs} signatures for detailed report...")
    
    # Build reverse mapping
    logger.info("Building reverse mapping for signature report...")
    reverse_mapping = build_reverse_mapping(category_mapping)
    logger.info(f"Reverse mapping built for {len(reverse_mapping)} features.")
    
    # Get attack columns
    attack_cols = get_attack_columns_for_report(file_type)
    
    # Calculate individual signature performance metrics
    logger.info(f"Calculating individual performance metrics for {total_sigs} signatures...")
    individual_metrics = {}
    if eval_df is not None and not eval_df.empty and signatures:
        sig_metrics_results = calculate_signature(eval_df, signatures)
        
        for idx, sig in enumerate(signatures):
            if idx < len(sig_metrics_results):
                result = sig_metrics_results[idx]
                tp = result.get('TP', 0)
                fp = result.get('FP', 0)
                tn = result.get('TN', 0)
                fn = result.get('FN', 0)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                individual_metrics[idx] = {
                    'tp': tp,
                    'fp': fp,
                    'tn': tn,
                    'fn': fn,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score
                }
        
        logger.info(f"Completed individual performance calculation for {len(individual_metrics)} signatures.")
    else:
        logger.warning("No evaluation data available for individual signature metrics.")
    
    # Extract attack types and build records
    logger.info(f"Extracting attack types for {total_sigs} signatures...")
    signature_records = []
    signature_records_reversed = []
    
    for sig_idx, sig in enumerate(signatures):
        if sig_idx % 100 == 0 or sig_idx == 1 or sig_idx == total_sigs:
            logger.info(f"  Processing signature {sig_idx}/{total_sigs}...")
        
        attacks = extract_attack_types_for_signature(sig, eval_df, attack_cols, file_type)
        metrics = individual_metrics.get(sig_idx, {})
        
        # Original report (with group numbers)
        signature_records.append({
            'signature_rule': str(sig),
            'created_turn': None,  # Non-incremental, so no turn info
            'attack_types': "|".join(sorted(attacks)),
            'tp': metrics.get('tp', 0),
            'fp': metrics.get('fp', 0),
            'tn': metrics.get('tn', 0),
            'fn': metrics.get('fn', 0),
            'precision': metrics.get('precision', 0.0),
            'recall': metrics.get('recall', 0.0),
            'f1_score': metrics.get('f1_score', 0.0)
        })
        
        # Reverse-mapped report (with original intervals/values)
        try:
            reversed_rule = reverse_map_rule(sig, reverse_mapping)
        except Exception as e:
            logger.warning(f"Failed to reverse map signature {sig_idx}: {e}. Using original rule.")
            reversed_rule = sig
        
        signature_records_reversed.append({
            'signature_rule': str(reversed_rule),
            'created_turn': None,
            'attack_types': "|".join(sorted(attacks)),
            'tp': metrics.get('tp', 0),
            'fp': metrics.get('fp', 0),
            'tn': metrics.get('tn', 0),
            'fn': metrics.get('fn', 0),
            'precision': metrics.get('precision', 0.0),
            'recall': metrics.get('recall', 0.0),
            'f1_score': metrics.get('f1_score', 0.0)
        })
    
    logger.info(f"Finished extracting attack types and metrics for all {total_sigs} signatures.")
    
    # Save to CSV files (same directory as csv_association)
    if os.name == 'nt':
        output_dir = f"../Dataset_Paral/signature/{file_type}/"
    else:
        output_dir = f"../Dataset_Paral/signature/{file_type}/"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save original report (with group numbers)
    base_filename = f"{file_type}_{file_number}_{association_method}_{association_metric}_{signature_ea}"
    detailed_filename = f"{base_filename}_detailed_signatures.csv"
    detailed_path = os.path.join(output_dir, detailed_filename)
    
    detailed_df = pd.DataFrame(signature_records)
    detailed_df.to_csv(detailed_path, index=False, encoding='utf-8-sig')
    logger.info(f"Detailed signatures (group numbers) saved to: {detailed_path}")
    
    # Save reverse-mapped report (with original intervals/values)
    reversed_filename = f"{base_filename}_detailed_signatures_reversed.csv"
    reversed_path = os.path.join(output_dir, reversed_filename)
    
    reversed_df = pd.DataFrame(signature_records_reversed)
    reversed_df.to_csv(reversed_path, index=False, encoding='utf-8-sig')
    logger.info(f"Detailed signatures (original intervals/values) saved to: {reversed_path}")
    
    logger.info(f"--- Detailed signature information saved to: {output_dir} ---")
