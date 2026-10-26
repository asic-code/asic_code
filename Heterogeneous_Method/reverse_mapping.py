"""
Reverse mapping utilities for converting group numbers back to original interval/categorical/binary values.
"""

import logging
import pandas as pd
import re

_log = logging.getLogger(__name__)


def build_reverse_mapping(category_mapping):
    """
    Build reverse mapping dictionaries from group numbers to interval/categorical/binary values.
    Returns a dictionary: {column_name: {group_number: original_value_string}}
    
    Args:
        category_mapping: Dictionary containing 'interval', 'categorical', and/or 'binary' mappings
        
    Returns:
        Dictionary mapping column names to {group_number: original_value_string} dictionaries
    """
    reverse_mapping = {}
    
    # Process interval mappings
    if 'interval' in category_mapping and isinstance(category_mapping['interval'], pd.DataFrame):
        interval_rules = category_mapping['interval']
        for col in interval_rules.columns:
            col_mapping = {}
            for rule_str in interval_rules[col].dropna():
                try:
                    # Parse "interval=group" format, keeping the original interval string exactly as is
                    if '=' in rule_str:
                        interval_part, group_str = rule_str.split('=', 1)  # Split only on first '=' to handle edge cases
                        group = int(group_str)
                        # Store the original interval string without modification
                        col_mapping[group] = interval_part
                except (ValueError, IndexError):
                    continue
            if col_mapping:
                reverse_mapping[col] = col_mapping
    
    # Process categorical mappings
    if 'categorical' in category_mapping:
        categorical_mapping = category_mapping['categorical']
        if isinstance(categorical_mapping, dict):
            for feature, mapping_dict in categorical_mapping.items():
                if isinstance(mapping_dict, dict):
                    # mapping_dict is {value: group}
                    col_mapping = {group: str(value) for value, group in mapping_dict.items()}
                    if col_mapping:
                        reverse_mapping[feature] = col_mapping
                elif isinstance(categorical_mapping, pd.DataFrame) and feature in categorical_mapping.columns:
                    # Handle DataFrame format if needed
                    pass
        elif isinstance(categorical_mapping, pd.DataFrame):
            for col in categorical_mapping.columns:
                # For DataFrame format, assume each row is "value=group"
                col_mapping = {}
                for rule_str in categorical_mapping[col].dropna():
                    try:
                        if '=' in rule_str:
                            value_part, group_str = rule_str.split('=', 1)
                            group = int(group_str)
                            col_mapping[group] = value_part
                    except (ValueError, IndexError):
                        continue
                if col_mapping:
                    reverse_mapping[col] = col_mapping
    
    # Process binary mappings (similar to categorical)
    if 'binary' in category_mapping:
        binary_mapping = category_mapping['binary']
        if isinstance(binary_mapping, dict):
            for feature, mapping_dict in binary_mapping.items():
                if isinstance(mapping_dict, dict):
                    col_mapping = {group: str(value) for value, group in mapping_dict.items()}
                    if col_mapping:
                        reverse_mapping[feature] = col_mapping
        elif isinstance(binary_mapping, pd.DataFrame):
            for col in binary_mapping.columns:
                # For DataFrame format, assume each row is "value=group"
                col_mapping = {}
                for rule_str in binary_mapping[col].dropna():
                    try:
                        if '=' in rule_str:
                            value_part, group_str = rule_str.split('=', 1)
                            group = int(group_str)
                            col_mapping[group] = value_part
                    except (ValueError, IndexError):
                        continue
                if col_mapping:
                    reverse_mapping[col] = col_mapping
    
    return reverse_mapping


def reverse_map_rule(rule, reverse_mapping):
    """
    Convert a rule from group numbers to original interval/categorical/binary values.
    Preserves the exact format of interval strings without any modification.
    
    Args:
        rule: Dictionary with {feature: group_number}
        reverse_mapping: Dictionary from build_reverse_mapping: {column_name: {group_number: original_value_string}}
    
    Returns:
        Dictionary with {feature: original_value_string}
    """
    reversed_rule = {}
    for feature, group_num in rule.items():
        if feature in reverse_mapping:
            grp_key = _normalize_group_key(group_num)
            if grp_key in reverse_mapping[feature]:
                # Use the exact original value string (especially important for intervals)
                reversed_rule[feature] = reverse_mapping[feature][grp_key]
            else:
                # Group number not found in mapping, keep as-is with a marker
                reversed_rule[feature] = f"<UNMAPPED_GROUP_{group_num}>"
        else:
            # Feature not in reverse mapping (might be label column or other), keep as-is
            reversed_rule[feature] = group_num
    return reversed_rule


def build_original_value_range_mapping(category_mapping, expanded_df, original_df, prefer_interval_bounds=False, mapped_df=None):
    """
    Build mapping from group numbers to original value ranges using rank intervals.
    If mapped_df is provided, it uses the ACTUAL mapped groups to find the original value range.
    This is 100% accurate as it uses the real assignment evidence.

    Args:
        category_mapping: dict with 'interval' mapping DataFrame (interval=group strings)
        expanded_df: DataFrame used to generate intervals (quantile-expanded)
        original_df: original DataFrame (no quantile expansion)
        prefer_interval_bounds: if True, use rank-space boundaries to find original values
        mapped_df: optional DataFrame with actual group IDs (evidence-based mapping)

    Returns:
        dict: {feature: {group_number: "[min, max]" or interval_str}}
    """
    interval_rules_df = category_mapping.get('interval')
    if not isinstance(interval_rules_df, pd.DataFrame) or interval_rules_df.empty:
        return {}

    def _parse_interval_str(interval_str):
        try:
            interval_str = interval_str.strip()
            left_bracket = interval_str[0]
            right_bracket = interval_str[-1]
            nums = [float(n) for n in re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', interval_str)]
            if len(nums) != 2:
                return None
            closed = 'neither'
            if left_bracket == '[' and right_bracket == ']':
                closed = 'both'
            elif left_bracket == '[':
                closed = 'left'
            elif right_bracket == ']':
                closed = 'right'
            return pd.Interval(nums[0], nums[1], closed=closed)
        except Exception:
            return None

    def _rank_in_interval(rank_series, interval_obj):
        if interval_obj is None:
            return pd.Series([False] * len(rank_series), index=rank_series.index)
        left = interval_obj.left
        right = interval_obj.right
        if interval_obj.closed == 'both':
            return (rank_series >= left) & (rank_series <= right)
        if interval_obj.closed == 'left':
            return (rank_series >= left) & (rank_series < right)
        if interval_obj.closed == 'right':
            return (rank_series > left) & (rank_series <= right)
        return (rank_series > left) & (rank_series < right)

    mapping = {}
    for col in interval_rules_df.columns:
        col_mapping = {}

        # --- 1. Evidence-based mapping (User's Suggestion) ---
        if mapped_df is not None and col in mapped_df.columns and col in original_df.columns:
            try:
                # Group actual raw values by their assigned group IDs
                common_idx = mapped_df.index.intersection(original_df.index)
                if not common_idx.empty:
                    stats = original_df.loc[common_idx, col].groupby(mapped_df.loc[common_idx, col]).agg(['min', 'max'])
                    for group_id, row in stats.iterrows():
                        try:
                            gid = int(float(group_id))
                            col_mapping[gid] = f"[{row['min']}, {row['max']}]"
                        except (ValueError, TypeError):
                            continue
                    if col_mapping:
                        mapping[col] = col_mapping
                        continue
            except Exception:
                pass

        # --- 2. Fallback: Rank-based reconstruction ---
        numeric_series = pd.to_numeric(expanded_df[col], errors='coerce').fillna(0)
        unique_vals = sorted(numeric_series.unique().tolist())
        if not unique_vals:
            continue
            
        if prefer_interval_bounds:
            rank_series = pd.Series(range(1, len(unique_vals) + 1))
            numeric_original = None
        else:
            value_to_rank = {val: idx + 1 for idx, val in enumerate(unique_vals)}
            numeric_original = pd.to_numeric(original_df[col], errors='coerce').fillna(0)
            rank_series = numeric_original.map(value_to_rank)

        for rule_str in interval_rules_df[col].dropna():
            try:
                interval_part, group_str = str(rule_str).split('=', 1)
                group = int(group_str.strip())
                interval_obj = _parse_interval_str(interval_part)
                
                if prefer_interval_bounds:
                    ranks = pd.Series(range(1, len(unique_vals) + 1))
                    r_mask = _rank_in_interval(ranks, interval_obj)
                    interval_values = [unique_vals[i - 1] for i, keep in zip(ranks, r_mask) if keep]
                    if interval_values:
                        col_mapping[group] = f"[{min(interval_values)}, {max(interval_values)}]"
                    else:
                        col_mapping[group] = interval_part.strip()
                else:
                    mask = _rank_in_interval(rank_series, interval_obj)
                    values_in_interval = numeric_original[mask] if numeric_original is not None else None
                    if values_in_interval is not None and not values_in_interval.empty:
                        col_mapping[group] = f"[{values_in_interval.min()}, {values_in_interval.max()}]"
                    else:
                        col_mapping[group] = interval_part.strip()
            except Exception:
                continue
        if col_mapping:
            mapping[col] = col_mapping
    return mapping


def _normalize_group_key(key):
    """Use int for numeric group keys so 0.0 and 0 match mapping lookups (dict keys are int)."""
    if isinstance(key, (int, float)) and key == int(key):
        return int(key)
    return key


def reverse_map_rule_with_fallback(rule, primary_mapping, fallback_mapping):
    """
    Reverse-map rule using primary mapping, fallback to another mapping if needed.
    """
    reversed_rule = {}
    for feature, group_num in rule.items():
        #if feature in primary_mapping and group_num in primary_mapping[feature]:
        grp_key = _normalize_group_key(group_num)
        feat_in_primary = feature in primary_mapping
        #grp_in_primary = group_num in primary_mapping.get(feature, {}) if feat_in_primary else False
        grp_in_primary = grp_key in primary_mapping.get(feature, {}) if feat_in_primary else False
        if feat_in_primary and grp_in_primary:
            '''
            reversed_rule[feature] = primary_mapping[feature][group_num]
        elif feature in fallback_mapping and group_num in fallback_mapping[feature]:
            reversed_rule[feature] = fallback_mapping[feature][group_num]
            # Log only when interval (in primary) but group missing - can cause rank-string in rule
            '''
            reversed_rule[feature] = primary_mapping[feature][grp_key]
        elif feature in fallback_mapping and grp_key in fallback_mapping[feature]:
            # This branch = we used fallback_mapping (build_reverse_mapping). For interval
            # features that is the interval string from category_mapping = rank-space, not value range.
            reversed_rule[feature] = fallback_mapping[feature][grp_key]
            if feat_in_primary and not grp_in_primary:
                val = fallback_mapping[feature][grp_key]
                _log.info(
                    "[FALLBACK] interval feat, grp missing in primary -> used fallback_mapping (rank-str from category_mapping): feat=%s grp=%s val=%s",
                    feature, group_num, val,
                )
        else:
            reversed_rule[feature] = group_num
            if feat_in_primary or feature in fallback_mapping:
                _log.info(
                    "[FALLBACK] unmapped (no fallback for feat/grp): feat=%s grp=%s",
                    feature, group_num,
                )
    return reversed_rule
