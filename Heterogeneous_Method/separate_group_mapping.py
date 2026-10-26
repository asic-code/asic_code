# Input embedded_data (after separate_bin)

import json
import pandas as pd
from Heterogeneous_Method.build_interval_mapping import build_interval_mapping_dataframe
import multiprocessing # Added for parallel processing
import re # --- NEW: Import re for parsing interval strings ---

# Global in-memory buffer for optional debugging (no files). Only used when enabled.
_INTERVAL_MAPPING_DEBUG_BUFFER = []
_INTERVAL_MAPPING_DEBUG_MAX_ENTRIES = 500  # safety cap

def flush_interval_mapping_debug(top_payloads=30):
    """
    Clear the in-memory interval mapping debug buffer.
    """
    global _INTERVAL_MAPPING_DEBUG_BUFFER
    if not _INTERVAL_MAPPING_DEBUG_BUFFER:
        return
    _INTERVAL_MAPPING_DEBUG_BUFFER = []

# Helper function for parallel interval mapping
def _map_single_interval_column(args):
    col_name, col_data, col_mapping_series, debug, value_rank_map = args
    debug_info = {
        "col": col_name,
        "branch": None,
        "rules_total": int(col_mapping_series.dropna().shape[0]) if hasattr(col_mapping_series, "dropna") else None,
    }
    try:
        # Convert category dtype to object for string operations
        if col_data.dtype.name == 'category':
            col_data = col_data.astype(str)
        
        # Define regex pattern outside f-string to avoid backslash issue
        interval_pattern = r'\(.*\]'
        # Only use .str accessor if the dtype is object/string; otherwise assume numeric
        contains_intervals = False
        if col_data.dtype == 'object' or pd.api.types.is_string_dtype(col_data):
            try:
                contains_intervals = col_data.str.contains(interval_pattern, na=False).any()
            except (AttributeError, TypeError):
                # If .str accessor fails, assume numeric data (no interval strings)
                contains_intervals = False
        if debug:
            debug_info["dtype"] = str(col_data.dtype)
            debug_info["contains_interval_strings"] = bool(contains_intervals)
        
        if col_data.dtype == 'object' and contains_intervals:
            debug_info["branch"] = "string_interval_map"
            # For interval strings, we need to map them directly to group numbers
            # Create a mapping from interval strings to group numbers
            interval_to_group = {}
            parse_fail_samples = []
            parse_fail_count = 0
            for s in col_mapping_series.dropna():
                try:
                    interval_str, group_num_str = s.split('=')
                    group_num = int(group_num_str.strip())
                    interval_to_group[interval_str.strip()] = group_num
                except:
                    parse_fail_count += 1
                    if debug and len(parse_fail_samples) < 3:
                        parse_fail_samples.append(str(s))
                    continue
            
            # Map interval strings directly to group numbers
            mapped_series = col_data.map(interval_to_group)
            final_mapping_info = {str(k): v for k, v in interval_to_group.items()}
            if debug:
                debug_info["mapping_keys"] = int(len(interval_to_group))
                debug_info["parse_fail_count"] = int(parse_fail_count)
                if parse_fail_samples:
                    debug_info["parse_fail_samples"] = parse_fail_samples
                # Unmatched interval strings (exact string mismatch)
                uniq_vals = pd.Series(col_data).dropna().astype(str).unique()
                unmatched = [v for v in uniq_vals if v not in interval_to_group]
                debug_info["unique_interval_strings_in_data"] = int(len(uniq_vals))
                debug_info["unmatched_interval_strings_count"] = int(len(unmatched))
                if unmatched:
                    debug_info["unmatched_interval_strings_sample"] = [str(x) for x in unmatched[:5]]
                debug_info["nan_after_map"] = int(pd.Series(mapped_series).isna().sum())
                debug_info["rows"] = int(len(mapped_series))
            return col_name, mapped_series, final_mapping_info, debug_info
        else:
            debug_info["branch"] = "numeric_cut_map"
            # Ensure input data is numeric, coercing errors to NaN, then filling with 0.
            numeric_coerced = pd.to_numeric(col_data, errors='coerce')
            if debug:
                debug_info["nan_after_coerce"] = int(numeric_coerced.isna().sum())
                debug_info["filled_to_zero_count"] = int(numeric_coerced.isna().sum())
            numeric_col_data = numeric_coerced.fillna(0)

        # Step 1: Parse the string mapping back into a list of Interval objects and group numbers
        parsed_mapping = []
        parse_fail_samples = []
        parse_fail_count = 0
        saw_inf_token = False
        for s in col_mapping_series.dropna():
            try:
                interval_str, group_num_str = s.split('=')
                group_num = int(group_num_str.strip())
                
                interval_str = interval_str.strip()
                if 'inf' in interval_str.lower():
                    saw_inf_token = True
                left_bracket = interval_str[0]
                right_bracket = interval_str[-1]
                
                #nums = [float(f) for f in re.findall(r'-?\\d+\\.?\\d*e?[-+]?\\d*', interval_str)]
                # Try multiple regex patterns to extract numbers
                nums = []
                # Pattern 1: Standard decimal numbers
                pattern1 = r'-?\d+\.?\d*'
                matches1 = re.findall(pattern1, interval_str)
                matches2 = []  # Initialize matches2 to avoid UnboundLocalError
                
                if len(matches1) >= 2:
                    nums = [float(f) for f in matches1[:2]]
                else:
                    # Pattern 2: More flexible pattern
                    pattern2 = r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
                    matches2 = re.findall(pattern2, interval_str)
                    if len(matches2) >= 2:
                        nums = [float(f) for f in matches2[:2]]
                
                if len(nums) != 2: 
                    parse_fail_count += 1
                    if debug and len(parse_fail_samples) < 3:
                        parse_fail_samples.append(interval_str)
                    continue
                
                closed = 'neither'
                if left_bracket == '[' and right_bracket == ']': closed = 'both'
                elif left_bracket == '[': closed = 'left'
                elif right_bracket == ']': closed = 'right'
                
                interval = pd.Interval(nums[0], nums[1], closed=closed)
                parsed_mapping.append({'interval': interval, 'group': group_num})
            except Exception as e:
                parse_fail_count += 1
                if debug and len(parse_fail_samples) < 3:
                    parse_fail_samples.append(str(s))
                continue

        if not parsed_mapping:
            if debug:
                debug_info["parsed_intervals"] = 0
                debug_info["parse_fail_count"] = int(parse_fail_count)
                if parse_fail_samples:
                    debug_info["parse_fail_samples"] = parse_fail_samples
                debug_info["saw_inf_token_in_rules"] = bool(saw_inf_token)
                debug_info["reason"] = "parsed_mapping_empty"
                debug_info["rows"] = int(len(numeric_col_data))
            return col_name, pd.Series(float('nan'), index=numeric_col_data.index), {}, debug_info

        # Step 2: Create a pd.IntervalIndex for efficient mapping
        parsed_mapping.sort(key=lambda x: x['interval'].left)
        intervals = [d['interval'] for d in parsed_mapping]
        interval_index = pd.IntervalIndex(intervals)

        # Step 3: Use rank-based mapping to align with Interval_inverse (qcut on ranked values)
        # This ensures the mapping is applied in the same space as interval creation.
        if value_rank_map:
            ranked_series = numeric_col_data.map(value_rank_map)
            if debug:
                debug_info["rank_unmapped_count"] = int(pd.Series(ranked_series).isna().sum())
        else:
            ranked_series = numeric_col_data.rank(method="dense")
        #binned_series_with_intervals = pd.cut(ranked_series, bins=interval_index)

        # Step 4: Map the resulting pd.Interval objects to their corresponding group numbers
        interval_to_group_map = {d['interval']: d['group'] for d in parsed_mapping}

        # If intervals overlap, fall back to manual mapping (prefer smaller intervals first).
        if interval_index.is_overlapping:
            if debug:
                debug_info["overlapping_intervals"] = True
            mapped_series = pd.Series([float('nan')] * len(ranked_series), index=ranked_series.index)
            sorted_intervals = sorted(parsed_mapping, key=lambda x: (x['interval'].right - x['interval'].left))
            for item in sorted_intervals:
                interval = item['interval']
                group = item['group']
                if interval.closed == 'both':
                    mask = (ranked_series >= interval.left) & (ranked_series <= interval.right)
                elif interval.closed == 'left':
                    mask = (ranked_series >= interval.left) & (ranked_series < interval.right)
                elif interval.closed == 'right':
                    mask = (ranked_series > interval.left) & (ranked_series <= interval.right)
                else:
                    mask = (ranked_series > interval.left) & (ranked_series < interval.right)
                assign_mask = mask & mapped_series.isna()
                if assign_mask.any():
                    mapped_series.loc[assign_mask] = group
            binned_series_with_intervals = pd.Series([None] * len(ranked_series), index=ranked_series.index)
        else:
            binned_series_with_intervals = pd.cut(ranked_series, bins=interval_index)
            mapped_series = binned_series_with_intervals.map(interval_to_group_map)

        final_mapping_info = {str(k): v for k, v in interval_to_group_map.items()}
        if debug:
            debug_info["parsed_intervals"] = int(len(parsed_mapping))
            debug_info["parse_fail_count"] = int(parse_fail_count)
            if parse_fail_samples:
                debug_info["parse_fail_samples"] = parse_fail_samples
            debug_info["saw_inf_token_in_rules"] = bool(saw_inf_token)
            debug_info["nan_after_cut"] = int(pd.Series(binned_series_with_intervals).isna().sum())
            debug_info["nan_after_map"] = int(pd.Series(mapped_series).isna().sum())
            debug_info["rows"] = int(len(mapped_series))
            # If NaNs exist, sample the raw values that produced NaN at cut-stage
            nan_mask = pd.Series(binned_series_with_intervals).isna()
            if nan_mask.any():
                sample_vals = pd.Series(numeric_col_data)[nan_mask].unique()[:5]
                sample_ranks = pd.Series(ranked_series)[nan_mask].unique()[:5]
                debug_info["nan_after_cut_value_sample"] = [float(x) for x in sample_vals.tolist()]
                debug_info["nan_after_cut_rank_sample"] = [float(x) for x in sample_ranks.tolist()]
        return col_name, mapped_series, final_mapping_info, debug_info
        
    except Exception as e:
        # Fallback for any unexpected error during the process
        if debug:
            debug_info["reason"] = "exception"
            debug_info["exception"] = repr(e)
            debug_info["rows"] = int(len(col_data))
        return col_name, pd.Series(float('nan'), index=col_data.index), {}, debug_info


def map_intervals_to_groups(df, category_mapping, data_list, regul='N', debug=False, debug_context=None, debug_top_n=30, value_rank_map=None):
    # mapped_df = pd.DataFrame() # Initialize later from results
    mapping_info = {}   # Save per-feature mapping information
    debug_enabled = bool(debug)

    # Ensure 'interval' key exists and is a DataFrame
    if 'interval' not in category_mapping or not isinstance(category_mapping['interval'], pd.DataFrame):
        # If no interval mapping, or mapping is not in expected format, handle appropriately
        # For now, assume interval_columns would be empty, or raise error
        interval_columns = []
        interval_df = pd.DataFrame(index=df.index) # Empty DataFrame with original index
    else:
        interval_columns = [col for col in category_mapping['interval'].columns if col in df.columns]
        if not interval_columns:
            interval_df = pd.DataFrame(index=df.index) # Empty DataFrame with original index
        else:
            interval_df = df[interval_columns]  # Organize only the conditions that want to map

    interval_mapping_df = category_mapping.get('interval', pd.DataFrame()) # interval information is taken from here

    tasks = []
    for col in interval_df.columns: # Iterate over columns present in both df and interval_mapping_df
        if col not in interval_mapping_df.columns:
            # This case should be less likely now due to pre-filtering interval_columns
            # print(f"Warning: Interval mapping for column `{col}` is missing in interval_mapping_df. Skipping.")
            # mapped_df[col] = interval_df[col] # Keep original if no mapping? Or NaN?
            continue 
        col_rank_map = value_rank_map.get(col) if isinstance(value_rank_map, dict) else None
        tasks.append((col, interval_df[col], interval_mapping_df[col], debug_enabled, col_rank_map))

    processed_columns = {}
    debug_infos = {}
    if tasks:
        num_processes = min(len(tasks), multiprocessing.cpu_count())
        # print(f"[MapIntervals] Using {num_processes} processes for {len(tasks)} interval columns.")
        try:
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(_map_single_interval_column, tasks)
        except Exception as e:
            results = [_map_single_interval_column(task) for task in tasks]
        
        for col_name, mapped_series, i_to_g, dbg in results:
            processed_columns[col_name] = mapped_series
            mapping_info[col_name] = i_to_g
            debug_infos[col_name] = dbg
    
    if processed_columns:
        mapped_df_from_intervals = pd.DataFrame(processed_columns, index=interval_df.index) # Ensure correct index
    else:
        # Ensure mapped_df is initialized correctly if no interval columns were processed
        mapped_df_from_intervals = pd.DataFrame(index=df.index) 

    # Concatenation logic (ensure data_list[0] and data_list[-1] have compatible indices)
    # It is assumed that data_list[0] (categorical) and data_list[-1] (binary) are already processed 
    # and have an index compatible with df and thus with mapped_df_from_intervals.
    # If data_list elements are empty DataFrames, they should also have the correct index.
    
    # Ensure all parts have an index, default to df.index if they are empty and don't have one
    part1 = data_list[0] if not data_list[0].empty else pd.DataFrame(index=df.index)
    part3 = data_list[len(data_list)-1] if not data_list[len(data_list)-1].empty else pd.DataFrame(index=df.index)

    # Ensure mapped_df_from_intervals also has the correct index if it's empty
    if mapped_df_from_intervals.empty and not processed_columns:
        mapped_df_from_intervals = pd.DataFrame(index=df.index)

    # Filter out empty DataFrames before concat to avoid issues if one part is truly empty of columns but has an index.
    # Concat will handle empty DFs gracefully if they have an index. The main concern is if they are None or lack an index.
    dfs_to_concat = []
    if not part1.empty or part1.shape[1] > 0: dfs_to_concat.append(part1)
    if not mapped_df_from_intervals.empty or mapped_df_from_intervals.shape[1] > 0: dfs_to_concat.append(mapped_df_from_intervals)
    if not part3.empty or part3.shape[1] > 0: dfs_to_concat.append(part3)
    
    if dfs_to_concat:
        mapped_df = pd.concat(dfs_to_concat, axis=1)
    else: # All parts were empty
        mapped_df = pd.DataFrame(index=df.index)
    
    mapped_info_df = build_interval_mapping_dataframe(mapping_info)

    # Update category_mapping['interval'] with the new mapping info
    if 'interval' in category_mapping and not mapped_info_df.empty:
        # Update the interval mapping with the new mapping info
        category_mapping['interval'] = mapped_info_df
    
    # Optional: write consolidated debug summary (at end of this function)
    if debug_enabled:
        reports = []
        for col in interval_columns:
            if col not in debug_infos:
                continue
            rep = debug_infos[col] or {}
            # Add rule samples for suspected parsing issues (inf tokens, parse failures)
            try:
                rules = interval_mapping_df[col].dropna().astype(str).tolist()
            except Exception:
                rules = []
            if rules and ("parse_fail_count" in rep and rep.get("parse_fail_count", 0) > 0) and "rule_samples" not in rep:
                rep["rule_samples"] = rules[:3]
            reports.append(rep)
        # Sort by NaN count (descending) then by parse failures
        def _sort_key(r):
            return (
                -(r.get("nan_after_map", 0) or 0),
                -(r.get("nan_after_cut", 0) or 0),
                -(r.get("parse_fail_count", 0) or 0),
            )
        reports.sort(key=_sort_key)
        payload = {
            "context": debug_context or {},
            "rows": int(len(df)),
            "interval_columns": int(len(interval_columns)),
            "problem_columns": [r for r in reports if (r.get("nan_after_map", 0) or 0) > 0 or r.get("reason")],
            "top_columns_by_nan": reports[:max(0, int(debug_top_n))],
        }
        # Append to global buffer (capped) so caller can flush once at end.
        global _INTERVAL_MAPPING_DEBUG_BUFFER
        _INTERVAL_MAPPING_DEBUG_BUFFER.append(payload)
        if len(_INTERVAL_MAPPING_DEBUG_BUFFER) > _INTERVAL_MAPPING_DEBUG_MAX_ENTRIES:
            # Drop oldest to keep memory bounded
            _INTERVAL_MAPPING_DEBUG_BUFFER = _INTERVAL_MAPPING_DEBUG_BUFFER[-_INTERVAL_MAPPING_DEBUG_MAX_ENTRIES:]

    if regul in ["N", "n"]:
        mapped_info_df = pd.concat([
            category_mapping.get('categorical', pd.DataFrame()),
            mapped_info_df,
            category_mapping.get('binary', pd.DataFrame())
        ], axis=1, ignore_index=False)

    return mapped_df, mapped_info_df
