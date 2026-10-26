# A collection of functions for splitting intervals

import pandas as pd
import multiprocessing # Added for parallel processing
import traceback # --- NEW: Import traceback for detailed error logging ---

# Helper function for interval_length_same
def _process_feature_same(args):
    feature_name, series_data, bins_count, labels = args
    try:
        # Ensure series_data is a Series, not a DataFrame column slice that might cause issues
        series_data_copy = series_data.copy()
        binned_series = pd.cut(series_data_copy, bins=bins_count, labels=labels, right=True)
        
        unique_intervals = binned_series.dropna().unique()
        # Sort intervals to ensure consistent mapping, especially if labels are not pre-sorted
        # For numeric labels like range(1, N+1), sorting is implicitly handled by range.
        # If labels could be arbitrary and unsorted, sorting unique_intervals before enumerate is safer.
        interval_to_group = {interval: i for i, interval in enumerate(sorted(list(unique_intervals)))}
        return feature_name, binned_series, interval_to_group
    except Exception as e:
        print(f"Error processing feature {feature_name} in _process_feature_same: {e}")
        return feature_name, pd.Series(dtype='category'), {}

def interval_length_same(df, features, n_splits=40):
    if not features:
        raise ValueError("Error: The `features` list is empty.")

    bins_count = n_splits
    labels = range(1, bins_count + 1)
    
    tasks = []
    for feature_name in features:
        if feature_name not in df.columns:
            raise KeyError(f"Error: Column `{feature_name}` does not exist in `df`.")
        tasks.append((feature_name, df[feature_name], bins_count, labels))

    results_list = []
    if tasks:
        '''
        num_processes = min(len(tasks), multiprocessing.cpu_count())
        # print(f"[interval_length_same] Using {num_processes} processes for {len(tasks)} features.")
        try:
            with multiprocessing.Pool(processes=num_processes) as pool:
                results_list = pool.map(_process_feature_same, tasks)
        except Exception as e:
            print(f"Error during parallel processing in interval_length_same: {e}. Falling back to sequential.")
            results_list = [_process_feature_same(task) for task in tasks]
        '''
        # print(f"[interval_length_same] Processing {len(tasks)} features sequentially.") # Optional: log sequential processing
        results_list = [_process_feature_same(task) for task in tasks] # Changed to sequential processing

    bin_series_list = []
    group_mapping_info = {}
    for feature_name, binned_s, mapping in results_list:
        if not binned_s.empty:
            bin_series_list.append(binned_s.rename(feature_name))
        if mapping:
            group_mapping_info[feature_name] = mapping
            
    if not bin_series_list: # if all features failed or no features to process
        return pd.DataFrame(), {}

    bin_df = pd.concat(bin_series_list, axis=1)
    return bin_df, group_mapping_info

# Helper function for interval_length_Inverse_Count
def _process_feature_inverse(args):
    feature_name, series_data, bins_count = args
    try:
        # Ensure series_data is a Series
        series_data_copy = series_data.copy()
        # --- FIX: Fill NaN values before ranking to prevent qcut failure - can Annotation
        series_data_copy.fillna(0, inplace=True)

        # --- FINAL FIX: Handle columns with a single unique value ---
        if series_data_copy.nunique() <= 1:
            # If only one unique value, qcut will fail. Create a single bin for it.
            # IMPORTANT: Interval_inverse uses rank space for mapping application, so
            # the interval must be defined in rank space (rank=1 for all rows).
            #unique_val = series_data_copy.unique()[0]
            #single_interval = pd.Interval(unique_val, unique_val, closed='both')
            # Use right-closed interval to match qcut default (right-closed)
            # Use a wider epsilon to avoid string rounding collapsing (1,1] to empty.
            single_interval = pd.Interval(1 - 0.5, 1, closed='right')
            # All values in the series map to this one interval.
            binned_series = pd.Series(single_interval, index=series_data_copy.index, dtype='category')
            # The mapping info dictionary maps this interval to group 0.
            interval_to_group = {single_interval: 0}
            return feature_name, binned_series, interval_to_group

        ranked_series = series_data_copy.rank(method="dense")
        #binned_series = pd.qcut(ranked_series, q=bins_count, labels=None, duplicates="drop")

        # Iteratively isolate dominant values based on remaining data + remaining bins.
        # Dominant rule: count(value) > expected_bin_size * alpha (alpha=2 by default).
        alpha = 2
        binned_series = pd.Series(index=series_data_copy.index, dtype='object')
        remaining_mask = pd.Series(True, index=series_data_copy.index)
        remaining_bins = bins_count
        dominant_values = []

        # Track a small sample of isolated dominant values for debug
        dominant_debug = []

        while remaining_bins > 0:
            remaining_values = series_data_copy[remaining_mask]
            if remaining_values.empty:
                break

            remaining_unique = remaining_values.nunique()
            expected_bin_size = len(remaining_values) / max(1, remaining_bins)

            # Stop if no dominant value exists (primary condition).
            value_counts = remaining_values.value_counts()
            top_value = value_counts.index[0]
            top_count = value_counts.iloc[0]
            if top_count <= expected_bin_size * alpha:
                break

            # Additional structural stop conditions (OR).
            if remaining_bins < (bins_count ** 0.5):
                break
            # NOTE: Do not stop on remaining_unique <= remaining_bins when a dominant value exists.
            # Dominant splitting is treated as a normalization step, independent of bin efficiency.

            # Isolate the most dominant value into its own single-rank interval.
            try:
                rank_val = ranked_series[series_data_copy == top_value].iloc[0]
                # Use right-closed interval to keep closure consistent with qcut bins
                # Use a wider epsilon to avoid string rounding collapsing (r,r] to empty.
                interval = pd.Interval(rank_val - 0.5, rank_val, closed='right')
                binned_series.loc[series_data_copy == top_value] = interval
                dominant_values.append(top_value)
                if len(dominant_debug) < 5:
                    dominant_debug.append({
                        "value": top_value,
                        "count": int(top_count),
                        "expected_bin_size": float(expected_bin_size),
                        "rank": float(rank_val)
                    })
                remaining_mask = remaining_mask & (series_data_copy != top_value)
                remaining_bins -= 1
            except Exception:
                break

        # Bin the remaining ranks into the remaining bins
        remaining_ranks = ranked_series[remaining_mask]
        if not remaining_ranks.empty:
            if remaining_ranks.nunique() <= 1 or remaining_bins <= 1:
                interval = pd.Interval(remaining_ranks.min(), remaining_ranks.max(), closed='both')
                binned_series.loc[remaining_mask] = interval
            else:
                remaining_bins = max(1, remaining_bins)
                remaining_bins_result = pd.qcut(remaining_ranks, q=remaining_bins, labels=None, duplicates="drop")
                binned_series.loc[remaining_mask] = remaining_bins_result

        binned_series = binned_series.astype('category')
        if dominant_debug:
            print(f"[DominantSplit] {feature_name}: isolated {len(dominant_values)} values. Sample={dominant_debug}")
        
        unique_intervals = binned_series.dropna().unique()
        # Sort intervals to ensure consistent mapping. pd.Interval objects are sortable.
        interval_to_group = {interval: i for i, interval in enumerate(sorted(list(unique_intervals)))}
        return feature_name, binned_series, interval_to_group
    except Exception as e:
        print(f"Error processing feature {feature_name} in _process_feature_inverse: {e}")
        # --- NEW: Print full traceback to diagnose the root cause ---
        traceback.print_exc()
        return feature_name, pd.Series(dtype='category'), {}

def interval_length_Inverse_Count(df, features, n_splits=40):
    if not features:
        raise ValueError("Error: The `features` list is empty.")

    bins_count = n_splits
    
    tasks = []
    for feature_name in features:
        if feature_name not in df.columns:
            raise KeyError(f"Error: Column `{feature_name}` does not exist in `df`.")
        tasks.append((feature_name, df[feature_name], bins_count))

    results_list = []
    if tasks:
        '''
        num_processes = min(len(tasks), multiprocessing.cpu_count())
        # print(f"[interval_length_Inverse_Count] Using {num_processes} processes for {len(tasks)} features.")
        try:
            with multiprocessing.Pool(processes=num_processes) as pool:
                results_list = pool.map(_process_feature_inverse, tasks)
        except Exception as e:
            print(f"Error during parallel processing in interval_length_Inverse_Count: {e}. Falling back to sequential.")
            results_list = [_process_feature_inverse(task) for task in tasks]
        '''
        # print(f"[interval_length_Inverse_Count] Processing {len(tasks)} features sequentially.") # Optional: log sequential processing
        results_list = [_process_feature_inverse(task) for task in tasks] # Changed to sequential processing
            
    bin_series_list = []
    group_mapping_info = {}
    for feature_name, binned_s, mapping in results_list:
        if not binned_s.empty:
            bin_series_list.append(binned_s.rename(feature_name))
        if mapping:
            group_mapping_info[feature_name] = mapping

    if not bin_series_list: # if all features failed or no features to process
        return pd.DataFrame(), {}
        
    bin_df = pd.concat(bin_series_list, axis=1)
    return bin_df, group_mapping_info