import json
import os

import numpy as np
import pandas as pd
from Clustering_Method.cluster_refinement import refine_clusters


def get_default_refine_config():
    return {
        "distance": True,
        "feature_deviation": True,
        "distance_percentile": 95.0,
        "feature_percentile": 95,
        "max_inconsistent_ratio": 0.08,
        "benign_ratio_threshold": 0.90,
        "use_nearest_known_normal": True,
        "w_local_center": 1.0,
        "w_nearest_known_normal": 0.7,
        "w_feature_deviation": 0.9,
        "local_center_mode": "centroid",
        "medoid_max_points": 1200,
        "partial_flip_mode": True,
        "adaptive_ratio_by_cluster_size": True,
        "adaptive_ratio_ref_cluster_size": 200.0,
        "adaptive_ratio_power": 0.5,
        "adaptive_ratio_min_scale": 0.5,
        "adaptive_ratio_max_scale": 1.5,
        "nearest_known_normal_max_points": 5000,
        "nearest_distance_batch_size": 4096,
        "second_gate_enabled": False,
        "second_gate_neighbor_clusters": 4,
        "second_gate_benign_capture_threshold": 0.10,
        "second_gate_attack_capture_threshold": 0.02,
        "second_gate_min_benign_pool": 20,
        "second_gate_min_attack_pool": 20,
        "second_gate_pool_max_benign": 4000,
        "second_gate_pool_max_attack": 4000,
        "second_gate_knn_k": 5,
        "second_gate_knn_min_score": 0.55,
        "second_gate_benign_model_min_score": 0.55,
        "second_gate_combined_min_score": 0.58,
        "second_gate_w_knn": 0.6,
        "second_gate_w_benign_model": 0.4,
        "second_gate_fail_open": True,
        "second_gate_seed": 42,
    }


def load_refine_config_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("config"), dict):
        cfg = payload["config"]
    elif isinstance(payload, dict):
        cfg = payload
    else:
        raise ValueError("Invalid refine config json format.")
    if not isinstance(cfg, dict):
        raise ValueError("Refine config is not a dictionary.")
    merged = get_default_refine_config()
    merged.update(cfg)
    return merged


def build_refine_cfg(
    *,
    max_inconsistent_ratio,
    benign_ratio_threshold,
    distance_percentile,
    feature_percentile,
    w_local_center,
    w_nearest_known_normal,
    w_feature_deviation,
    local_center_mode,
    medoid_max_points,
    nearest_known_normal_max_points,
    nearest_distance_batch_size,
):
    cfg = get_default_refine_config()
    cfg.update(
        {
            "distance_percentile": float(distance_percentile),
            "feature_percentile": float(feature_percentile),
            "max_inconsistent_ratio": float(max_inconsistent_ratio),
            "benign_ratio_threshold": float(benign_ratio_threshold),
            "w_local_center": float(w_local_center),
            "w_nearest_known_normal": float(w_nearest_known_normal),
            "w_feature_deviation": float(w_feature_deviation),
            "local_center_mode": local_center_mode,
            "medoid_max_points": int(medoid_max_points),
            "nearest_known_normal_max_points": int(nearest_known_normal_max_points),
            "nearest_distance_batch_size": int(nearest_distance_batch_size),
        }
    )
    return cfg


def resolve_output_file_type(file_type: str) -> str:
    if file_type in ["CICIDS2017", "CICIDS"]:
        return "CICIDS2017"
    if file_type in ["DARPA", "DARPA98"]:
        return "DARPA98"
    if file_type in ["CICModbus23", "CICModbus"]:
        return "CICModbus23"
    if file_type in ["IoTID20", "IoTID"]:
        return "IoTID20"
    if file_type == "Kitsune":
        return "Kitsune"
    if file_type in ["CICIoT", "CICIoT2023"]:
        return "CICIoT2023"
    return file_type


def default_best_clustering_csv(file_type: str, file_number: int, dataset_ex_dir="../Dataset_ex"):
    out_type = resolve_output_file_type(file_type)
    return os.path.join(dataset_ex_dir, "load_dataset", out_type, f"best_clustering_{out_type}_{file_number}.csv")


def safe_binary_series(s):
    out = pd.to_numeric(pd.Series(s), errors="coerce")
    return out.where(out.isin([0, 1]))


def compute_binary_metrics(y_true, y_pred):
    yt = safe_binary_series(y_true)
    yp = safe_binary_series(y_pred)
    mask = yt.notna() & yp.notna()
    if int(mask.sum()) == 0:
        return {
            "jaccard": 0.0,
            "dice": 0.0,
            "recall_attack": 0.0,
            "precision_attack": 0.0,
            "f1_attack": 0.0,
            "acc": 0.0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "tn": 0,
        }
    yt = yt[mask].astype(int).to_numpy()
    yp = yp[mask].astype(int).to_numpy()
    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    acc = (tp + tn) / len(yt) if len(yt) > 0 else 0.0
    return {
        "jaccard": float(jaccard),
        "dice": float(dice),
        "recall_attack": float(recall),
        "precision_attack": float(precision),
        "f1_attack": float(f1),
        "acc": float(acc),
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tn": tn,
    }


def compute_cluster_pool_purity(y_true, y_pred):
    yt = safe_binary_series(y_true)
    yp = safe_binary_series(y_pred)
    mask = yt.notna() & yp.notna()
    if int(mask.sum()) == 0:
        return {"attack_pool_purity": 0.0, "benign_pool_purity": 0.0}
    yt = yt[mask].astype(int).to_numpy()
    yp = yp[mask].astype(int).to_numpy()
    attack_pool_purity = float(np.mean(yt[yp == 1] == 1)) if np.any(yp == 1) else 0.0
    benign_pool_purity = float(np.mean(yt[yp == 0] == 0)) if np.any(yp == 0) else 0.0
    return {
        "attack_pool_purity": attack_pool_purity,
        "benign_pool_purity": benign_pool_purity,
    }


def load_refine_eval_inputs(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    if "label" not in df.columns or "cluster" not in df.columns:
        raise ValueError(f"CSV must include 'label' and 'cluster': {csv_path}")
    y_true = safe_binary_series(df["label"])
    y_pred = safe_binary_series(df["cluster"])
    valid = y_true.notna() & y_pred.notna()
    work = df.loc[valid].copy()
    work["label"] = y_true[valid].astype(int).values
    work["cluster"] = y_pred[valid].astype(int).values
    feature_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    for c in ["label", "cluster", "adjusted_cluster", "Unnamed:0"]:
        if c in feature_cols:
            feature_cols.remove(c)
    if not feature_cols:
        raise ValueError("No numeric feature columns available for refine evaluation.")
    X = work[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X_np = X.to_numpy(dtype=np.float32)
    labels = work["label"].to_numpy(dtype=np.int32)
    clusters = work["cluster"].to_numpy(dtype=np.int32)
    known_normal_idx = np.where(labels == 0)[0]
    return work, X_np, labels, clusters, known_normal_idx


def run_refine_config_existing_csv_comparison(
    *,
    refine_cfg,
    datasets,
    file_number,
    dataset_ex_dir="../Dataset_ex",
    output_dir="../Dataset_ex/refine_tuning",
    refine_config_json_path=None,
    logger=None,
):
    def _log(msg, level="info"):
        if logger is None:
            return
        fn = getattr(logger, level, None)
        if callable(fn):
            fn(msg)
        elif callable(logger):
            logger(msg)

    datasets = [x.strip() for x in datasets if str(x).strip()]
    if not datasets:
        raise ValueError("No dataset specified for refine comparison")

    os.makedirs(output_dir, exist_ok=True)
    file_no = int(file_number)

    rows = []
    _log(f"[RefineCompare] start datasets={datasets}, file_number={file_no}")
    for ds in datasets:
        csv_path = default_best_clustering_csv(ds, file_no, dataset_ex_dir)
        if not os.path.exists(csv_path):
            rows.append({"file_type": ds, "status": "missing_csv", "csv_path": csv_path})
            _log(f"[RefineCompare] {ds}: missing csv ({csv_path})")
            continue
        try:
            _, X_np, y_true, cluster_before, kn_idx = load_refine_eval_inputs(csv_path)
            before = compute_binary_metrics(y_true, cluster_before)
            before_purity = compute_cluster_pool_purity(y_true, cluster_before)

            refined = refine_clusters(X_np, cluster_before, config=refine_cfg, protected_indices=kn_idx)

            after = compute_binary_metrics(y_true, refined)
            after_purity = compute_cluster_pool_purity(y_true, refined)

            delta_j = float(after["jaccard"] - before["jaccard"])
            delta_d = float(after["dice"] - before["dice"])
            delta_ap = float(after_purity["attack_pool_purity"] - before_purity["attack_pool_purity"])
            delta_bp = float(after_purity["benign_pool_purity"] - before_purity["benign_pool_purity"])
            flip01 = int(np.sum((cluster_before == 0) & (refined == 1)))
            flip10 = int(np.sum((cluster_before == 1) & (refined == 0)))
            n = int(len(refined))
            flip_rate = float((flip01 + flip10) / max(1, n))

            if delta_j >= 0.01 and delta_d >= 0.01:
                impact = "improved"
            elif delta_j <= -0.01 or delta_d <= -0.01:
                impact = "degraded"
            else:
                impact = "stable"

            row = {
                "file_type": ds,
                "status": "ok",
                "rows": n,
                "csv_path": csv_path,
                "flip_0_to_1": flip01,
                "flip_1_to_0": flip10,
                "flip_rate_total": flip_rate,
                "jaccard_before": before["jaccard"],
                "jaccard_after": after["jaccard"],
                "jaccard_delta": delta_j,
                "dice_before": before["dice"],
                "dice_after": after["dice"],
                "dice_delta": delta_d,
                "attack_purity_before": before_purity["attack_pool_purity"],
                "attack_purity_after": after_purity["attack_pool_purity"],
                "attack_purity_delta": delta_ap,
                "benign_purity_before": before_purity["benign_pool_purity"],
                "benign_purity_after": after_purity["benign_pool_purity"],
                "benign_purity_delta": delta_bp,
                "impact_summary": impact,
            }
            rows.append(row)
            _log(
                f"[RefineCompare] {ds}: impact={impact}, "
                f"J {before['jaccard']:.4f}->{after['jaccard']:.4f} ({delta_j:+.4f}), "
                f"D {before['dice']:.4f}->{after['dice']:.4f} ({delta_d:+.4f}), "
                f"AP {before_purity['attack_pool_purity']:.4f}->{after_purity['attack_pool_purity']:.4f} ({delta_ap:+.4f}), "
                f"BP {before_purity['benign_pool_purity']:.4f}->{after_purity['benign_pool_purity']:.4f} ({delta_bp:+.4f}), "
                f"flips(0->1={flip01},1->0={flip10})"
            )
        except Exception as e:
            rows.append({"file_type": ds, "status": f"error: {e}", "csv_path": csv_path})
            _log(f"[RefineCompare] {ds}: failed ({e})", level="warning")

    base = os.path.splitext(os.path.basename(refine_config_json_path))[0] if refine_config_json_path else "default"
    out_csv = os.path.join(output_dir, f"refine_compare_existing_csvs_{base}_n{file_no}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    _log(f"[RefineCompare] saved: {out_csv}")
    return out_csv, rows
