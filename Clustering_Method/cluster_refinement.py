import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from Clustering_Method.refine_second_gate import (
    build_local_arena_state,
    filter_candidates_with_second_gate,
)


# 1. Cluster center utilities

def compute_centroid(X):
    return np.mean(X, axis=0)


def compute_medoid(X):
    D = pairwise_distances(X)
    return X[np.argmin(D.sum(axis=1))]


# 2. Intra-cluster inconsistency detection

def distance_based_outliers(
    X,
    cluster_labels,
    method="centroid",
    percentile=97.5,
    protected_indices=None,
    return_scores=False,
):
    inconsistent = defaultdict(set)
    scores_by_cluster = defaultdict(dict)
    protected = set(int(i) for i in protected_indices) if protected_indices is not None else set()

    for c in np.unique(cluster_labels):
        idx = np.where(cluster_labels == c)[0]
        if protected:
            idx = np.array([i for i in idx if int(i) not in protected], dtype=int)
        Xc = X[idx]

        if len(Xc) < 5:
            continue

        center = compute_centroid(Xc) if method == "centroid" else compute_medoid(Xc)
        dists = np.linalg.norm(Xc - center, axis=1)
        thr = np.percentile(dists, percentile)

        for i, d in zip(idx, dists):
            if d > thr:
                inconsistent[c].add(i)
                if return_scores:
                    scores_by_cluster[c][int(i)] = float(d)
    if return_scores:
        return inconsistent, scores_by_cluster
    return inconsistent


def feature_deviation_outliers(
    X,
    cluster_labels,
    feature_weights=None,
    percentile=95,
    eps=1e-6,
    protected_indices=None,
    return_scores=False,
):
    inconsistent = defaultdict(set)
    scores_by_cluster = defaultdict(dict)
    feature_weights = (
        np.ones(X.shape[1]) if feature_weights is None else feature_weights
    )
    protected = set(int(i) for i in protected_indices) if protected_indices is not None else set()

    for c in np.unique(cluster_labels):
        idx = np.where(cluster_labels == c)[0]
        if protected:
            idx = np.array([i for i in idx if int(i) not in protected], dtype=int)
        Xc = X[idx]

        if len(Xc) < 5:
            continue

        mu = Xc.mean(axis=0)
        sigma = Xc.std(axis=0) + eps

        deviation = np.abs((Xc - mu) / sigma)
        scores = (deviation * feature_weights).mean(axis=1)
        thr = np.percentile(scores, percentile)

        for i, s in zip(idx, scores):
            if s > thr:
                inconsistent[c].add(i)
                if return_scores:
                    scores_by_cluster[c][int(i)] = float(s)
    if return_scores:
        return inconsistent, scores_by_cluster
    return inconsistent


def assign_soft_labels(labels, inconsistent_indices, uncertain_label=-1):
    # This does NOT flip labels; it only assigns a temporary -1.
    labels = labels.copy()
    for idx in inconsistent_indices:
        labels[idx] = uncertain_label
    return labels


# 3. Conservative relabeling logic (Main Logic)

def _normalize_map(values_map):
    if not values_map:
        return {}
    vals = np.array(list(values_map.values()), dtype=float)
    v_min = float(np.min(vals))
    v_max = float(np.max(vals))
    if not np.isfinite(v_min) or not np.isfinite(v_max) or v_max <= v_min:
        return {k: 0.0 for k in values_map}
    return {k: (float(v) - v_min) / (v_max - v_min) for k, v in values_map.items()}


def _build_local_center(X, cluster_indices, protected_set, mode="centroid", medoid_max_points=1500):
    # Prefer protected known-normal rows inside this cluster
    local_idx = [int(i) for i in cluster_indices if int(i) in protected_set]
    if not local_idx:
        local_idx = [int(i) for i in cluster_indices]
    if not local_idx:
        return None
    X_local = X[local_idx]
    if mode == "medoid":
        if len(X_local) > medoid_max_points:
            rng = np.random.default_rng(42)
            keep = rng.choice(len(X_local), size=medoid_max_points, replace=False)
            X_local = X_local[keep]
        return compute_medoid(X_local)
    return compute_centroid(X_local)


def _effective_max_inconsistent_ratio(
    base_ratio,
    cluster_size,
    adaptive_enabled=True,
    ref_cluster_size=200.0,
    power=0.5,
    min_scale=0.5,
    max_scale=1.5,
):
    """
    Size-adaptive ratio:
    - small cluster -> lower effective ratio (more conservative)
    - large cluster -> higher effective ratio (more relaxed)
    """
    base = float(base_ratio)
    if not adaptive_enabled:
        return base
    n = max(1.0, float(cluster_size))
    ref_n = max(1.0, float(ref_cluster_size))
    raw_scale = (n / ref_n) ** float(power)
    scale = min(max(float(raw_scale), float(min_scale)), float(max_scale))
    eff = base * scale
    return min(max(float(eff), 0.0), 1.0)


def conservative_relabel(
    X,
    cluster_labels,
    inconsistent_by_cluster,
    distance_scores_by_cluster=None,
    feature_scores_by_cluster=None,
    nearest_model=None,
    max_inconsistent_ratio=0.1,
    benign_ratio_threshold=0.7,
    w_local_center=1.0,
    w_nearest=0.7,
    w_feature=0.9,
    local_center_mode="centroid",
    medoid_max_points=1500,
    partial_flip_mode=True,
    nearest_batch_size=4096,
    adaptive_ratio_by_cluster_size=True,
    adaptive_ratio_ref_cluster_size=200.0,
    adaptive_ratio_power=0.5,
    adaptive_ratio_min_scale=0.5,
    adaptive_ratio_max_scale=1.5,
    protected_indices=None,
    second_gate_enabled=False,
    second_gate_state=None,
    second_gate_config=None,
):
    """
    Rules:
    - attack -> benign flip: NEVER
    - Inconsistent samples from benign-heavy clusters are flipped to attack
    - If the inconsistent ratio is too high, the cluster refinement is aborted
    """

    refined = np.asarray(cluster_labels).copy()
    protected = set(int(i) for i in protected_indices) if protected_indices is not None else set()
    distance_scores_by_cluster = distance_scores_by_cluster or {}
    feature_scores_by_cluster = feature_scores_by_cluster or {}

    for c, inconsistent_idx in inconsistent_by_cluster.items():
        idx = np.where(cluster_labels == c)[0]
        cluster_size = len(idx)

        if cluster_size == 0:
            continue

        candidates = [int(i) for i in inconsistent_idx if int(i) not in protected and refined[int(i)] == 0]
        if not candidates:
            continue
        ratio = len(candidates) / cluster_size

        # Cluster majority determination
        benign_ratio = np.mean(cluster_labels[idx] == 0)
        if benign_ratio < benign_ratio_threshold:
            continue

        # Build score components for candidates
        local_center = _build_local_center(
            X, idx, protected, mode=local_center_mode, medoid_max_points=medoid_max_points
        )
        local_scores = {}
        if local_center is not None:
            cand_arr = np.array(candidates, dtype=int)
            d = np.linalg.norm(X[cand_arr] - local_center, axis=1)
            local_scores = {int(i): float(v) for i, v in zip(cand_arr, d)}

        nearest_scores = {}
        if nearest_model is not None and candidates:
            cand_arr = np.array(candidates, dtype=int)
            for s in range(0, len(cand_arr), nearest_batch_size):
                part = cand_arr[s:s + nearest_batch_size]
                dist, _ = nearest_model.kneighbors(X[part], n_neighbors=1)
                for i, dv in zip(part, dist[:, 0]):
                    nearest_scores[int(i)] = float(dv)

        feat_scores = feature_scores_by_cluster.get(c, {})
        dist_scores = distance_scores_by_cluster.get(c, {})

        # Use distance outlier score as fallback where local center distance is unavailable
        if local_scores:
            local_component = local_scores
        else:
            local_component = {i: float(dist_scores.get(i, 0.0)) for i in candidates}

        n_local = _normalize_map(local_component)
        n_near = _normalize_map(nearest_scores)
        n_feat = _normalize_map({i: float(feat_scores.get(i, 0.0)) for i in candidates})

        scored = []
        for i in candidates:
            score = (
                w_local_center * n_local.get(i, 0.0)
                + w_nearest * n_near.get(i, 0.0)
                + w_feature * n_feat.get(i, 0.0)
            )
            scored.append((i, float(score)))
        scored.sort(key=lambda t: t[1], reverse=True)

        eff_ratio = _effective_max_inconsistent_ratio(
            max_inconsistent_ratio,
            cluster_size,
            adaptive_enabled=adaptive_ratio_by_cluster_size,
            ref_cluster_size=adaptive_ratio_ref_cluster_size,
            power=adaptive_ratio_power,
            min_scale=adaptive_ratio_min_scale,
            max_scale=adaptive_ratio_max_scale,
        )

        if partial_flip_mode:
            k = int(np.floor(eff_ratio * cluster_size))
            if k <= 0:
                k = 1
            flip_targets = [i for i, _ in scored[: min(k, len(scored))]]
        else:
            if ratio > eff_ratio:
                continue
            flip_targets = [i for i, _ in scored]

        print(
            f"[Refine Debug] Cluster {c}: size={cluster_size}, candidates={len(candidates)}, "
            f"ratio={ratio:.3f}, eff_ratio={eff_ratio:.3f}, benign_ratio={benign_ratio:.3f}, flip={len(flip_targets)}"
        )

        if second_gate_enabled:
            approved_targets, gate_info = filter_candidates_with_second_gate(
                X,
                flip_targets,
                c,
                second_gate_state,
                config=(second_gate_config or {}),
            )
            print(
                f"[Refine 2ndGate] Cluster {c}: in={len(flip_targets)}, out={len(approved_targets)}, "
                f"reason={gate_info.get('reason')}"
            )
            flip_targets = approved_targets

        for i in flip_targets:
            refined[i] = 1
    # Hard protection: known-normal indices must stay benign.
    if protected:
        for i in protected:
            if 0 <= int(i) < len(refined):
                refined[int(i)] = 0
    
    before = np.sum(cluster_labels == 1)
    after  = np.sum(refined == 1)
    print(f"[Refine Debug] Attack count: {before} -> {after} (Δ={after-before})")

    return refined


# 4. Unified refinement interface (pipeline entry point)

def refine_clusters(
    X,
    cluster_labels,
    feature_weights=None,
    config=None,
    protected_indices=None,
):
    """
    Final refinement entry.
    Always returns refined {0,1} labels.
    """

    config = config or {}
    X_arr = np.asarray(X)
    protected = sorted({int(i) for i in protected_indices}) if protected_indices is not None else []

    # Step 1: detect inconsistent samples
    inconsistent = defaultdict(set)
    distance_scores = defaultdict(dict)
    feature_scores = defaultdict(dict)

    if config.get("distance", True):
        out_d, score_d = distance_based_outliers(
            X_arr,
            cluster_labels,
            method=config.get("center", "centroid"),
            percentile=config.get("distance_percentile", 80.0),
            protected_indices=protected,
            return_scores=True,
        )
        for c in out_d:
            inconsistent[c] |= out_d[c]
            distance_scores[c].update(score_d.get(c, {}))

    if config.get("feature_deviation", True):
        out_f, score_f = feature_deviation_outliers(
            X_arr,
            cluster_labels,
            feature_weights=feature_weights,
            percentile=config.get("feature_percentile", 85),
            protected_indices=protected,
            return_scores=True,
        )
        for c in out_f:
            inconsistent[c] |= out_f[c]
            feature_scores[c].update(score_f.get(c, {}))

    second_gate_enabled = bool(config.get("second_gate_enabled", False))
    second_gate_state = None
    if second_gate_enabled:
        second_gate_state = build_local_arena_state(
            X_arr,
            cluster_labels,
            protected_indices=protected,
            config=config,
        )

    # Build nearest-known-normal model once to limit CPU/memory overhead.
    nearest_model = None
    if config.get("use_nearest_known_normal", True) and protected:
        ref_idx = np.array(protected, dtype=int)
        max_ref = int(config.get("nearest_known_normal_max_points", 5000))
        if len(ref_idx) > max_ref:
            rng = np.random.default_rng(42)
            ref_idx = rng.choice(ref_idx, size=max_ref, replace=False)
        nearest_model = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nearest_model.fit(X_arr[ref_idx])

    # Step 2: conservative relabeling
    refined_labels = conservative_relabel(
        X_arr,
        cluster_labels,
        inconsistent,
        distance_scores_by_cluster=distance_scores,
        feature_scores_by_cluster=feature_scores,
        nearest_model=nearest_model,
        max_inconsistent_ratio=config.get("max_inconsistent_ratio", 0.25),
        benign_ratio_threshold=config.get("benign_ratio_threshold", 0.65),
        w_local_center=float(config.get("w_local_center", 1.0)),
        w_nearest=float(config.get("w_nearest_known_normal", 0.7)),
        w_feature=float(config.get("w_feature_deviation", 0.9)),
        local_center_mode=config.get("local_center_mode", "centroid"),
        medoid_max_points=int(config.get("medoid_max_points", 1500)),
        partial_flip_mode=bool(config.get("partial_flip_mode", True)),
        nearest_batch_size=int(config.get("nearest_distance_batch_size", 4096)),
        adaptive_ratio_by_cluster_size=bool(config.get("adaptive_ratio_by_cluster_size", True)),
        adaptive_ratio_ref_cluster_size=float(config.get("adaptive_ratio_ref_cluster_size", 200.0)),
        adaptive_ratio_power=float(config.get("adaptive_ratio_power", 0.5)),
        adaptive_ratio_min_scale=float(config.get("adaptive_ratio_min_scale", 0.5)),
        adaptive_ratio_max_scale=float(config.get("adaptive_ratio_max_scale", 1.5)),
        protected_indices=protected,
        second_gate_enabled=second_gate_enabled,
        second_gate_state=second_gate_state,
        second_gate_config=config,
    )

    return refined_labels
