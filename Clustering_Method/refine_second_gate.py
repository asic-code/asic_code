import numpy as np
from sklearn.neighbors import NearestNeighbors


def _sample_indices(indices, max_points, rng):
    arr = np.asarray(indices, dtype=int)
    if arr.size <= max_points:
        return arr
    return rng.choice(arr, size=int(max_points), replace=False)


def build_local_arena_state(X, cluster_labels, protected_indices=None, config=None):
    cfg = config or {}
    rng = np.random.default_rng(int(cfg.get("second_gate_seed", 42)))
    X_arr = np.asarray(X)
    labels = np.asarray(cluster_labels)
    protected_set = set(int(i) for i in (protected_indices or []))

    unique_clusters = np.unique(labels)
    cluster_to_idx = {}
    centroids = {}
    capture_ratio = {}

    for c in unique_clusters:
        idx = np.where(labels == c)[0].astype(int)
        cluster_to_idx[int(c)] = idx
        if idx.size == 0:
            continue
        centroids[int(c)] = X_arr[idx].mean(axis=0)
        prot_cnt = int(sum(1 for i in idx if int(i) in protected_set))
        capture_ratio[int(c)] = float(prot_cnt / max(1, idx.size))

    centroid_keys = sorted(centroids.keys())
    if not centroid_keys:
        return {
            "cluster_to_idx": cluster_to_idx,
            "capture_ratio": capture_ratio,
            "neighbors": {},
            "protected_set": protected_set,
            "rng": rng,
        }
    centroid_mat = np.vstack([centroids[k] for k in centroid_keys])
    dmat = np.linalg.norm(centroid_mat[:, None, :] - centroid_mat[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)

    n_neighbors = max(1, int(cfg.get("second_gate_neighbor_clusters", 4)))
    neighbors = {}
    for i, c in enumerate(centroid_keys):
        order = np.argsort(dmat[i])[:n_neighbors]
        neighbors[int(c)] = [int(centroid_keys[j]) for j in order if np.isfinite(dmat[i, j])]

    return {
        "cluster_to_idx": cluster_to_idx,
        "capture_ratio": capture_ratio,
        "neighbors": neighbors,
        "protected_set": protected_set,
        "rng": rng,
    }


def filter_candidates_with_second_gate(X, candidate_indices, cluster_id, arena_state, config=None):
    cfg = config or {}
    cand = np.asarray(candidate_indices, dtype=int)
    if cand.size == 0:
        return [], {"reason": "no_candidates"}

    X_arr = np.asarray(X)
    neighbors = arena_state.get("neighbors", {})
    cluster_to_idx = arena_state.get("cluster_to_idx", {})
    capture_ratio = arena_state.get("capture_ratio", {})
    protected_set = arena_state.get("protected_set", set())
    rng = arena_state.get("rng", np.random.default_rng(42))

    near_clusters = [int(cluster_id)] + list(neighbors.get(int(cluster_id), []))
    benign_cap_thr = float(cfg.get("second_gate_benign_capture_threshold", 0.10))
    attack_cap_thr = float(cfg.get("second_gate_attack_capture_threshold", 0.02))

    benign_pool = []
    attack_pool = []
    for cc in near_clusters:
        idx = cluster_to_idx.get(int(cc), np.array([], dtype=int))
        if idx.size == 0:
            continue
        cap = float(capture_ratio.get(int(cc), 0.0))
        prot_idx = np.array([i for i in idx if int(i) in protected_set], dtype=int)
        non_prot_idx = np.array([i for i in idx if int(i) not in protected_set], dtype=int)
        if cc == int(cluster_id):
            if prot_idx.size > 0:
                benign_pool.extend(prot_idx.tolist())
        else:
            if cap >= benign_cap_thr and prot_idx.size > 0:
                benign_pool.extend(prot_idx.tolist())
            if cap <= attack_cap_thr and non_prot_idx.size > 0:
                attack_pool.extend(non_prot_idx.tolist())

    min_b = int(cfg.get("second_gate_min_benign_pool", 20))
    min_a = int(cfg.get("second_gate_min_attack_pool", 20))
    fail_open = bool(cfg.get("second_gate_fail_open", True))
    if len(benign_pool) < min_b or len(attack_pool) < min_a:
        if fail_open:
            return cand.tolist(), {"reason": "pool_too_small_fail_open", "benign_pool": len(benign_pool), "attack_pool": len(attack_pool)}
        return [], {"reason": "pool_too_small_fail_closed", "benign_pool": len(benign_pool), "attack_pool": len(attack_pool)}

    b_idx = _sample_indices(benign_pool, int(cfg.get("second_gate_pool_max_benign", 4000)), rng)
    a_idx = _sample_indices(attack_pool, int(cfg.get("second_gate_pool_max_attack", 4000)), rng)
    X_b = X_arr[b_idx]
    X_a = X_arr[a_idx]
    X_cand = X_arr[cand]

    k = max(1, int(cfg.get("second_gate_knn_k", 5)))
    k_b = min(k, X_b.shape[0])
    k_a = min(k, X_a.shape[0])
    if k_b <= 0 or k_a <= 0:
        if fail_open:
            return cand.tolist(), {"reason": "invalid_k_fail_open"}
        return [], {"reason": "invalid_k_fail_closed"}

    nn_b = NearestNeighbors(n_neighbors=k_b, algorithm="auto")
    nn_a = NearestNeighbors(n_neighbors=k_a, algorithm="auto")
    nn_b.fit(X_b)
    nn_a.fit(X_a)
    d_b = nn_b.kneighbors(X_cand, n_neighbors=k_b, return_distance=True)[0].mean(axis=1)
    d_a = nn_a.kneighbors(X_cand, n_neighbors=k_a, return_distance=True)[0].mean(axis=1)
    s_knn = d_b / (d_a + d_b + 1e-12)

    # local benign-only model score: higher means less benign-like
    mu_b = X_b.mean(axis=0)
    std_b = X_b.std(axis=0) + 1e-6
    z2 = ((X_cand - mu_b) / std_b) ** 2
    z_mean = z2.mean(axis=1)
    s_bm = z_mean / (1.0 + z_mean)

    w_knn = float(cfg.get("second_gate_w_knn", 0.6))
    w_bm = float(cfg.get("second_gate_w_benign_model", 0.4))
    s_comb = w_knn * s_knn + w_bm * s_bm

    thr_knn = float(cfg.get("second_gate_knn_min_score", 0.55))
    thr_bm = float(cfg.get("second_gate_benign_model_min_score", 0.55))
    thr_comb = float(cfg.get("second_gate_combined_min_score", 0.58))

    keep_mask = (s_knn >= thr_knn) & (s_bm >= thr_bm) & (s_comb >= thr_comb)
    approved = cand[keep_mask].astype(int).tolist()

    return approved, {
        "reason": "ok",
        "in": int(cand.size),
        "out": int(len(approved)),
        "benign_pool": int(len(b_idx)),
        "attack_pool": int(len(a_idx)),
    }
