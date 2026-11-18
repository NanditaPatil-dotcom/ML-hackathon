"""
PathSnap HMM & Viterbi Route Reconstruction Engine:
Implements sequential map-matching:
- Hidden states: Candidate road segments at each GPS observation
- Emission probabilities: Derived from trained classifier score + spatial distance
- Transition probabilities: Derived from road-network topology (direct connection, loop, or network penalty)
- Viterbi recurrence: Computes globally most coherent route sequence
"""

import numpy as np

class HMMRouteDecoder:
    def __init__(self, arcs, transition_self=0.6, transition_connected=0.35, transition_disconnected=1e-5):
        """
        arcs: numpy array of shape (N_arcs, 2) where row index is arc_id, cols are [from_node, to_node]
        """
        self.arcs = arcs
        self.transition_self = transition_self
        self.transition_connected = transition_connected
        self.transition_disconnected = transition_disconnected
        
        # Build quick adjacency lookup: from_node -> list of (arc_id, to_node)
        self.outgoing = {}
        for arc_id, (u, v) in enumerate(arcs):
            self.outgoing.setdefault(u, []).append(arc_id)
            
    def compute_transition_log_prob(self, prev_arc, curr_arc):
        if prev_arc == curr_arc:
            return np.log(self.transition_self)
            
        u_prev, v_prev = self.arcs[prev_arc]
        u_curr, v_curr = self.arcs[curr_arc]
        
        # If directly connected: previous arc ends where current arc begins
        if v_prev == u_curr:
            return np.log(self.transition_connected)
            
        # If share an intersection node
        if v_prev == v_curr or u_prev == u_curr:
            return np.log(self.transition_connected * 0.2)
            
        # Disconnected penalty
        return np.log(self.transition_disconnected)
        
    def decode(self, candidates_per_obs):
        """
        Viterbi decoding over candidate sequences.
        candidates_per_obs: dict mapping t (0..T-1) -> list of candidate dicts
        Each candidate dict must contain:
            - 'arc_id': int
            - 'score': float (predicted probability from ML model, or 1/(dist+1))
        Returns:
            best_route: list of arc_id values for each observation t
        """
        T = len(candidates_per_obs)
        if T == 0:
            return []
            
        # Filter observations with candidates
        obs_times = sorted(candidates_per_obs.keys())
        
        # dp[t][i] stores max log prob ending at candidate i of observation t
        dp = []
        backpointer = []
        
        # 1. Initialization (t = 0)
        c0 = candidates_per_obs[obs_times[0]]
        dp_0 = []
        for cand in c0:
            emission_prob = max(cand.get("score", 0.5), 1e-4)
            dp_0.append(np.log(emission_prob))
        dp.append(np.array(dp_0))
        backpointer.append([0] * len(c0))
        
        # 2. Viterbi Forward Recursion
        for t_idx in range(1, len(obs_times)):
            t = obs_times[t_idx]
            curr_cands = candidates_per_obs[t]
            prev_cands = candidates_per_obs[obs_times[t_idx - 1]]
            
            dp_t = np.zeros(len(curr_cands))
            bp_t = np.zeros(len(curr_cands), dtype=int)
            
            for j, c_curr in enumerate(curr_cands):
                arc_curr = c_curr["arc_id"]
                emission_log_p = np.log(max(c_curr.get("score", 0.5), 1e-4))
                
                best_val = -np.inf
                best_i = 0
                
                for i, c_prev in enumerate(prev_cands):
                    arc_prev = c_prev["arc_id"]
                    trans_log_p = self.compute_transition_log_prob(arc_prev, arc_curr)
                    val = dp[t_idx - 1][i] + trans_log_p
                    if val > best_val:
                        best_val = val
                        best_i = i
                        
                dp_t[j] = best_val + emission_log_p
                bp_t[j] = best_i
                
            dp.append(dp_t)
            backpointer.append(bp_t)
            
        # 3. Termination & Backtracking
        best_last_idx = int(np.argmax(dp[-1]))
        reconstructed_path = [candidates_per_obs[obs_times[-1]][best_last_idx]["arc_id"]]
        curr_idx = best_last_idx
        
        for t_idx in range(len(obs_times) - 1, 0, -1):
            curr_idx = backpointer[t_idx][curr_idx]
            arc_id = candidates_per_obs[obs_times[t_idx - 1]][curr_idx]["arc_id"]
            reconstructed_path.insert(0, arc_id)
            
        return reconstructed_path

def reconstruct_route_for_trip(trip_dir, trip_id, model, search_radius=75.0, top_k=8):
    """
    End-to-end route reconstruction pipeline for a single trip.
    Batches feature inference for sub-second inference latency.
    """
    from pathsnap.data.legacy_eda import load_trip_track, load_nodes, load_arcs, load_route
    from pathsnap.data.candidate_gen import generate_candidates_for_trip
    from pathsnap.data.feature_eng import match_track_to_route
    
    track = load_trip_track(trip_dir, trip_id)
    nodes = load_nodes(trip_dir, trip_id)
    arcs = load_arcs(trip_dir, trip_id)
    route = load_route(trip_dir, trip_id)
    
    if track is None or nodes is None or arcs is None:
        return None
        
    cands_dict = generate_candidates_for_trip(track, nodes, arcs, search_radius=search_radius, top_k=top_k)
    
    # Batch all candidate feature rows
    all_feat_rows = []
    cand_ptrs = [] # (obs_idx, cand_idx)
    
    for obs_idx, cands in cands_dict.items():
        min_dist = cands[0]["dist_to_segment"] if cands else 0.0
        for cand_idx, c in enumerate(cands):
            all_feat_rows.append([
                c["dist_to_segment"],
                c["projection_ratio"],
                c["heading_diff"],
                c["cos_heading"],
                c["segment_length"],
                c["gps_speed"],
                c["rank_by_dist"],
                c["dist_to_segment"] - min_dist,
                c["dist_to_segment"] / (min_dist + 1e-4)
            ])
            cand_ptrs.append((obs_idx, cand_idx))
            
    # Single batched ML inference call
    if all_feat_rows:
        X_mat = np.array(all_feat_rows, dtype=np.float32)
        probs = model.predict_proba(X_mat)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_mat)
        for (obs_idx, cand_idx), p in zip(cand_ptrs, probs):
            cands_dict[obs_idx][cand_idx]["score"] = float(p)
            
    # 1. Classifier-Only Prediction (Top-1 Argmax)
    clf_path = [max(cands, key=lambda c: c["score"])["arc_id"] for t, cands in cands_dict.items()]
    
    # 2. Classifier + HMM Viterbi Decoding
    decoder = HMMRouteDecoder(arcs)
    hmm_path = decoder.decode(cands_dict)
    
    # Metrics
    metrics = {}
    
    # Continuity
    def calc_continuity(path):
        if len(path) <= 1:
            return 100.0
        same_or_step = sum(path[i] == path[i-1] or arcs[path[i-1]][1] == arcs[path[i]][0] for i in range(1, len(path)))
        return float((same_or_step / (len(path) - 1)) * 100.0)
        
    metrics["clf_continuity"] = calc_continuity(clf_path)
    metrics["hmm_continuity"] = calc_continuity(hmm_path)
    
    # Accuracy against ground truth route if available
    if route is not None:
        true_arcs = match_track_to_route(track, nodes, arcs, route)
        clf_correct = sum(clf_path[i] == true_arcs[i] for i in range(len(clf_path)) if i in true_arcs)
        hmm_correct = sum(hmm_path[i] == true_arcs[i] for i in range(len(hmm_path)) if i in true_arcs)
        total_eval = len([i for i in range(len(clf_path)) if i in true_arcs])
        
        metrics["clf_top1_acc"] = float(clf_correct / max(total_eval, 1) * 100.0)
        metrics["hmm_top1_acc"] = float(hmm_correct / max(total_eval, 1) * 100.0)
        
    return {
        "trip_id": trip_id,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route,
        "clf_path": clf_path,
        "hmm_path": hmm_path,
        "metrics": metrics
    }
