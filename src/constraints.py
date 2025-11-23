from typing import List, Tuple, Set, Dict, Optional
import json
import networkx as nx
import numpy as np

class ConstraintStore:
    """
    Stores and manages user feedback constraints.
    Now supports more complex relationships via a graph structure.
    """
    def __init__(self, baseline_embeddings: Optional[np.ndarray] = None):
        self.baseline_embeddings = baseline_embeddings  # Store baseline for distance tracking

        self.must_links: Set[Tuple[int, int]] = set()
        self.cannot_links: Set[Tuple[int, int]] = set()
        self.miscluster_flags: Set[int] = set()

        # Cluster-level constraints
        self.cluster_merges: List[Tuple[int, int]] = [] # (c_id1, c_id2)
        self.cluster_splits: Set[int] = set() # c_id

        # Semantic constraints
        self.cluster_labels: Dict[int, str] = {} # c_id -> "Label"
        self.emphasized_keywords: Set[str] = set()

        # Graph for transitivity (if A-B and B-C -> A-C)
        self.ml_graph = nx.Graph()

        # Baseline distances for constraints
        self.baseline_distances: Dict[Tuple[int, int], float] = {}

    def add_must_link(self, i: int, j: int):
        if i == j: return
        pair = tuple(sorted((i, j)))

        # Conflict check
        if pair in self.cannot_links:
            print(f"[ConstraintStore] Conflict! Removing CannotLink for {pair}")
            self.cannot_links.remove(pair)

        # Store baseline distance if available
        if self.baseline_embeddings is not None and pair not in self.baseline_distances:
            baseline_dist = self._compute_baseline_distance(i, j)
            self.baseline_distances[pair] = baseline_dist

        self.must_links.add(pair)
        self.ml_graph.add_edge(i, j)

    def add_cannot_link(self, i: int, j: int):
        if i == j: return
        pair = tuple(sorted((i, j)))

        # Conflict check: reachability in ML graph
        if self.ml_graph.has_edge(i, j) or (i in self.ml_graph and j in self.ml_graph and nx.has_path(self.ml_graph, i, j)):
            print(f"[ConstraintStore] Conflict! {pair} are already connected via MustLinks. Ignoring CannotLink.")
            return

        if pair in self.must_links:
            self.must_links.remove(pair)

        # Store baseline distance if available
        if self.baseline_embeddings is not None and pair not in self.baseline_distances:
            baseline_dist = self._compute_baseline_distance(i, j)
            self.baseline_distances[pair] = baseline_dist

        self.cannot_links.add(pair)

    def add_miscluster(self, i: int):
        self.miscluster_flags.add(i)

    def add_cluster_split(self, cluster_id: int):
        self.cluster_splits.add(cluster_id)
        print(f"[ConstraintStore] Marked cluster {cluster_id} for splitting")

    def set_cluster_label(self, cluster_id: int, label: str):
        self.cluster_labels[cluster_id] = label
        print(f"[ConstraintStore] Cluster {cluster_id} labeled as '{label}'")

    def add_emphasized_keyword(self, keyword: str):
        self.emphasized_keywords.add(keyword.lower())
        print(f"[ConstraintStore] Emphasizing concept: '{keyword}'")

    def _compute_baseline_distance(self, i: int, j: int) -> float:
        """Compute baseline distance between two items using learned distance function."""
        if self.baseline_embeddings is None:
            return 0.0

        embeddings = self.baseline_embeddings
        if embeddings.ndim == 3:
            # Multi-head: use learned distance (min across heads)
            head_distances = []
            for k in range(embeddings.shape[1]):
                dist = np.linalg.norm(embeddings[i, k] - embeddings[j, k])
                head_distances.append(dist)
            return min(head_distances)
        else:
            # Single head
            return np.linalg.norm(embeddings[i] - embeddings[j])

    def get_constraint_improvements(self, current_embeddings: np.ndarray) -> Dict:
        """
        Compute how much constraint distances have improved from baseline.

        Returns dict with baseline distances, current distances, and improvements.
        """
        if self.baseline_embeddings is None:
            return {}

        improvements = {
            "must_links": [],
            "cannot_links": [],
            "avg_must_link_improvement": 0.0,
            "avg_cannot_link_improvement": 0.0
        }

        # Process must-links
        for pair in self.must_links:
            if pair in self.baseline_distances:
                baseline_dist = self.baseline_distances[pair]
                current_dist = self._compute_current_distance(pair[0], pair[1], current_embeddings)
                improvement = baseline_dist - current_dist  # Positive = got closer (good for must-links)
                improvements["must_links"].append({
                    "pair": pair,
                    "baseline_distance": baseline_dist,
                    "current_distance": current_dist,
                    "improvement": improvement
                })

        # Process cannot-links
        for pair in self.cannot_links:
            if pair in self.baseline_distances:
                baseline_dist = self.baseline_distances[pair]
                current_dist = self._compute_current_distance(pair[0], pair[1], current_embeddings)
                improvement = current_dist - baseline_dist  # Positive = got farther (good for cannot-links)
                improvements["cannot_links"].append({
                    "pair": pair,
                    "baseline_distance": baseline_dist,
                    "current_distance": current_dist,
                    "improvement": improvement
                })

        # Compute averages
        if improvements["must_links"]:
            improvements["avg_must_link_improvement"] = np.mean([ml["improvement"] for ml in improvements["must_links"]])
        if improvements["cannot_links"]:
            improvements["avg_cannot_link_improvement"] = np.mean([cl["improvement"] for cl in improvements["cannot_links"]])

        return improvements

    def _compute_current_distance(self, i: int, j: int, current_embeddings: np.ndarray) -> float:
        """Compute current distance between two items."""
        if current_embeddings.ndim == 3:
            # Multi-head: use learned distance (min across heads)
            head_distances = []
            for k in range(current_embeddings.shape[1]):
                dist = np.linalg.norm(current_embeddings[i, k] - current_embeddings[j, k])
                head_distances.append(dist)
            return min(head_distances)
        else:
            # Single head
            return np.linalg.norm(current_embeddings[i] - current_embeddings[j])

    def stats(self) -> Dict:
        return {
            "num_must_links": len(self.must_links),
            "num_cannot_links": len(self.cannot_links),
            "num_misclusters": len(self.miscluster_flags),
            "num_labels": len(self.cluster_labels),
            "num_keywords": len(self.emphasized_keywords),
            "ml_components": nx.number_connected_components(self.ml_graph)
        }
