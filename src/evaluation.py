"""
Evaluation module for clustering quality assessment and progress tracking.

Provides comprehensive metrics for:
- Standard clustering quality measures
- Constraint satisfaction rates
- Cluster stability across iterations
- Item-level uncertainty analysis
- Progress reporting and comparison
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import torch


@dataclass
class ClusterMetrics:
    """Container for standard clustering quality metrics."""
    silhouette_score: Union[float, str]
    calinski_harabasz_score: Union[float, str]
    davies_bouldin_score: Union[float, str]
    cluster_sizes: List[int]
    cluster_variances: List[float]
    # Multi-head learned metrics
    learned_silhouette_score: Optional[Union[float, str]] = None
    learned_calinski_harabasz_score: Optional[Union[float, str]] = None
    learned_davies_bouldin_score: Optional[Union[float, str]] = None


@dataclass
class ConstraintMetrics:
    """Container for constraint satisfaction metrics."""
    must_link_satisfaction: float
    cannot_link_satisfaction: float
    total_must_links: int
    total_cannot_links: int
    satisfied_must_links: int
    satisfied_cannot_links: int
    # Learned distance-based constraint metrics
    avg_must_link_distance: Optional[float] = None
    avg_cannot_link_distance: Optional[float] = None
    constraint_separation_score: Optional[float] = None


@dataclass
class StabilityMetrics:
    """Container for cluster stability metrics."""
    assignment_stability: float
    cluster_purity_stability: Optional[float] = None
    centroid_shift: Optional[float] = None


@dataclass
class UncertaintyMetrics:
    """Container for item-level uncertainty analysis."""
    uncertain_items: List[Tuple[int, Dict]]
    boundary_pairs: List[Tuple[int, int, float]]
    high_variance_clusters: List[Tuple[int, float]]


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""
    cluster_metrics: ClusterMetrics
    constraint_metrics: ConstraintMetrics
    stability_metrics: Optional[StabilityMetrics]
    uncertainty_metrics: UncertaintyMetrics
    timestamp: str
    iteration: int = 0
    constraint_improvements: Optional[Dict] = None


class ClusterEvaluator:
    """
    Comprehensive evaluator for clustering quality and progress tracking.

    Provides methods to assess:
    - Clustering quality using standard metrics
    - Constraint satisfaction rates
    - Stability across iterations
    - Item-level uncertainty analysis
    """

    def __init__(self):
        self.previous_labels: Optional[np.ndarray] = None
        self.previous_centroids: Optional[np.ndarray] = None
        self.iteration_count = 0

    def evaluate_clustering(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        labels: np.ndarray,
        centroids: Optional[np.ndarray] = None
    ) -> ClusterMetrics:
        """
        Evaluate clustering quality using standard metrics.

        Args:
            embeddings: [N, D] or [N, num_heads, D] embeddings
            labels: [N] cluster assignments
            centroids: [K, D] or [K, num_heads, D] cluster centroids

        Returns:
            ClusterMetrics with quality scores and statistics
        """
        # Handle multi-head embeddings
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        original_embeddings = embeddings.copy()
        is_multihead = embeddings.ndim == 3

        if embeddings.ndim == 3:
            # For standard metrics, flatten multi-head embeddings
            embeddings = embeddings.reshape(embeddings.shape[0], -1)

        if centroids is not None and centroids.ndim == 3:
            centroids_flat = centroids.reshape(centroids.shape[0], -1)
        else:
            centroids_flat = centroids

        # Compute standard metrics
        try:
            silhouette = silhouette_score(embeddings, labels)
        except ValueError:
            silhouette = "N/A (need at least 2 clusters or samples)"

        try:
            ch_score = calinski_harabasz_score(embeddings, labels)
        except ValueError:
            ch_score = "N/A"

        try:
            db_score = davies_bouldin_score(embeddings, labels)
        except ValueError:
            db_score = "N/A"

        # Compute learned metrics using multi-head distance function
        learned_silhouette = None
        learned_ch_score = None
        learned_db_score = None

        if is_multihead and centroids is not None:
            try:
                learned_silhouette = self._compute_learned_silhouette_score(original_embeddings, labels, centroids)
            except (ValueError, Exception):
                learned_silhouette = "N/A"

            try:
                learned_ch_score = self._compute_learned_calinski_harabasz(original_embeddings, labels, centroids)
            except (ValueError, Exception):
                learned_ch_score = "N/A"

            try:
                learned_db_score = self._compute_learned_davies_bouldin(original_embeddings, labels, centroids)
            except (ValueError, Exception):
                learned_db_score = "N/A"

        # Compute cluster statistics
        cluster_sizes = []
        cluster_variances = []

        for k in range(len(np.unique(labels))):
            mask = (labels == k)
            cluster_sizes.append(np.sum(mask))

            if np.any(mask):
                cluster_points = embeddings[mask]
                if centroids is not None:
                    center = centroids[k]
                else:
                    center = np.mean(cluster_points, axis=0)

                # Mean squared distance to center
                variance = np.mean(np.sum((cluster_points - center)**2, axis=1))
                cluster_variances.append(variance)
            else:
                cluster_variances.append(0.0)

        return ClusterMetrics(
            silhouette_score=silhouette,
            calinski_harabasz_score=ch_score,
            davies_bouldin_score=db_score,
            cluster_sizes=cluster_sizes,
            cluster_variances=cluster_variances,
            learned_silhouette_score=learned_silhouette,
            learned_calinski_harabasz_score=learned_ch_score,
            learned_davies_bouldin_score=learned_db_score
        )

    def evaluate_constraint_satisfaction(
        self,
        labels: np.ndarray,
        must_links: List[Tuple[int, int]],
        cannot_links: List[Tuple[int, int]],
        embeddings: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> ConstraintMetrics:
        """
        Evaluate how well constraints are satisfied.

        Args:
            labels: [N] cluster assignments
            must_links: List of (i, j) pairs that should be in same cluster
            cannot_links: List of (i, j) pairs that should be in different clusters
            embeddings: Optional [N, D] or [N, num_heads, D] embeddings for distance-based metrics

        Returns:
            ConstraintMetrics with satisfaction rates
        """
        # Must-link satisfaction
        satisfied_must_links = 0
        for i, j in must_links:
            if labels[i] == labels[j]:
                satisfied_must_links += 1

        must_link_satisfaction = satisfied_must_links / len(must_links) if must_links else 1.0

        # Cannot-link satisfaction
        satisfied_cannot_links = 0
        for i, j in cannot_links:
            if labels[i] != labels[j]:
                satisfied_cannot_links += 1

        cannot_link_satisfaction = satisfied_cannot_links / len(cannot_links) if cannot_links else 1.0

        # Compute learned distance-based metrics if embeddings provided
        avg_must_link_distance = None
        avg_cannot_link_distance = None
        constraint_separation_score = None

        if embeddings is not None:
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.detach().cpu().numpy()

            # Compute average learned distances for must-links and cannot-links
            if must_links:
                must_distances = []
                for i, j in must_links:
                    if embeddings.ndim == 3:
                        # Multi-head: use learned distance (min across heads)
                        head_distances = []
                        for k in range(embeddings.shape[1]):
                            dist = np.linalg.norm(embeddings[i, k] - embeddings[j, k])
                            head_distances.append(dist)
                        learned_dist = min(head_distances)
                    else:
                        learned_dist = np.linalg.norm(embeddings[i] - embeddings[j])
                    must_distances.append(learned_dist)
                avg_must_link_distance = np.mean(must_distances)

            if cannot_links:
                cannot_distances = []
                for i, j in cannot_links:
                    if embeddings.ndim == 3:
                        # Multi-head: use learned distance (min across heads)
                        head_distances = []
                        for k in range(embeddings.shape[1]):
                            dist = np.linalg.norm(embeddings[i, k] - embeddings[j, k])
                            head_distances.append(dist)
                        learned_dist = min(head_distances)
                    else:
                        learned_dist = np.linalg.norm(embeddings[i] - embeddings[j])
                    cannot_distances.append(learned_dist)
                avg_cannot_link_distance = np.mean(cannot_distances)

            # Constraint separation score: how much farther apart cannot-links are vs must-links
            if avg_must_link_distance is not None and avg_cannot_link_distance is not None:
                if avg_must_link_distance > 0:
                    constraint_separation_score = avg_cannot_link_distance / avg_must_link_distance
                else:
                    constraint_separation_score = float('inf') if avg_cannot_link_distance > 0 else 1.0

        return ConstraintMetrics(
            must_link_satisfaction=must_link_satisfaction,
            cannot_link_satisfaction=cannot_link_satisfaction,
            total_must_links=len(must_links),
            total_cannot_links=len(cannot_links),
            satisfied_must_links=satisfied_must_links,
            satisfied_cannot_links=satisfied_cannot_links,
            avg_must_link_distance=avg_must_link_distance,
            avg_cannot_link_distance=avg_cannot_link_distance,
            constraint_separation_score=constraint_separation_score
        )

    def evaluate_stability(
        self,
        current_labels: np.ndarray,
        current_centroids: Optional[np.ndarray] = None
    ) -> StabilityMetrics:
        """
        Evaluate stability compared to previous iteration.

        Args:
            current_labels: Current cluster assignments
            current_centroids: Current cluster centroids

        Returns:
            StabilityMetrics showing changes from previous iteration
        """
        if self.previous_labels is None:
            # First evaluation
            assignment_stability = 1.0
            cluster_purity_stability = None
            centroid_shift = None
        else:
            # Assignment stability (fraction of items with same cluster)
            min_len = min(len(self.previous_labels), len(current_labels))
            assignment_stability = np.mean(
                self.previous_labels[:min_len] == current_labels[:min_len]
            )

            # Cluster purity stability (how well old clusters map to new ones)
            cluster_purity_stability = self._compute_cluster_purity_stability(
                self.previous_labels, current_labels
            )

            # Centroid shift
            if current_centroids is not None and self.previous_centroids is not None:
                centroid_shift = np.mean([
                    np.linalg.norm(c1 - c2)
                    for c1, c2 in zip(self.previous_centroids, current_centroids)
                ])
            else:
                centroid_shift = None

        # Update previous state
        self.previous_labels = current_labels.copy()
        if current_centroids is not None:
            self.previous_centroids = current_centroids.copy()

        return StabilityMetrics(
            assignment_stability=assignment_stability,
            cluster_purity_stability=cluster_purity_stability,
            centroid_shift=centroid_shift
        )

    def evaluate_uncertainty(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        labels: np.ndarray,
        centroids: np.ndarray,
        top_k: int = 10
    ) -> UncertaintyMetrics:
        """
        Evaluate item-level uncertainty and cluster quality issues.

        Args:
            embeddings: [N, D] or [N, num_heads, D] embeddings
            labels: [N] cluster assignments
            centroids: [K, D] or [K, num_heads, D] centroids
            top_k: Number of top uncertain items to return

        Returns:
            UncertaintyMetrics with uncertain items and problematic clusters
        """
        # Handle tensor conversion
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        if embeddings.ndim == 3:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)

        if centroids.ndim == 3:
            centroids = centroids.reshape(centroids.shape[0], -1)

        # Compute item uncertainty scores
        uncertain_items = self._compute_item_scores(embeddings, centroids, labels, top_k)

        # Find boundary pairs (close items in different clusters)
        boundary_pairs = self._find_boundary_pairs(embeddings, labels, centroids)

        # Identify high-variance clusters
        high_variance_clusters = []
        for k in range(len(centroids)):
            mask = (labels == k)
            if np.any(mask):
                cluster_points = embeddings[mask]
                variance = np.mean(np.sum((cluster_points - centroids[k])**2, axis=1))
                high_variance_clusters.append((k, variance))

        high_variance_clusters.sort(key=lambda x: x[1], reverse=True)

        return UncertaintyMetrics(
            uncertain_items=uncertain_items,
            boundary_pairs=boundary_pairs,
            high_variance_clusters=high_variance_clusters
        )

    def create_evaluation_report(
        self,
        engine,
        iteration: Optional[int] = None
    ) -> EvaluationReport:
        """
        Create a comprehensive evaluation report for the current engine state.

        Args:
            engine: ClusterRefinementEngine instance
            iteration: Optional iteration number

        Returns:
            Complete EvaluationReport
        """
        if iteration is None:
            iteration = self.iteration_count
            self.iteration_count += 1

        # Get embeddings and labels
        if engine.z is None or engine.labels is None:
            raise ValueError("Engine has no clustering results to evaluate")

        embeddings = engine.z
        labels = engine.labels
        centroids = engine.clustering.get_centroids()

        # Compute all metrics
        cluster_metrics = self.evaluate_clustering(embeddings, labels, centroids)

        constraint_metrics = self.evaluate_constraint_satisfaction(
            labels,
            list(engine.constraints.must_links),
            list(engine.constraints.cannot_links),
            embeddings
        )

        # Add constraint distance improvements
        constraint_improvements = engine.constraints.get_constraint_improvements(embeddings)

        stability_metrics = self.evaluate_stability(labels, centroids)

        uncertainty_metrics = self.evaluate_uncertainty(embeddings, labels, centroids)

        # Create timestamp
        from datetime import datetime
        timestamp = datetime.now().isoformat()

        return EvaluationReport(
            cluster_metrics=cluster_metrics,
            constraint_metrics=constraint_metrics,
            stability_metrics=stability_metrics,
            uncertainty_metrics=uncertainty_metrics,
            constraint_improvements=constraint_improvements,
            timestamp=timestamp,
            iteration=iteration
        )

    def print_report(self, report: EvaluationReport, verbose: bool = True):
        """
        Print a formatted evaluation report.

        Args:
            report: EvaluationReport to print
            verbose: Whether to show detailed information
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION REPORT - Iteration {report.iteration}")
        print(f"Timestamp: {report.timestamp}")
        print(f"{'='*60}")

        # Cluster Quality Metrics
        print("\n📊 CLUSTER QUALITY METRICS:")
        cm = report.cluster_metrics
        print(f"  Silhouette Score: {cm.silhouette_score}")
        print(f"  Calinski-Harabasz Score: {cm.calinski_harabasz_score}")
        print(f"  Davies-Bouldin Score: {cm.davies_bouldin_score}")

        # Learned Metrics (if available)
        if cm.learned_silhouette_score is not None:
            print("\n🎯 LEARNED METRICS (Multi-Head Distance):")
            print(f"  Learned Silhouette Score: {cm.learned_silhouette_score}")
            print(f"  Learned Calinski-Harabasz Score: {cm.learned_calinski_harabasz_score}")
            print(f"  Learned Davies-Bouldin Score: {cm.learned_davies_bouldin_score}")

        print("\n🏗️  CLUSTER STATISTICS:")
        total_items = sum(cm.cluster_sizes)
        for i, (size, variance) in enumerate(zip(cm.cluster_sizes, cm.cluster_variances)):
            percentage = size / total_items * 100 if total_items > 0 else 0
            print(f"    Cluster {i}: {size} items ({percentage:.1f}%), variance: {variance:.3f}")

        # Constraint Satisfaction
        print("\n🎯 CONSTRAINT SATISFACTION:")
        cs = report.constraint_metrics
        print(f"  Must-links: {cs.satisfied_must_links}/{cs.total_must_links} satisfied ({cs.must_link_satisfaction:.1%})")
        print(f"  Cannot-links: {cs.satisfied_cannot_links}/{cs.total_cannot_links} satisfied ({cs.cannot_link_satisfaction:.1%})")
        print(f"  Total constraints: {cs.total_must_links + cs.total_cannot_links}")

        # Learned distance-based constraint metrics
        if cs.avg_must_link_distance is not None or cs.avg_cannot_link_distance is not None:
            print("\n📏 LEARNED DISTANCE CONSTRAINTS:")
            if cs.avg_must_link_distance is not None:
                print(f"  Avg must-link distance: {cs.avg_must_link_distance:.3f}")
            if cs.avg_cannot_link_distance is not None:
                print(f"  Avg cannot-link distance: {cs.avg_cannot_link_distance:.3f}")
            if cs.constraint_separation_score is not None:
                print(f"  Constraint separation score: {cs.constraint_separation_score:.2f}x")

        # Constraint distance improvements (baseline vs current)
        if report.constraint_improvements and report.constraint_improvements.get("must_links"):
            print("\n📈 CONSTRAINT DISTANCE IMPROVEMENTS:")
            avg_ml_improvement = report.constraint_improvements.get("avg_must_link_improvement", 0.0)
            avg_cl_improvement = report.constraint_improvements.get("avg_cannot_link_improvement", 0.0)

            if abs(avg_ml_improvement) > 0.001:
                print(f"  Must-link distance improvement: {avg_ml_improvement:+.3f} (closer = better)")
            if abs(avg_cl_improvement) > 0.001:
                print(f"  Cannot-link distance improvement: {avg_cl_improvement:+.3f} (farther = better)")

            # Show individual constraint improvements (first few)
            must_links = report.constraint_improvements.get("must_links", [])
            cannot_links = report.constraint_improvements.get("cannot_links", [])

            if must_links:
                print(f"  Top must-link improvements:")
                for i, ml in enumerate(must_links[:3]):
                    print(f"    {ml['pair']}: {ml['baseline_distance']:.3f} → {ml['current_distance']:.3f} ({ml['improvement']:+.3f})")

            if cannot_links:
                print(f"  Top cannot-link improvements:")
                for i, cl in enumerate(cannot_links[:3]):
                    print(f"    {cl['pair']}: {cl['baseline_distance']:.3f} → {cl['current_distance']:.3f} ({cl['improvement']:+.3f})")

        # Stability Metrics
        if report.stability_metrics:
            print("\n🔄 STABILITY METRICS:")
            sm = report.stability_metrics
            print(f"  Assignment stability: {sm.assignment_stability:.1%}")
            if sm.cluster_purity_stability is not None:
                print(f"  Cluster purity stability: {sm.cluster_purity_stability:.3f}")
            if sm.centroid_shift is not None:
                print(f"  Average centroid shift: {sm.centroid_shift:.3f}")

        # Uncertainty Analysis
        if verbose:
            print("\n🎲 UNCERTAINTY ANALYSIS:")
            um = report.uncertainty_metrics

            print(f"  Top uncertain items:")
            for i, (idx, scores) in enumerate(um.uncertain_items[:5]):
                print(f"    {i+1}. Item {idx}: margin={scores['margin']:.3f}, dist={scores['dist_to_centroid']:.3f}")

            print(f"  High-variance clusters:")
            for i, (cluster_id, variance) in enumerate(um.high_variance_clusters[:3]):
                print(f"    {i+1}. Cluster {cluster_id}: variance={variance:.3f}")

            print(f"  Boundary pairs found: {len(um.boundary_pairs)}")

        print(f"\n{'='*60}")

    def compare_reports(self, old_report: EvaluationReport, new_report: EvaluationReport) -> Dict:
        """
        Compare two evaluation reports to show progress.

        Args:
            old_report: Baseline report
            new_report: Updated report

        Returns:
            Dictionary with improvements/changes
        """
        improvements = {}

        # Cluster quality improvements
        for metric in ['silhouette_score', 'calinski_harabasz_score']:
            old_val = old_report.cluster_metrics.__dict__[metric]
            new_val = new_report.cluster_metrics.__dict__[metric]

            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                improvements[metric] = new_val - old_val

        # Davies-Bouldin score (lower is better)
        old_db = old_report.cluster_metrics.davies_bouldin_score
        new_db = new_report.cluster_metrics.davies_bouldin_score
        if isinstance(old_db, (int, float)) and isinstance(new_db, (int, float)):
            improvements['davies_bouldin_improvement'] = old_db - new_db

        # Constraint satisfaction improvements
        improvements['must_link_improvement'] = (
            new_report.constraint_metrics.must_link_satisfaction -
            old_report.constraint_metrics.must_link_satisfaction
        )
        improvements['cannot_link_improvement'] = (
            new_report.constraint_metrics.cannot_link_satisfaction -
            old_report.constraint_metrics.cannot_link_satisfaction
        )

        # Stability
        if old_report.stability_metrics and new_report.stability_metrics:
            improvements['assignment_stability'] = new_report.stability_metrics.assignment_stability

        return improvements

    def _compute_item_scores(
        self,
        embeddings: np.ndarray,
        centroids: np.ndarray,
        labels: np.ndarray,
        top_k: int
    ) -> List[Tuple[int, Dict]]:
        """Compute uncertainty scores for items (similar to engine._compute_item_scores)."""
        centroid_dists = np.linalg.norm(embeddings[:, None, :] - centroids[None, :, :], axis=2)
        nearest = np.argsort(centroid_dists, axis=1)
        top1 = nearest[:, 0]
        top2 = nearest[:, 1]
        indices = np.arange(len(embeddings))

        dist_to_centroid = centroid_dists[indices, top1]
        margin = centroid_dists[indices, top2] - centroid_dists[indices, top1]

        scores = []
        for idx in range(len(embeddings)):
            scores.append((
                idx,
                {
                    "dist_to_centroid": dist_to_centroid[idx],
                    "margin": margin[idx],
                }
            ))

        # Sort by low margin first (uncertain), then by large distance
        scores.sort(key=lambda item: (item[1]["margin"], -item[1]["dist_to_centroid"]))
        return scores[:top_k]

    def _find_boundary_pairs(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
        max_pairs: int = 20
    ) -> List[Tuple[int, int, float]]:
        """Find pairs of points that are close but in different clusters."""
        boundary_pairs = []

        # For efficiency, only check points near cluster boundaries
        uncertain_items = self._compute_item_scores(embeddings, centroids, labels, 50)
        candidate_indices = [idx for idx, _ in uncertain_items]

        for i in candidate_indices:
            for j in candidate_indices:
                if i >= j or labels[i] == labels[j]:
                    continue

                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                boundary_pairs.append((i, j, dist))

        # Sort by distance (closest first)
        boundary_pairs.sort(key=lambda x: x[2])
        return boundary_pairs[:max_pairs]

    def _compute_cluster_purity_stability(
        self,
        old_labels: np.ndarray,
        new_labels: np.ndarray
    ) -> float:
        """Compute how well old clusters map to new clusters."""
        # This is a simplified measure - could be enhanced with Hungarian algorithm
        # for optimal cluster mapping
        old_clusters = set(old_labels)
        new_clusters = set(new_labels)

        # For each old cluster, find best matching new cluster
        total_purity = 0.0
        for old_c in old_clusters:
            old_mask = (old_labels == old_c)
            old_size = np.sum(old_mask)

            if old_size == 0:
                continue

            # Count intersections with new clusters
            max_overlap = 0
            for new_c in new_clusters:
                overlap = np.sum((old_labels == old_c) & (new_labels == new_c))
                max_overlap = max(max_overlap, overlap)

            total_purity += max_overlap / old_size

        return total_purity / len(old_clusters) if old_clusters else 1.0

    def _compute_learned_distance_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distance matrix using learned multi-head distance function.

        Distance between i,j is min over heads of ||embeddings[i,k] - embeddings[j,k]||_2

        Args:
            embeddings: [N, num_heads, D] multi-head embeddings

        Returns:
            distance_matrix: [N, N] distance matrix
        """
        N, num_heads, D = embeddings.shape
        distance_matrix = np.zeros((N, N))

        # For each pair of points
        for i in range(N):
            for j in range(i + 1, N):
                # Compute distance per head
                head_distances = []
                for k in range(num_heads):
                    dist = np.linalg.norm(embeddings[i, k] - embeddings[j, k])
                    head_distances.append(dist)

                # Learned distance is minimum across heads
                learned_dist = min(head_distances)
                distance_matrix[i, j] = learned_dist
                distance_matrix[j, i] = learned_dist

        return distance_matrix

    def _compute_learned_silhouette_score(self, embeddings: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """
        Compute silhouette score using learned multi-head distance function.
        """
        from sklearn.metrics import silhouette_samples

        # Get unique clusters
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            raise ValueError("Need at least 2 clusters for silhouette score")

        distance_matrix = self._compute_learned_distance_matrix(embeddings)
        silhouette_vals = silhouette_samples(distance_matrix, labels, metric="precomputed")
        return np.mean(silhouette_vals)

    def _compute_learned_calinski_harabasz(self, embeddings: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """
        Compute Calinski-Harabasz score using learned multi-head distance function.
        """
        # Get unique clusters
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            raise ValueError("Need at least 2 clusters")

        N, num_heads, D = embeddings.shape
        n_clusters = len(unique_labels)

        # Compute overall centroid using learned distance
        overall_centroid = np.zeros((num_heads, D))
        for k in range(num_heads):
            overall_centroid[k] = np.mean(embeddings[:, k], axis=0)

        # Compute between-cluster dispersion
        between_dispersion = 0.0
        for k in range(num_heads):
            for cluster_id in unique_labels:
                cluster_mask = (labels == cluster_id)
                cluster_size = np.sum(cluster_mask)
                if cluster_size > 0:
                    cluster_centroid = np.mean(embeddings[cluster_mask, k], axis=0)
                    between_dispersion += cluster_size * np.sum((cluster_centroid - overall_centroid[k]) ** 2)

        # Compute within-cluster dispersion
        within_dispersion = 0.0
        for cluster_id in unique_labels:
            cluster_mask = (labels == cluster_id)
            cluster_points = embeddings[cluster_mask]
            cluster_centroid = np.mean(cluster_points, axis=0)

            for point in cluster_points:
                # Use learned distance (min across heads)
                head_distances = []
                for k in range(num_heads):
                    dist = np.sum((point[k] - cluster_centroid[k]) ** 2)
                    head_distances.append(dist)
                learned_dist_sq = min(head_distances)
                within_dispersion += learned_dist_sq

        # Calinski-Harabasz score
        if within_dispersion == 0:
            return float('inf')
        return (between_dispersion / (n_clusters - 1)) / (within_dispersion / (N - n_clusters))

    def _compute_learned_davies_bouldin(self, embeddings: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """
        Compute Davies-Bouldin score using learned multi-head distance function.
        """
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            raise ValueError("Need at least 2 clusters")

        n_clusters = len(unique_labels)

        # Compute centroids for each cluster per head
        cluster_centroids = np.zeros((n_clusters, embeddings.shape[1], embeddings.shape[2]))
        for i, cluster_id in enumerate(unique_labels):
            cluster_mask = (labels == cluster_id)
            for k in range(embeddings.shape[1]):
                cluster_centroids[i, k] = np.mean(embeddings[cluster_mask, k], axis=0)

        # Compute within-cluster dispersion for each cluster
        within_dispersion = np.zeros(n_clusters)
        for i, cluster_id in enumerate(unique_labels):
            cluster_mask = (labels == cluster_id)
            cluster_points = embeddings[cluster_mask]

            total_dispersion = 0.0
            for point in cluster_points:
                # Use learned distance to cluster centroid
                head_distances = []
                for k in range(embeddings.shape[1]):
                    dist = np.sum((point[k] - cluster_centroids[i, k]) ** 2)
                    head_distances.append(dist)
                learned_dist_sq = min(head_distances)
                total_dispersion += learned_dist_sq

            within_dispersion[i] = np.sqrt(total_dispersion / len(cluster_points)) if len(cluster_points) > 0 else 0

        # Compute between-cluster distances
        between_distances = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                # Learned distance between centroids (min across heads)
                head_distances = []
                for k in range(embeddings.shape[1]):
                    dist = np.linalg.norm(cluster_centroids[i, k] - cluster_centroids[j, k])
                    head_distances.append(dist)
                learned_dist = min(head_distances)
                between_distances[i, j] = learned_dist
                between_distances[j, i] = learned_dist

        # Compute Davies-Bouldin index
        cluster_scores = np.zeros(n_clusters)
        for i in range(n_clusters):
            max_ratio = 0.0
            for j in range(n_clusters):
                if i != j:
                    ratio = (within_dispersion[i] + within_dispersion[j]) / between_distances[i, j]
                    max_ratio = max(max_ratio, ratio)
            cluster_scores[i] = max_ratio

        return np.mean(cluster_scores)


# Convenience functions for quick evaluation
def evaluate_engine(engine, iteration: int = 0) -> EvaluationReport:
    """
    Quick evaluation of a ClusterRefinementEngine.

    Args:
        engine: Engine instance to evaluate
        iteration: Optional iteration number

    Returns:
        Complete evaluation report
    """
    evaluator = ClusterEvaluator()
    return evaluator.create_evaluation_report(engine, iteration)


def print_evaluation_report(engine, iteration: int = 0, verbose: bool = True):
    """
    Print a formatted evaluation report for an engine.

    Args:
        engine: Engine instance to evaluate
        iteration: Optional iteration number
        verbose: Whether to show detailed information
    """
    evaluator = ClusterEvaluator()
    report = evaluator.create_evaluation_report(engine, iteration)
    evaluator.print_report(report, verbose)
    return report
