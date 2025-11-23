import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import Tuple, Dict, List

import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress numpy warnings globally for this module
# These warnings occur in sklearn's internal matrix operations with high-dimensional data
np.seterr(divide='ignore', invalid='ignore', over='ignore')

class ClusteringModule:
    """
    Manages clustering operations (K-Means) and cluster diagnostics.
    """
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        # Use MiniBatchKMeans which is more numerically stable and faster
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            n_init=10,
            init='k-means++',
            max_iter=100,
            batch_size=1024,   # Large batch size for stability
            tol=1e-6,
            random_state=42
        )
        self.centroids = None
        self.labels = None
        
    def fit_predict(self, z: np.ndarray) -> np.ndarray:
        """
        Runs K-Means on latent vectors z.
        z: [N, num_heads, p] - multi-head embeddings
        Returns: labels [N]
        """
        # Suppress all numpy warnings during clustering
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore")
            
            if z.ndim == 3:
                # Multi-head: flatten to [N, num_heads * p]
                N, num_heads, p = z.shape
                z = z.reshape(N, num_heads * p)

            # Check for numerical issues in input
            if not np.isfinite(z).all():
                print("Warning: Non-finite values in clustering input, replacing with zeros")
                z = np.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)

            # Ensure reasonable scale - normalize if not already normalized
            z_norm = np.linalg.norm(z, axis=1, keepdims=True)
            if np.any(z_norm == 0):
                print("Warning: Zero-norm vectors in clustering input, adding small noise")
                z += np.random.normal(0, 1e-6, z.shape)
                z_norm = np.linalg.norm(z, axis=1, keepdims=True)  # Recalculate
            elif not np.allclose(z_norm, 1.0, atol=1e-6):
                # Re-normalize if not unit norm (add epsilon to prevent division by zero)
                z = z / (z_norm + 1e-10)

            # Clip values to prevent extreme numbers
            z = np.clip(z, -10.0, 10.0)

            # Add small regularization noise to prevent numerical issues in sklearn
            # This helps with the matrix multiplication operations in K-means
            regularization_noise = np.random.normal(0, 1e-8, z.shape)
            z = z + regularization_noise

            # Re-normalize after adding noise (add epsilon to prevent division by zero)
            z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-10)

            # Convert to float64 for numerical stability in sklearn
            z = z.astype(np.float64)

            # Run k-means (warnings already suppressed by outer context)
            self.labels = self.kmeans.fit_predict(z)
        
        self.centroids = self.kmeans.cluster_centers_

        # Ensure centroids are also normalized (add epsilon to prevent division by zero)
        centroid_norms = np.linalg.norm(self.centroids, axis=1, keepdims=True)
        if not np.allclose(centroid_norms, 1.0, atol=1e-6):
            self.centroids = self.centroids / (centroid_norms + 1e-10)

        return self.labels
    
    def get_centroids(self) -> np.ndarray:
        return self.centroids
    
    def get_cluster_stats(self, z: np.ndarray, labels: np.ndarray) -> List[Dict]:
        """
        Compute basic stats per cluster: size, variance, etc.
        """
        stats = []
        for k in range(self.n_clusters):
            mask = (labels == k)
            if not np.any(mask):
                stats.append({"id": k, "size": 0, "variance": 0.0})
                continue
                
            cluster_z = z[mask]
            center = self.centroids[k]
            # Mean squared distance to center
            variance = np.mean(np.sum((cluster_z - center)**2, axis=1))
            stats.append({
                "id": k, 
                "size": int(np.sum(mask)), 
                "variance": float(variance)
            })
        return stats

