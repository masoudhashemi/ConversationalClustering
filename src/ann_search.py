"""
Approximate Nearest Neighbor Search for Large-Scale Active Learning

This module provides efficient nearest neighbor search capabilities using FAISS,
enabling the active learning system to scale from O(N²) to O(N log N) complexity.
"""

import numpy as np
import faiss
from typing import List, Tuple, Optional, Dict, Any
import psutil
import os
from dataclasses import dataclass


@dataclass
class ANNConfig:
    """Configuration for approximate nearest neighbor search."""
    index_type: str = "IVF"  # IVF, HNSW, or Flat
    nlist: int = 100  # Number of clusters for IVF
    nprobe: int = 10  # Number of clusters to search for IVF
    ef_construction: int = 200  # HNSW construction parameter
    ef_search: int = 64  # HNSW search parameter
    use_gpu: bool = False  # Enable GPU acceleration if available
    quantize_embeddings: bool = True  # Use PQ quantization
    m_pq: int = 8  # Number of sub-quantizers for PQ
    nbits_pq: int = 8  # Bits per sub-quantizer


class ApproximateNeighborSearch:
    """
    Efficient approximate nearest neighbor search using FAISS.

    Supports multiple index types:
    - IVF (Inverted File): Good balance of speed/accuracy for large datasets
    - HNSW (Hierarchical Navigable Small World): Best accuracy, slower
    - Flat: Exact search, O(N) per query
    """

    def __init__(self, config: ANNConfig = None):
        self.config = config or ANNConfig()
        self.index = None
        self.dimension = None
        self.is_trained = False
        self.embeddings = None  # Keep reference to original embeddings
        self.quantizer = None

        # Memory monitoring
        self.memory_usage = []

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def build_index(self, embeddings: np.ndarray) -> None:
        """
        Build the FAISS index for efficient nearest neighbor search.

        Args:
            embeddings: [N, D] array of embeddings
        """
        # Record memory before building
        start_memory = self._get_memory_usage()

        self.embeddings = embeddings.copy()
        N, D = embeddings.shape
        self.dimension = D

        # Ensure float32 for FAISS
        if self.embeddings.dtype != np.float32:
            self.embeddings = self.embeddings.astype(np.float32)

        print(f"Building index for {N} embeddings of dimension {D}...")
        # Choose index type based on dataset size and requirements
        if N < 10000:
            # For small datasets, exact search is fine
            self.index = faiss.IndexFlatIP(D)  # Inner product (cosine similarity)
        elif self.config.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(D, self.config.ef_construction)
            self.index.hnsw.efSearch = self.config.ef_search
        elif self.config.index_type == "IVF":
            # IVF with PQ for memory efficiency
            if self.config.quantize_embeddings:
                # Use PQ quantization to reduce memory usage
                self.quantizer = faiss.IndexFlatIP(D)
                self.index = faiss.IndexIVFPQ(
                    self.quantizer, D,
                    self.config.nlist,
                    self.config.m_pq,
                    self.config.nbits_pq
                )
            else:
                # Standard IVF
                self.quantizer = faiss.IndexFlatIP(D)
                self.index = faiss.IndexIVFFlat(
                    self.quantizer, D, self.config.nlist
                )
        else:
            # Default to IVF
            self.quantizer = faiss.IndexFlatIP(D)
            self.index = faiss.IndexIVFFlat(self.quantizer, D, self.config.nlist)

        # Normalize embeddings for cosine similarity
        if self.embeddings.flags.c_contiguous:
            faiss.normalize_L2(self.embeddings)
        else:
            self.embeddings = np.ascontiguousarray(self.embeddings)
            faiss.normalize_L2(self.embeddings)

        # Train the index if needed
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            print(f"Training {self.config.index_type} index...")
            self.index.train(self.embeddings)
            self.is_trained = True

        # Add vectors to index
        print(f"Adding {N} vectors to index...")
        self.index.add(self.embeddings)

        # Monitor memory usage (calculate delta to estimate index size)
        final_memory = self._get_memory_usage()
        index_size_estimate = max(0.0, final_memory - start_memory)
        self.memory_usage.append(index_size_estimate)
        
        print(f"Index built successfully. Estimated index size: {index_size_estimate:.1f} MB")
    def search(self, query_embeddings: np.ndarray, k: int = 10,
               return_distances: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.

        Args:
            query_embeddings: [M, D] query vectors
            k: Number of nearest neighbors to find
            return_distances: Whether to return distances

        Returns:
            indices: [M, K] array of nearest neighbor indices
            distances: [M, K] array of distances (if return_distances=True)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Normalize query embeddings
        query_norm = query_embeddings.copy()
        if query_norm.dtype != np.float32:
            query_norm = query_norm.astype(np.float32)
        if not query_norm.flags.c_contiguous:
            query_norm = np.ascontiguousarray(query_norm)
        faiss.normalize_L2(query_norm)

        # Set search parameters for IVF
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.config.nprobe

        # Perform search
        if return_distances:
            distances, indices = self.index.search(query_norm, k)
            return indices, distances
        else:
            _, indices = self.index.search(query_norm, k)
            return indices

    def find_boundary_pairs(self, labels: np.ndarray, uncertain_items: List[int],
                          max_pairs: int = 10) -> List[Tuple[int, int, float]]:
        """
        Find cross-cluster pairs that are close in embedding space.

        This is the key optimization for active learning - instead of computing
        all pairwise distances (O(N²)), we use ANN to find nearest neighbors
        efficiently.

        Args:
            labels: [N] cluster assignments
            uncertain_items: List of item indices to consider as seeds
            max_pairs: Maximum number of boundary pairs to return

        Returns:
            List of (item_i, item_j, distance) tuples for boundary pairs
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not available. Call build_index() first.")

        boundary_pairs = []
        processed_pairs = set()

        # For each uncertain item, find its nearest neighbors
        for idx in uncertain_items[:max_pairs * 2]:  # Limit to avoid too many queries
            # Find nearest neighbors (excluding self)
            query_emb = self.embeddings[idx:idx+1]  # [1, D]
            k_search = min(20, self.embeddings.shape[0])
            if k_search <= 1:
                continue
            indices, distances = self.search(query_emb, k=k_search)

            nn_indices = indices[0][1:11]  # Skip self (index 0), take next 10
            nn_scores = distances[0][1:11]

            current_label = labels[idx]

            # Check which neighbors are in different clusters
            for nn_idx, score in zip(nn_indices, nn_scores):
                if nn_idx < 0:
                    # FAISS may return -1 when not enough neighbors are available.
                    continue
                if labels[nn_idx] != current_label:
                    # Found a boundary pair
                    pair = tuple(sorted([int(idx), int(nn_idx)]))
                    if pair not in processed_pairs:
                        # FAISS IP returns similarity (higher is closer).
                        # Convert to cosine distance-like value so lower means closer.
                        distance = 1.0 - float(score)
                        boundary_pairs.append((int(idx), int(nn_idx), distance))
                        processed_pairs.add(pair)

                        if len(boundary_pairs) >= max_pairs:
                            break

            if len(boundary_pairs) >= max_pairs:
                break

        # Sort by distance (closest first)
        boundary_pairs.sort(key=lambda x: x[2])
        return boundary_pairs

    def find_high_variance_clusters(self, labels: np.ndarray, centroids: np.ndarray,
                                  variance_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Identify clusters with high internal variance using efficient sampling.

        Args:
            labels: [N] cluster assignments
            centroids: [K, D] cluster centroids
            variance_threshold: Quantile threshold for "high variance"

        Returns:
            List of cluster info dictionaries with variance metrics
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not available. Call build_index() first.")

        cluster_stats = []
        unique_labels = np.unique(labels)

        for cluster_id in unique_labels:
            mask = labels == cluster_id
            cluster_points = self.embeddings[mask]

            if len(cluster_points) < 2:
                continue

            # Compute variance efficiently
            centroid = centroids[cluster_id]
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            variance = np.mean(distances ** 2)

            cluster_stats.append({
                'id': int(cluster_id),
                'size': int(np.sum(mask)),
                'variance': float(variance),
                'centroid': centroid
            })

        if not cluster_stats:
            return []

        # Find high variance clusters
        variances = [s['variance'] for s in cluster_stats]
        threshold = np.quantile(variances, variance_threshold)

        high_variance_clusters = [
            stat for stat in cluster_stats
            if stat['variance'] >= threshold
        ]

        return high_variance_clusters

    def update_index(self, new_embeddings: np.ndarray, indices_to_update: Optional[List[int]] = None) -> None:
        """
        Update the index with new/modified embeddings.

        Args:
            new_embeddings: [M, D] updated embeddings
            indices_to_update: Which indices to update (if None, append new vectors)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        if indices_to_update is None:
            # Append new vectors
            faiss.normalize_L2(new_embeddings)
            self.index.add(new_embeddings)
            # Update stored embeddings
            if self.embeddings is not None:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
        else:
            # For updating existing vectors, we need to rebuild for most FAISS indices
            # This is a limitation - IVF indices don't support efficient updates
            print("Warning: Updating existing vectors requires index rebuild")
            all_embeddings = self.embeddings.copy()
            for idx, new_emb in zip(indices_to_update, new_embeddings):
                all_embeddings[idx] = new_emb
            self.build_index(all_embeddings)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            'current_memory_mb': self._get_memory_usage(),
            'index_memory_history': self.memory_usage,
            'index_type': self.config.index_type,
            'dimension': self.dimension,
            'quantized': self.config.quantize_embeddings
        }
