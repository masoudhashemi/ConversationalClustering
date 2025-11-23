"""
Embedding Quantization for Memory-Efficient Large-Scale Clustering

This module provides quantization techniques to reduce memory footprint of embeddings
while maintaining clustering quality. Supports multiple quantization methods:

- Product Quantization (PQ)
- Scalar Quantization
- Binary Quantization
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
import pickle


class BaseQuantizer(ABC):
    """Abstract base class for embedding quantizers."""

    @abstractmethod
    def fit(self, embeddings: np.ndarray) -> None:
        """Fit the quantizer to the data."""
        pass

    @abstractmethod
    def quantize(self, embeddings: np.ndarray) -> np.ndarray:
        """Quantize embeddings."""
        pass

    @abstractmethod
    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        """Dequantize to original space."""
        pass

    @abstractmethod
    def get_memory_savings(self) -> float:
        """Return memory savings ratio (original_size / quantized_size)."""
        pass


class ProductQuantizer(BaseQuantizer):
    """
    Product Quantization for memory-efficient embedding storage.

    Decomposes high-dimensional vectors into low-dimensional subspaces
    and quantizes each subspace separately.
    """

    def __init__(self, n_subquantizers: int = 8, n_bits: int = 8, normalize: bool = True):
        """
        Args:
            n_subquantizers: Number of subspaces (M)
            n_bits: Bits per sub-quantizer (typically 8)
            normalize: Whether to normalize vectors before quantization
        """
        self.n_subquantizers = n_subquantizers
        self.n_bits = n_bits
        self.normalize = normalize
        self.n_centroids = 2 ** n_bits

        # Learned parameters
        self.centroids = None  # [M, 2^nbits, D/M]
        self.dimension = None
        self.subspace_dim = None

        # Memory tracking
        self.original_memory_mb = 0
        self.quantized_memory_mb = 0

    def fit(self, embeddings: np.ndarray) -> None:
        """
        Fit PQ centroids to the training data.

        Args:
            embeddings: [N, D] training embeddings
        """
        N, D = embeddings.shape
        self.dimension = D
        # Ensure the dimension is divisible by the number of subquantizers
        assert D % self.n_subquantizers == 0, (
            f"Embedding dimension {D} must be divisible by n_subquantizers="
            f"{self.n_subquantizers} for ProductQuantizer."
        )
        self.subspace_dim = D // self.n_subquantizers

        if self.normalize:
            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings = embeddings / norms

        self.original_memory_mb = embeddings.nbytes / 1024 / 1024

        # Split into subspaces
        subspaces = []
        for m in range(self.n_subquantizers):
            start_dim = m * self.subspace_dim
            end_dim = (m + 1) * self.subspace_dim
            subspace = embeddings[:, start_dim:end_dim]
            subspaces.append(subspace)

        # Learn centroids for each subspace
        self.centroids = np.zeros((self.n_subquantizers, self.n_centroids, self.subspace_dim))

        for m in range(self.n_subquantizers):
            subspace_data = subspaces[m]

            # Safety check for small datasets
            effective_n_clusters = self.n_centroids
            if subspace_data.shape[0] < self.n_centroids:
                print(f"Warning: Not enough data for {self.n_centroids} clusters in quantization. "
                      f"Reducing to {subspace_data.shape[0]} clusters.")
                effective_n_clusters = subspace_data.shape[0]

            # Use k-means to find centroids
            from sklearn.cluster import MiniBatchKMeans
            import warnings

            # Suppress sklearn warnings for small datasets - they're benign
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
                kmeans = MiniBatchKMeans(
                    n_clusters=effective_n_clusters,
                    batch_size=1024,
                    random_state=42,
                    n_init=3
                )
                kmeans.fit(subspace_data)
            
            # If we reduced clusters, pad the rest with zeros or duplicate
            if effective_n_clusters < self.n_centroids:
                # Pad with last centroid
                padding = np.tile(kmeans.cluster_centers_[-1:], (self.n_centroids - effective_n_clusters, 1))
                centers = np.vstack([kmeans.cluster_centers_, padding])
                self.centroids[m] = centers
            else:
                self.centroids[m] = kmeans.cluster_centers_

        print(f"Trained PQ quantizer: {self.n_subquantizers} subspaces, "
              f"{self.n_centroids} centroids each, {self.subspace_dim}D subspaces")

    def quantize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Quantize embeddings using learned centroids.

        Args:
            embeddings: [N, D] embeddings to quantize

        Returns:
            codes: [N, M] array of centroid indices
        """
        if self.centroids is None:
            raise ValueError("Quantizer not fitted. Call fit() first.")

        N, D = embeddings.shape
        if D != self.dimension:
            raise ValueError(f"Embedding dimension {D} doesn't match fitted dimension {self.dimension}")

        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms

        # Quantize each subspace
        codes = np.zeros((N, self.n_subquantizers), dtype=np.uint8)

        for m in range(self.n_subquantizers):
            start_dim = m * self.subspace_dim
            end_dim = (m + 1) * self.subspace_dim
            subspace = embeddings[:, start_dim:end_dim]

            # Find nearest centroid for each vector
            centroids_m = self.centroids[m]  # [2^nbits, D/M]

            # Compute distances to all centroids
            distances = np.linalg.norm(
                subspace[:, None, :] - centroids_m[None, :, :], axis=2
            )  # [N, 2^nbits]

            codes[:, m] = np.argmin(distances, axis=1).astype(np.uint8)

        self.quantized_memory_mb = codes.nbytes / 1024 / 1024
        return codes

    def dequantize(self, codes: np.ndarray) -> np.ndarray:
        """
        Reconstruct embeddings from quantization codes.

        Args:
            codes: [N, M] quantization codes

        Returns:
            reconstructed: [N, D] reconstructed embeddings
        """
        if self.centroids is None:
            raise ValueError("Quantizer not fitted. Call fit() first.")

        N = codes.shape[0]
        reconstructed = np.zeros((N, self.dimension), dtype=np.float32)

        for m in range(self.n_subquantizers):
            start_dim = m * self.subspace_dim
            end_dim = (m + 1) * self.subspace_dim

            # Get centroids for this subspace
            centroid_indices = codes[:, m]
            centroids_m = self.centroids[m]  # [2^nbits, D/M]

            # Reconstruct subspace
            reconstructed[:, start_dim:end_dim] = centroids_m[centroid_indices]

        return reconstructed

    def get_memory_savings(self) -> float:
        """Return memory savings ratio."""
        if self.original_memory_mb == 0:
            return 1.0
        return self.original_memory_mb / self.quantized_memory_mb


class ScalarQuantizer(BaseQuantizer):
    """
    Simple scalar quantization using uniform quantization.
    """

    def __init__(self, n_bits: int = 8, normalize: bool = True):
        self.n_bits = n_bits
        self.normalize = normalize
        self.n_levels = 2 ** n_bits
        self.min_val = None
        self.max_val = None

    def fit(self, embeddings: np.ndarray) -> None:
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms

        self.min_val = np.min(embeddings)
        self.max_val = np.max(embeddings)
        self.original_memory_mb = embeddings.nbytes / 1024 / 1024

    def quantize(self, embeddings: np.ndarray) -> np.ndarray:
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms

        # Scale to [0, 2^nbits - 1]
        scaled = (embeddings - self.min_val) / (self.max_val - self.min_val)
        scaled = np.clip(scaled, 0, 1)
        quantized = (scaled * (self.n_levels - 1)).astype(np.uint8)

        self.quantized_memory_mb = quantized.nbytes / 1024 / 1024
        return quantized

    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        # Scale back to original range
        scaled = quantized.astype(np.float32) / (self.n_levels - 1)
        return scaled * (self.max_val - self.min_val) + self.min_val

    def get_memory_savings(self) -> float:
        if self.original_memory_mb == 0:
            return 1.0
        return self.original_memory_mb / self.quantized_memory_mb


class BinaryQuantizer(BaseQuantizer):
    """
    Binary quantization - extreme compression at cost of precision.
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.mean_vector = None

    def fit(self, embeddings: np.ndarray) -> None:
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms

        self.mean_vector = np.mean(embeddings, axis=0)
        self.original_memory_mb = embeddings.nbytes / 1024 / 1024

    def quantize(self, embeddings: np.ndarray) -> np.ndarray:
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms

        # Center and binarize
        centered = embeddings - self.mean_vector
        binary = (centered > 0).astype(np.uint8)

        self.quantized_memory_mb = binary.nbytes / 1024 / 1024
        return binary

    def dequantize(self, binary: np.ndarray) -> np.ndarray:
        # Convert back to approximate real values
        return (binary.astype(np.float32) - 0.5) * 2 + self.mean_vector

    def get_memory_savings(self) -> float:
        if self.original_memory_mb == 0:
            return 1.0
        return self.original_memory_mb / self.quantized_memory_mb


class QuantizedEmbeddingStorage:
    """
    Manages quantized embeddings with automatic compression/decompression.
    """

    def __init__(self, quantizer_type: str = "pq", **quantizer_kwargs):
        self.quantizer_type = quantizer_type

        if quantizer_type == "pq":
            self.quantizer = ProductQuantizer(**quantizer_kwargs)
        elif quantizer_type == "scalar":
            self.quantizer = ScalarQuantizer(**quantizer_kwargs)
        elif quantizer_type == "binary":
            self.quantizer = BinaryQuantizer(**quantizer_kwargs)
        else:
            raise ValueError(f"Unknown quantizer type: {quantizer_type}")

        self.quantized_data = None
        self.original_shape = None
        self.is_fitted = False

    def compress(self, embeddings: np.ndarray) -> None:
        """
        Compress and store embeddings.

        Args:
            embeddings: [N, D] embeddings to compress
        """
        print(f"Compressing {embeddings.shape[0]} embeddings using {self.quantizer_type} quantization...")

        if not self.is_fitted:
            self.quantizer.fit(embeddings)
            self.is_fitted = True

        self.quantized_data = self.quantizer.quantize(embeddings)
        self.original_shape = embeddings.shape

        savings = self.quantizer.get_memory_savings()
        print(f"Compressed {embeddings.shape[0]} embeddings with {savings:.1f}x memory savings")
    def decompress(self) -> np.ndarray:
        """
        Decompress stored embeddings.

        Returns:
            reconstructed: [N, D] reconstructed embeddings
        """
        if self.quantized_data is None:
            raise ValueError("No quantized data available. Call compress() first.")

        return self.quantizer.dequantize(self.quantized_data)

    def get_compressed_size(self) -> int:
        """Get size of compressed data in bytes."""
        if self.quantized_data is None:
            return 0
        return self.quantized_data.nbytes

    def save(self, filepath: str) -> None:
        """Save quantized embeddings to disk."""
        data = {
            'quantizer_type': self.quantizer_type,
            'quantizer_state': pickle.dumps(self.quantizer),
            'quantized_data': self.quantized_data,
            'original_shape': self.original_shape,
            'is_fitted': self.is_fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> 'QuantizedEmbeddingStorage':
        """Load quantized embeddings from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        instance = cls(data['quantizer_type'])
        instance.quantizer = pickle.loads(data['quantizer_state'])
        instance.quantized_data = data['quantized_data']
        instance.original_shape = data['original_shape']
        instance.is_fitted = data['is_fitted']

        return instance
