import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Iterator, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import psutil
import os
from tqdm import tqdm

class TextEncoder:
    """
    Wrapper for the frozen LLM encoder using SentenceBERT with batch processing support.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu",
                 batch_size: int = 32, max_memory_gb: float = 8.0):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_memory_gb = max_memory_gb

        print(f"Loading SentenceTransformer model: {model_name}...")
        self.model = SentenceTransformer(model_name, device=device)

        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Memory monitoring
        self.memory_usage_history = []

    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024 / 1024

    def _batches_generator(self, texts: List[str], batch_size: int) -> Iterator[Tuple[List[str], int, int]]:
        """Generate batches of texts with indices."""
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            start_idx = i
            end_idx = min(i + batch_size, len(texts))
            yield batch_texts, start_idx, end_idx

    def encode_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode texts in batches to manage memory usage.

        Args:
            texts: List of texts to encode
            show_progress: Whether to show progress bar

        Returns:
            embeddings: [N, D] array of embeddings
        """
        n_texts = len(texts)
        embeddings = np.zeros((n_texts, self.embedding_dim), dtype=np.float32)

        print(f"Encoding {n_texts} texts in batches of {self.batch_size}...")

        # Create progress bar
        batch_iter = self._batches_generator(texts, self.batch_size)
        if show_progress:
            batch_iter = tqdm(batch_iter, total=(n_texts + self.batch_size - 1) // self.batch_size,
                             desc="Encoding batches")

        for batch_texts, start_idx, end_idx in batch_iter:
            # Encode batch
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False
            ).astype(np.float32)

            # Store in result array
            embeddings[start_idx:end_idx] = batch_embeddings

            # Monitor memory usage
            current_memory = self._get_memory_usage()
            self.memory_usage_history.append(current_memory)

            # Check memory limits
            if current_memory > self.max_memory_gb:
                print(f"Warning: Memory usage ({current_memory:.2f}GB) approaching limit ({self.max_memory_gb}GB)")
                # Force garbage collection if needed
                import gc
                gc.collect()

        return self._postprocess_embeddings(embeddings)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts with automatic batching for large datasets.

        For small datasets (< 1000), uses single batch for efficiency.
        For large datasets, uses configured batch size.
        """
        n_texts = len(texts)

        # Decide whether to use batching
        if n_texts <= 1000:
            # Small dataset - encode all at once
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            embeddings = embeddings.astype(np.float32)
        else:
            # Large dataset - use batch processing
            embeddings = self.encode_batch(texts, show_progress=True)

        return self._postprocess_embeddings(embeddings)

    def _postprocess_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Post-process embeddings to handle numerical issues and normalization.
        """
        # Check for numerical issues
        if np.any(np.isnan(embeddings)):
            print("Warning: NaN values found in embeddings, replacing with zeros")
            embeddings = np.nan_to_num(embeddings, nan=0.0)

        if np.any(np.isinf(embeddings)):
            print("Warning: Inf values found in embeddings, replacing with large finite values")
            embeddings = np.clip(embeddings, -1e6, 1e6)

        # Ensure reasonable scale
        embedding_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(embedding_norm == 0):
            print("Warning: Zero-norm embeddings found, adding small noise")
            embeddings += np.random.normal(0, 1e-6, embeddings.shape)

        return embeddings

    def get_memory_stats(self) -> dict:
        """Get memory usage statistics."""
        return {
            'current_memory_gb': self._get_memory_usage(),
            'peak_memory_gb': max(self.memory_usage_history) if self.memory_usage_history else 0,
            'memory_history': self.memory_usage_history.copy(),
            'batch_size': self.batch_size,
            'max_memory_limit_gb': self.max_memory_gb
        }

class EmbeddingAdapter(nn.Module):
    """
    A multi-head adapter to transform base embeddings (h_i) into multiple clustering spaces (z^{(k)}_i for k=1 to num_heads).

    Supports multiple aspects/views of similarity (inspired by UM²L multi-metric learning).
    Each head learns a different projection, allowing disentangled representations.

    If non_linear=True, each head becomes a small MLP.
    """
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 1, non_linear: bool = False, hidden_dim: int = 512):
        super().__init__()
        self.num_heads = num_heads
        self.non_linear = non_linear
        self.output_dim = output_dim

        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            if non_linear:
                head = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
                # Initialize with smaller weights to prevent numerical issues
                nn.init.xavier_uniform_(head[0].weight, gain=0.1)
                nn.init.xavier_uniform_(head[2].weight, gain=0.1)
                nn.init.zeros_(head[0].bias)
                nn.init.zeros_(head[2].bias)
            else:
                # Minimalist: Just a linear map z = W h + b
                head = nn.Linear(input_dim, output_dim)
                # Initialize with identity-like transformation initially (scaled down)
                nn.init.normal_(head.weight, mean=0.0, std=0.1)
                nn.init.zeros_(head.bias)
            self.heads.append(head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns: [batch_size, num_heads, output_dim]
        """
        z_list = []
        for head in self.heads:
            z = head(x)
            # Always normalize for cosine-similarity-like behavior in Euclidean space
            z = F.normalize(z, p=2, dim=1)

            # Additional numerical stability: ensure no zero vectors
            z_norm = torch.norm(z, dim=1, keepdim=True)
            # If any vectors are zero (which shouldn't happen with normalization), add small noise
            if torch.any(z_norm == 0):
                z = z + torch.randn_like(z) * 1e-6
            z_list.append(z)

        # Stack into [batch_size, num_heads, output_dim]
        z = torch.stack(z_list, dim=1)
        return z
