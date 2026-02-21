import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional

class FeedbackLoss(nn.Module):
    """
    Computes loss terms for unsupervised clustering structure and user feedback.
    """
    def __init__(
        self, 
        margin_ml: float = 0.2, 
        margin_cl: float = 1.0,
        weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.margin_ml = margin_ml
        self.margin_cl = margin_cl
        
        # Default weights if not provided
        self.weights = {
            "must_link": 2.0,      # High priority
            "cannot_link": 2.0,    # High priority
            "miscluster": 1.5,     # Medium-high priority
            "concept": 1.0,        # Standard priority
            "cluster_split": 1.5,  # Medium-high priority
            "keyword": 1.0         # Standard priority
        }
        if weights:
            self.weights.update(weights)

    def forward(
        self,
        z: torch.Tensor,
        centroids: torch.Tensor,
        assignments: torch.Tensor,
        must_links: List[Tuple[int, int]],
        cannot_links: List[Tuple[int, int]],
        label_embeddings: Optional[Dict[int, torch.Tensor]] = None,
        miscluster_flags: Optional[List[int]] = None,
        cluster_splits: Optional[List[int]] = None,
        emphasized_keywords: Optional[List[str]] = None,
        keyword_embeddings: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z: [B, num_heads, p] - embeddings for the batch (multi-head)
        centroids: [K, num_heads, p] - current cluster centroids per head
        assignments: [B] - current hard cluster assignments for batch (indices)
        must_links: list of (local_idx_i, local_idx_j) relevant to this batch
        cannot_links: list of (local_idx_i, local_idx_j) relevant to this batch
        label_embeddings: dict mapping cluster_id -> embedding tensor [num_heads, p]
        miscluster_flags: list of item indices marked as misclustered
        cluster_splits: list of cluster_ids marked for splitting
        emphasized_keywords: list of keyword strings to emphasize
        keyword_embeddings: dict mapping keyword -> embedding tensor [num_heads, p]

        Returns: (loss_cluster, loss_feedback)
        """
        B, num_heads, p = z.shape
        K = centroids.shape[0]

        # 1. Unsupervised Cluster Loss: Average across heads
        # L_cluster = mean over heads of mean( ||z_i^{(k)} - mu_{c_i}^{(k)}||^2 )

        loss_cluster_per_head = []
        for k in range(num_heads):
            z_k = z[:, k, :]  # [B, p]
            centroids_k = centroids[:, k, :]  # [K, p]
            # Gather the centroid corresponding to each item's assignment
            batch_centroids_k = centroids_k[assignments]  # [B, p]
            # Squared Euclidean distance
            dist = torch.sum((z_k - batch_centroids_k)**2, dim=1)
            loss_cluster_per_head.append(torch.mean(dist))
        loss_cluster = torch.mean(torch.stack(loss_cluster_per_head))

        # 2. Feedback Loss (Metric Learning) - Compute per head, aggregate with min (UM²L style: at least one head satisfies)
        loss_ml_per_head = []
        loss_cl_per_head = []
        loss_concept_per_head = [torch.tensor(0.0, device=z.device) for _ in range(num_heads)]
        
        # Must-links: minimize distance (with margin) per head, aggregate with min
        if must_links:
            ml_i = [p[0] for p in must_links]
            ml_j = [p[1] for p in must_links]

            for k in range(num_heads):
                z_k = z[:, k, :]  # [B, p]
                z_i_k = z_k[ml_i]
                z_j_k = z_k[ml_j]

                d_ml = torch.sum((z_i_k - z_j_k)**2, dim=1)
                loss_ml_k = torch.mean(torch.clamp(d_ml - self.margin_ml, min=0))
                loss_ml_per_head.append(loss_ml_k)

        # Cannot-links: maximize distance (up to margin) per head, aggregate with min
        if cannot_links:
            cl_i = [p[0] for p in cannot_links]
            cl_j = [p[1] for p in cannot_links]

            for k in range(num_heads):
                z_k = z[:, k, :]
                z_i_k = z_k[cl_i]
                z_j_k = z_k[cl_j]

                d_cl = torch.sum((z_i_k - z_j_k)**2, dim=1)
                loss_cl_k = torch.mean(torch.clamp(self.margin_cl - d_cl, min=0))
                loss_cl_per_head.append(loss_cl_k)

        # Concept Loss: Pull centroids towards label embeddings per head
        if label_embeddings:
            for cluster_id, label_emb in label_embeddings.items():  # label_emb: [num_heads, p]
                if label_emb.device != z.device:
                    label_emb = label_emb.to(z.device)

                # Ensure label_emb has the right shape [num_heads, p]
                if label_emb.dim() == 1:
                    # If it's a 1D tensor, assume it's for single head and expand
                    label_emb = label_emb.unsqueeze(0).expand(num_heads, -1)

                if cluster_id < K and label_emb.shape[0] >= num_heads:
                    for k in range(num_heads):
                        mu_k_head = centroids[cluster_id, k, :]  # [p]
                        label_emb_k = label_emb[k, :]  # [p]
                        loss_concept_per_head[k] += torch.sum((mu_k_head - label_emb_k)**2)

            # Average over clusters if any
            if label_embeddings:
                for k in range(num_heads):
                    loss_concept_per_head[k] /= max(1, len(label_embeddings))
        else:
            loss_concept_per_head = [torch.tensor(0.0, device=z.device)] * num_heads

        # Miscluster Loss: Push misclustered items away from their current cluster centroids
        loss_miscluster_per_head = []
        if miscluster_flags:
            miscluster_indices = miscluster_flags
            for k in range(num_heads):
                z_k = z[:, k, :]  # [B, p]
                centroids_k = centroids[:, k, :]  # [K, p]
                batch_centroids_k = centroids_k[assignments]  # [B, p]

                # Only compute for misclustered items in this batch
                batch_miscluster_mask = torch.tensor([i in miscluster_indices for i in range(B)], device=z.device)
                if batch_miscluster_mask.any():
                    miscluster_z = z_k[batch_miscluster_mask]
                    miscluster_centroids = batch_centroids_k[batch_miscluster_mask]
                    # Push away from current centroid (encourage reassignment).
                    # Zero loss once distance is above margin.
                    miscluster_distances = torch.sum((miscluster_z - miscluster_centroids)**2, dim=1)
                    loss_miscluster_k = torch.mean(
                        torch.clamp(self.margin_cl - miscluster_distances, min=0)
                    )
                else:
                    loss_miscluster_k = torch.tensor(0.0, device=z.device)
                loss_miscluster_per_head.append(loss_miscluster_k)

        # Cluster Split Loss: Encourage variance in clusters marked for splitting.
        # Uses 1/(var + eps) so the loss is high when the cluster is tight and
        # drops toward zero as items spread out, giving a strong gradient signal.
        loss_split_per_head = []
        if cluster_splits:
            for k in range(num_heads):
                z_k = z[:, k, :]  # [B, p]
                centroids_k = centroids[:, k, :]  # [K, p]

                split_loss_k = torch.tensor(0.0, device=z.device)
                for cluster_id in cluster_splits:
                    if cluster_id < K:
                        cluster_mask = (assignments == cluster_id)
                        if cluster_mask.sum() > 1:
                            cluster_z = z_k[cluster_mask]
                            cluster_centroid = centroids_k[cluster_id]
                            sq_dists = torch.sum((cluster_z - cluster_centroid)**2, dim=1)
                            variance = sq_dists.mean()
                            split_loss_k += 1.0 / (variance + 1e-4)

                loss_split_per_head.append(split_loss_k)

        # Keyword Emphasis Loss: Maximize variance of projections along keyword
        # directions so the adapter learns to separate items by these concepts.
        loss_keyword_per_head = []
        if emphasized_keywords and keyword_embeddings:
            for k in range(num_heads):
                z_k = z[:, k, :]  # [B, p]

                keyword_loss_k = torch.tensor(0.0, device=z.device)
                for keyword in emphasized_keywords:
                    if keyword in keyword_embeddings:
                        keyword_emb = keyword_embeddings[keyword][k, :]  # [p]
                        keyword_dir = F.normalize(keyword_emb.unsqueeze(0), dim=1)  # [1, p]
                        projections = (z_k * keyword_dir).sum(dim=1)  # [B]
                        variance = torch.var(projections)
                        keyword_loss_k += 1.0 / (variance + 1e-4)

                if emphasized_keywords:
                    keyword_loss_k /= len(emphasized_keywords)

                loss_keyword_per_head.append(keyword_loss_k)

        loss_concept = torch.mean(torch.stack(loss_concept_per_head)) if loss_concept_per_head else torch.tensor(0.0, device=z.device)

        # Aggregate feedback losses: use min across heads (at least one head should satisfy constraints)
        # Apply weights to each component
        
        if loss_ml_per_head:
            loss_ml = torch.min(torch.stack(loss_ml_per_head)) * self.weights["must_link"]
        else:
            loss_ml = torch.tensor(0.0, device=z.device)

        if loss_cl_per_head:
            loss_cl = torch.min(torch.stack(loss_cl_per_head)) * self.weights["cannot_link"]
        else:
            loss_cl = torch.tensor(0.0, device=z.device)

        # Concept loss is already averaged, apply weight
        loss_concept = loss_concept * self.weights["concept"]

        # Aggregate additional losses with weights
        loss_miscluster = (torch.min(torch.stack(loss_miscluster_per_head)) if loss_miscluster_per_head else torch.tensor(0.0, device=z.device)) * self.weights["miscluster"]
        loss_split = (torch.min(torch.stack(loss_split_per_head)) if loss_split_per_head else torch.tensor(0.0, device=z.device)) * self.weights["cluster_split"]
        loss_keyword = (torch.min(torch.stack(loss_keyword_per_head)) if loss_keyword_per_head else torch.tensor(0.0, device=z.device)) * self.weights["keyword"]

        loss_feedback = loss_ml + loss_cl + loss_concept + loss_miscluster + loss_split + loss_keyword
        
        return loss_cluster, loss_feedback

    def decorrelation_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Optional: encourage feature dimensions to be uncorrelated.
        z: [B, num_heads, p] - apply per head and sum
        """
        B, num_heads, p = z.shape
        if B < 2: return torch.tensor(0.0, device=z.device)

        total_loss = 0.0
        for k in range(num_heads):
            z_k = z[:, k, :]  # [B, p]

            # Center
            z_centered = z_k - z_k.mean(dim=0, keepdim=True)
            # Covariance
            cov = (z_centered.T @ z_centered) / (B - 1)
            # Off-diagonal should be zero
            identity = torch.eye(p, device=z.device)
            loss_decor_k = torch.norm(cov - identity, p='fro')**2
            total_loss += loss_decor_k

        return total_loss
