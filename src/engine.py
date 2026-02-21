import torch
import torch.nn.functional as F
import json
import torch.optim as optim
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import copy

from src.models import TextEncoder, EmbeddingAdapter
from src.clustering import ClusteringModule
from src.constraints import ConstraintStore
from src.optimization import FeedbackLoss
from src.llm_utils import LLMOracle
from src.schema import ChatMessage, InteractionContext, SystemState, FeedbackType
from src.feedback_handlers import FeedbackExecutor
from src.ann_search import ApproximateNeighborSearch, ANNConfig

class ClusterRefinementEngine:
    def __init__(
        self,
        texts: List[str],
        n_clusters: int = 5,
        device: str = "cpu",
        use_projection_head: bool = False,
        num_heads: int = 1,
        llm_model: str = "gpt-4o",
        enable_ann_search: bool = True,
        ann_config: Optional[ANNConfig] = None,
    ):
        self.texts = texts
        self.n_clusters = n_clusters
        self.device = device

        self.encoder = TextEncoder(device=device)
        self.constraints = ConstraintStore()
        self.llm = LLMOracle(model_name=llm_model)
        self.clustering = ClusteringModule(n_clusters=n_clusters)
        self.feedback_executor = FeedbackExecutor()

        # Phase 1 optimizations
        self.enable_ann_search = enable_ann_search

        # Initialize ANN search
        if enable_ann_search:
            self.ann_config = ann_config or ANNConfig()
            self.ann_search = ApproximateNeighborSearch(self.ann_config)
        else:
            self.ann_search = None

        # Encoding with batch processing
        print("Encoding texts with batch processing...")
        self.h = torch.tensor(self.encoder.encode_batch(texts), device=device)
        input_dim = self.h.shape[1]

        self.adapter = EmbeddingAdapter(
            input_dim=input_dim,
            output_dim=input_dim,
            num_heads=num_heads,
            non_linear=use_projection_head
        ).to(device)

        self.optimizer = optim.Adam(self.adapter.parameters(), lr=0.001)
        
        # Prioritize user constraints heavily
        self.criterion = FeedbackLoss(
            weights={
                "must_link": 5.0,      # Very high priority
                "cannot_link": 5.0,    # Very high priority
                "miscluster": 3.0,     # High priority
                "concept": 2.0,        # Medium priority
                "cluster_split": 3.0,  # High priority
                "keyword": 2.0         # Medium priority
            }
        )

        # Store baseline embeddings (before any adaptation) for constraint distance tracking
        with torch.no_grad():
            self.baseline_embeddings = self.adapter(self.h).cpu().numpy()

        self.constraints = ConstraintStore(baseline_embeddings=self.baseline_embeddings)

        self.z: Optional[torch.Tensor] = None
        self.labels: Optional[np.ndarray] = None
        self.history: List[ChatMessage] = []

        # Track items that have already been surfaced in active-learning
        # questions so we avoid repeating the same suggestions.
        self._asked_item_ids: set = set()

        # State management for rollback
        self.step_counter: int = 0
        self.state_history: List[SystemState] = []
        self.initial_state: Optional[SystemState] = None

        self._update_clustering()
        # Save an initial checkpoint so rollback/reset can always return to step 0.
        self.initial_state = self._create_checkpoint("Initial state")
        self.state_history.append(copy.deepcopy(self.initial_state))
        self.step_counter = 1

    def _update_clustering(self):
        with torch.no_grad():
            self.z = self.adapter(self.h)
            z_np = self.z.cpu().numpy()
            self.labels = self.clustering.fit_predict(z_np)

        # Build ANN index for active learning (if enabled)
        if self.enable_ann_search and self.ann_search is not None:
            print("Building ANN index for active learning...")
            # Use flattened multi-head embeddings for ANN (matches clustering space)
            if z_np.ndim == 3:
                ann_embeddings = z_np.reshape(z_np.shape[0], -1)
            else:
                ann_embeddings = z_np

            # Rely on FAISS's own PQ / indexing strategy; no external PQ here
            self.ann_search.build_index(ann_embeddings)

        # Auto-Summarization Step
        self._generate_cluster_metadata()

    def _generate_cluster_metadata(self):
        """
        Simple extraction of top keywords/terms per cluster to serve as titles.
        In a real system, we could ask the LLM to title them.
        For now, we use a simple frequency-based heuristic or just placeholders
        that get updated if the user renames them.
        """
        if self.labels is None: return
        
        # Reset or update metadata
        # Ideally we keep user-defined labels if they exist in constraints
        self.cluster_metadata = {}
        
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(stop_words='english', max_features=10)
        
        for k in range(self.n_clusters):
            indices = np.where(self.labels == k)[0]
            if len(indices) == 0:
                self.cluster_metadata[k] = {"title": f"Cluster {k} (Empty)", "keywords": []}
                continue
                
            cluster_texts = [self.texts[i] for i in indices]
            try:
                X = vectorizer.fit_transform(cluster_texts)
                keywords = vectorizer.get_feature_names_out()
                # Simple title: top 2 keywords
                title = f"{keywords[0]}/{keywords[1]}" if len(keywords) >= 2 else f"Cluster {k}"
            except ValueError: 
                # items might be stopwords only or empty
                title = f"Cluster {k}"
                keywords = []
            
            # Override with user label if present
            if k in self.constraints.cluster_labels:
                title = self.constraints.cluster_labels[k]
                
            self.cluster_metadata[k] = {
                "title": title,
                "keywords": list(keywords)
            }

    def get_cluster_summaries(self) -> List[Dict]:
        """Generate simple summaries for LLM context."""
        summaries = []
        if self.labels is None: return []
        
        # Ensure metadata exists
        if not hasattr(self, 'cluster_metadata'):
            self._generate_cluster_metadata()
            
        for k in range(self.n_clusters):
            indices = np.where(self.labels == k)[0]
            if len(indices) == 0: continue
            
            meta = self.cluster_metadata.get(k, {})
            examples = [self.texts[i] for i in indices[:3]]
            
            summaries.append({
                "cluster_id": int(k),
                "title": meta.get("title", f"Cluster {k}"),
                "keywords": meta.get("keywords", []),
                "size": int(len(indices)),
                "examples": examples
            })
        return summaries

    def chat(self, user_message: str) -> str:
        """
        Process a user message, extract feedback, update constraints, and reply.
        """
        # Handle special commands directly
        if user_message.lower().strip() in ["show history", "history", "what happened"]:
            return self.show_history()

        # 1. Build Context
        context = InteractionContext(
            cluster_summaries=self.get_cluster_summaries(),
            # In a real UI, we might know what the user is looking at
            focused_item_ids=None
        )

        # 2. Append to history
        self.history.append(ChatMessage(role="user", content=user_message))

        # 3. Call LLM
        response = self.llm.chat_with_feedback(self.history, context)

        # 4. Execute Actions
        if response.actions:
            print(f"Executing {len(response.actions)} feedback actions...")

            # Check for special actions
            has_rollback = any(a.feedback_type == FeedbackType.ROLLBACK for a in response.actions)
            has_reset = any(a.feedback_type == FeedbackType.RESET_SESSION for a in response.actions)
            has_show_history = any(a.feedback_type == FeedbackType.SHOW_HISTORY for a in response.actions)

            if has_show_history:
                # Handle show history immediately
                history_text = self.show_history()
                response.reply_to_user += f"\n\n{history_text}"

            elif has_rollback or has_reset:
                # Execute rollback/reset without saving checkpoint
                self.feedback_executor.execute(response.actions, self.constraints, context=self)

            else:
                # Normal feedback actions
                self.feedback_executor.execute(response.actions, self.constraints, context=self)

                # Save checkpoint before training
                action_types = [str(a.feedback_type) for a in response.actions]
                desc = f"Applied: {', '.join(action_types)}"
                self._save_checkpoint(desc)

                # Trigger a training step
                self.train_step(epochs=10)

        # 5. Update history and return reply
        self.history.append(ChatMessage(role="assistant", content=response.reply_to_user))
        return response.reply_to_user

    def train_step(self, epochs: int = 10):
        """Same training loop as before"""
        if not self.constraints.must_links and not self.constraints.cannot_links and not self.constraints.cluster_labels and not self.constraints.miscluster_flags and not self.constraints.cluster_splits and not self.constraints.emphasized_keywords:
            return

        print(f"Training for {epochs} epochs...")
        self.adapter.train()
        ml = list(self.constraints.must_links)
        cl = list(self.constraints.cannot_links)

        # Prepare semantic labels for Concept Loss
        label_embeddings = {}
        if self.constraints.cluster_labels:
            # We need to encode the label strings. 
            # Ideally, we cache this, but for simplicity we encode on demand.
            # Note: self.encoder.encode() might be slow if called every step, so we do it once here.
            labels_to_encode = []
            c_ids = []
            for cid, text in self.constraints.cluster_labels.items():
                c_ids.append(cid)
                labels_to_encode.append(text)
            
            if labels_to_encode:
                embs = self.encoder.encode(labels_to_encode) # [k, d]
                # We need to adapt these label embeddings into Z space too? 
                # Or do we treat them as ground truth in Z space?
                # Usually, labels are "concepts" in the base space (h) that we want Z to align with.
                # But Z is the metric space. If we want Z to cluster around "Billing", 
                # we should map "Billing" to Z space and pull centroids there.
                # So: label_z = adapter(label_h)
                # But wait! If we train adapter to move label_z, we might just move the label 
                # to the cluster instead of the cluster to the label.
                # Better approach: Fix the label concept in a semantic space? 
                # Actually, simpler: Just map the label to Z using the *current* adapter (detached?)
                # or map it and let the adapter learn to put the cluster there.
                # Let's map h_label -> z_label and pull mu_k -> z_label.
                
                embs_tensor = torch.tensor(embs, device=self.device)
                # We want to optimize the adapter so that the cluster points land near the label.
                # So we shouldn't detach the label embedding if we want the adapter to put the points there?
                # Actually, we want the adapter to map "Billing Documents" to the same place as "Billing Label".
                # So yes, pass label_h through adapter.
                
                # Store base embeddings for loop
                self.label_base_embs = {cid: emb for cid, emb in zip(c_ids, embs_tensor)}

        # Prepare keyword embeddings for emphasis
        keyword_embeddings = {}
        if self.constraints.emphasized_keywords:
            keywords_to_encode = list(self.constraints.emphasized_keywords)
            if keywords_to_encode:
                keyword_embs = self.encoder.encode(keywords_to_encode)  # [k, d]
                keyword_embs_tensor = torch.tensor(keyword_embs, device=self.device)
                # Store base embeddings for loop
                self.keyword_base_embs = {kw: emb for kw, emb in zip(keywords_to_encode, keyword_embs_tensor)}
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            z = self.adapter(self.h)

            # Check for numerical issues in z
            if not torch.isfinite(z).all():
                print(f"Warning: Non-finite values in z at epoch {epoch}, skipping step")
                continue

            # Ensure z is properly normalized (per head)
            z_norm = torch.norm(z, dim=2, keepdim=True)  # [N, num_heads, 1, p] -> norm over p
            if not torch.allclose(z_norm, torch.ones_like(z_norm), atol=1e-6):
                z = F.normalize(z, p=2, dim=2)  # Normalize over last dim

            with torch.no_grad():
                z_np = z.detach().cpu().numpy()
                # Additional check for NaN/inf in numpy array
                if not np.isfinite(z_np).all():
                    print(f"Warning: Non-finite values in z_np at epoch {epoch}, skipping step")
                    continue
                self.labels = self.clustering.fit_predict(z_np)
            centroids_np = self.clustering.get_centroids()  # [K, num_heads*p]
            num_heads = z.shape[1]
            p = z.shape[2]
            centroids = torch.tensor(centroids_np, device=self.device, dtype=torch.float32).view(self.n_clusters, num_heads, p)
            assignments = torch.tensor(self.labels, device=self.device, dtype=torch.long)

            # Transform label base embeddings to Z space for this step
            label_z_map = {}
            if hasattr(self, 'label_base_embs'):
                for cid, h_lbl in self.label_base_embs.items():
                    # h_lbl is [d], unsqueeze to [1, d] -> [1, num_heads, p]
                    z_label = self.adapter(h_lbl.unsqueeze(0))  # [1, num_heads, p]
                    label_z_map[cid] = z_label.squeeze(0)  # [num_heads, p]

            # Transform keyword base embeddings to Z space for this step
            keyword_z_map = {}
            if hasattr(self, 'keyword_base_embs'):
                for kw, h_kw in self.keyword_base_embs.items():
                    # h_kw is [d], unsqueeze to [1, d] -> [1, num_heads, p]
                    z_keyword = self.adapter(h_kw.unsqueeze(0))  # [1, num_heads, p]
                    keyword_z_map[kw] = z_keyword.squeeze(0)  # [num_heads, p]

            loss_cluster, loss_fb = self.criterion(
                z, centroids, assignments, ml, cl, label_z_map,
                miscluster_flags=list(self.constraints.miscluster_flags),
                cluster_splits=list(self.constraints.cluster_splits),
                emphasized_keywords=list(self.constraints.emphasized_keywords),
                keyword_embeddings=keyword_z_map
            )
            loss = loss_cluster + 3.0 * loss_fb

            # Check for finite loss
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss at epoch {epoch}, skipping step")
                continue

            loss.backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=1.0)

            self.optimizer.step()

        self._update_clustering()

    def increase_n_clusters(self, delta: int = 1):
        """Increase the number of clusters (e.g. after a subcluster request)."""
        old_k = self.n_clusters
        self.n_clusters += delta
        self.clustering = ClusteringModule(n_clusters=self.n_clusters)
        print(f"[Engine] Increased n_clusters: {old_k} -> {self.n_clusters}")
        # Re-fit immediately so centroids are populated before any checkpoint.
        if self.z is not None:
            z_np = self.z.detach().cpu().numpy()
            self.labels = self.clustering.fit_predict(z_np)

    # ------------------------------------------------------------------
    # Active learning helpers
    # ------------------------------------------------------------------
    def propose_feedback_questions(self, max_questions: int = 3) -> List[Dict]:
        """
        Build candidate items/pairs/clusters with rich metadata and let the LLM
        select the best questions to ask the user.

        Previously-asked item IDs are excluded to avoid repetitive suggestions.
        """
        candidates, context = self._build_candidate_pool()
        if not candidates:
            return []

        # Filter out candidates whose items were already asked about.
        filtered = []
        for c in candidates:
            ids = set(c.get("ids", []))
            if ids and ids.issubset(self._asked_item_ids):
                continue
            filtered.append(c)
        candidates = filtered or candidates  # fall back if everything filtered

        context_str = json.dumps(context, indent=2)
        questions = self.llm.propose_questions(candidates, context_str)[:max_questions]

        # Remember which items we just surfaced so we don't repeat them.
        for q in questions:
            for item_id in q.get("item_ids", []):
                self._asked_item_ids.add(item_id)

        return questions

    def _build_candidate_pool(
        self,
        top_items: int = 6,
        top_pairs: int = 6,
        top_clusters: int = 3,
    ) -> Tuple[List[Dict], Dict]:
        if self.z is None or self.labels is None:
            return [], {}

        z_np = self.z.detach().cpu().numpy()
        # Flatten multi-head embeddings for candidate generation (metrics match K-means space)
        if z_np.ndim == 3:
            N, num_heads, p = z_np.shape
            z_np = z_np.reshape(N, num_heads * p)
            
        labels = self.labels
        centroids = self.clustering.get_centroids()
        cluster_stats = self.clustering.get_cluster_stats(z_np, labels)

        item_scores = self._compute_item_scores(z_np, centroids, labels)
        item_candidates = [
            {
                "candidate_id": f"item_{idx}",
                "type": "item_uncertain",
                "ids": [int(idx)],
                "cluster_id": int(labels[idx]),
                "text": self._excerpt(idx),
                "metrics": {
                    "distance_to_centroid": float(score["dist_to_centroid"]),
                    "margin_to_second_centroid": float(score["margin"]),
                },
                "reason": "High uncertainty (small margin to second cluster / high centroid distance).",
            }
            for idx, score in item_scores[:top_items]
        ]

        pair_candidates = self._build_pair_candidates(
            z_np, labels, item_scores[: top_pairs * 2]
        )

        cluster_candidates = []

        # Use ANN-based cluster analysis if available
        if self.enable_ann_search and self.ann_search is not None:
            high_variance_clusters = self.ann_search.find_high_variance_clusters(
                labels, centroids, variance_threshold=0.8
            )
            for stat in high_variance_clusters[:top_clusters]:
                cluster_candidates.append(
                    {
                        "candidate_id": f"cluster_{stat['id']}",
                        "type": "cluster_high_variance",
                        "cluster_id": int(stat["id"]),
                        "examples": [
                            self._excerpt(idx)
                            for idx in np.where(labels == stat["id"])[0][:3]
                        ],
                        "metrics": {
                            "variance": float(stat["variance"]),
                            "size": int(stat["size"]),
                        },
                        "reason": "Cluster shows high internal variance and may need splitting or relabeling (ANN-analyzed).",
                    }
                )
        else:
            # Fallback to original method
            sorted_clusters = sorted(
                cluster_stats, key=lambda c: (c["variance"], -c["size"]), reverse=True
            )
            for stat in sorted_clusters[:top_clusters]:
                cluster_candidates.append(
                    {
                        "candidate_id": f"cluster_{stat['id']}",
                        "type": "cluster_high_variance",
                        "cluster_id": int(stat["id"]),
                        "examples": [
                            self._excerpt(idx)
                            for idx in np.where(labels == stat["id"])[0][:3]
                        ],
                        "metrics": {
                            "variance": float(stat["variance"]),
                            "size": int(stat["size"]),
                        },
                        "reason": "Cluster shows high internal variance and may need splitting or relabeling.",
                    }
                )

        context = {
            "cluster_stats": cluster_stats,
            "total_items": len(self.texts),
            "top_uncertain_items": [
                {"id": int(idx), "cluster": int(labels[idx])}
                for idx, _ in item_scores[:top_items]
            ],
        }

        candidates: List[Dict] = []
        candidates.extend(item_candidates)
        candidates.extend(pair_candidates[:top_pairs])
        candidates.extend(cluster_candidates)

        # Deduplicate candidates that refer to the same underlying text
        # (e.g. repeated documents in the dataset).
        seen_texts: set = set()
        unique_candidates: List[Dict] = []
        for c in candidates:
            text_key = c.get("text") or str(c.get("texts", "")) or str(c.get("examples", ""))
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)
            unique_candidates.append(c)
        return unique_candidates, context

    def _compute_item_scores(self, z: np.ndarray, centroids: np.ndarray, labels: np.ndarray):
        """Return items sorted by low margin / high distance."""
        centroid_dists = np.linalg.norm(z[:, None, :] - centroids[None, :, :], axis=2)
        nearest = np.argsort(centroid_dists, axis=1)
        top1 = nearest[:, 0]
        top2 = nearest[:, 1]
        indices = np.arange(len(z))

        dist_to_centroid = centroid_dists[indices, top1]
        margin = centroid_dists[indices, top2] - centroid_dists[indices, top1]

        scores = []
        for idx in range(len(z)):
            scores.append(
                (
                    idx,
                    {
                        "dist_to_centroid": dist_to_centroid[idx],
                        "margin": margin[idx],
                    },
                )
            )

        # Sort by small margin first (uncertain), then by large distance
        scores.sort(key=lambda item: (item[1]["margin"], -item[1]["dist_to_centroid"]))
        return scores

    def _build_pair_candidates(
        self, z: np.ndarray, labels: np.ndarray, seed_items: List[Tuple[int, Dict]]
    ) -> List[Dict]:
        candidates = []

        # Use ANN search for efficient boundary pair detection
        if self.enable_ann_search and self.ann_search is not None:
            uncertain_indices = [idx for idx, _ in seed_items[:10]]  # Limit for efficiency
            boundary_pairs = self.ann_search.find_boundary_pairs(
                labels, uncertain_indices, max_pairs=len(seed_items)
            )

            for idx, j, distance in boundary_pairs:
                candidates.append(
                    {
                        "candidate_id": f"pair_{idx}_{j}",
                        "type": "pair_boundary",
                        "ids": [int(idx), int(j)],
                        "clusters": [int(labels[idx]), int(labels[j])],
                        "texts": [self._excerpt(idx), self._excerpt(j)],
                        "metrics": {
                            "distance": float(distance),
                        },
                        "reason": "Items are close in embedding space but currently assigned to different clusters (ANN-detected).",
                    }
                )
        else:
            # Fallback to original brute force method (for small datasets)
            print("Warning: Using brute force pair detection (consider enabling ANN search)")
            num_points = len(z)
            for idx, _ in seed_items[:5]:  # Limit for performance
                same_cluster = labels == labels[idx]
                cross_cluster = ~same_cluster
                if not np.any(cross_cluster):
                    continue

                distances = np.linalg.norm(z[idx] - z, axis=1)
                distances[idx] = np.inf
                masked = np.where(cross_cluster, distances, np.inf)
                j = int(np.argmin(masked))
                if j >= num_points or np.isinf(masked[j]):
                    continue

                candidates.append(
                    {
                        "candidate_id": f"pair_{idx}_{j}",
                        "type": "pair_boundary",
                        "ids": [int(idx), int(j)],
                        "clusters": [int(labels[idx]), int(labels[j])],
                        "texts": [self._excerpt(idx), self._excerpt(j)],
                        "metrics": {
                            "distance": float(masked[j]),
                        },
                        "reason": "Items are close in embedding space but currently assigned to different clusters.",
                    }
                )
        return candidates

    def _excerpt(self, idx: int, max_chars: int = 160) -> str:
        text = self.texts[idx]
        return (text[: max_chars - 3] + "...") if len(text) > max_chars else text

    # ------------------------------------------------------------------
    # State management and rollback functionality
    # ------------------------------------------------------------------

    def _create_checkpoint(self, description: str = "") -> SystemState:
        """Create a snapshot of the current system state."""
        return SystemState(
            step_number=self.step_counter,
            timestamp=datetime.now().isoformat(),
            description=description,

            # Core state
            adapter_state={
                'state_dict': copy.deepcopy(self.adapter.state_dict()),
                'optimizer_state': copy.deepcopy(self.optimizer.state_dict())
            },
            labels=self.labels.tolist() if self.labels is not None else [],
            centroids=self.clustering.get_centroids().tolist(),

            # Constraints
            must_links=list(self.constraints.must_links),
            cannot_links=list(self.constraints.cannot_links),
            miscluster_flags=list(self.constraints.miscluster_flags),
            cluster_splits=list(self.constraints.cluster_splits),
            cluster_labels=dict(self.constraints.cluster_labels),
            emphasized_keywords=list(self.constraints.emphasized_keywords),

            # Metadata
            cluster_metadata=copy.deepcopy(getattr(self, 'cluster_metadata', {}))
        )

    def _save_checkpoint(self, description: str = ""):
        """Save current state to history."""
        checkpoint = self._create_checkpoint(description)
        self.state_history.append(checkpoint)
        self.step_counter += 1

        # Keep only last 10 checkpoints to save memory
        if len(self.state_history) > 10:
            self.state_history.pop(0)

    def rollback_to_step(self, step_number: int):
        """Rollback system to a specific step."""
        # Find the checkpoint
        checkpoint = None
        for state in reversed(self.state_history):
            if state.step_number == step_number:
                checkpoint = state
                break

        if not checkpoint:
            print(f"[Engine] No checkpoint found for step {step_number}")
            return False

        print(f"[Engine] Rolling back to step {step_number}: {checkpoint.description}")

        # Restore adapter state
        self.adapter.load_state_dict(checkpoint.adapter_state['state_dict'])
        self.optimizer.load_state_dict(checkpoint.adapter_state['optimizer_state'])

        # Restore clustering state
        self.labels = np.array(checkpoint.labels)
        self.clustering = ClusteringModule(n_clusters=self.n_clusters)
        centroids_array = np.array(checkpoint.centroids)
        self.clustering.kmeans.cluster_centers_ = centroids_array
        self.clustering.centroids = centroids_array

        # Restore constraints
        self.constraints = ConstraintStore(baseline_embeddings=self.baseline_embeddings)
        for ml in checkpoint.must_links:
            self.constraints.add_must_link(ml[0], ml[1])
        for cl in checkpoint.cannot_links:
            self.constraints.add_cannot_link(cl[0], cl[1])
        for flag in checkpoint.miscluster_flags:
            self.constraints.add_miscluster(flag)
        for split_id in checkpoint.cluster_splits:
            self.constraints.add_cluster_split(split_id)
        self.constraints.cluster_labels = dict(checkpoint.cluster_labels)
        self.constraints.emphasized_keywords = set(checkpoint.emphasized_keywords)

        # Restore metadata
        self.cluster_metadata = checkpoint.cluster_metadata

        # Update embeddings
        self._update_clustering()

        # Remove checkpoints after the rollback point
        self.state_history = [s for s in self.state_history if s.step_number <= step_number]
        self.step_counter = step_number + 1

        return True

    def show_history(self) -> str:
        """Return formatted history of steps for user."""
        if not self.state_history:
            return "No history available yet."

        lines = ["Conversation History:"]
        for state in self.state_history:
            lines.append(f"Step {state.step_number}: {state.description} ({state.timestamp})")

        lines.append(f"\nCurrent step: {self.step_counter}")
        return "\n".join(lines)

    def reset_session(self):
        """Reset to initial state."""
        if self.initial_state:
            print("[Engine] Resetting session to initial state")
            checkpoint = copy.deepcopy(self.initial_state)

            # Restore adapter state
            self.adapter.load_state_dict(checkpoint.adapter_state['state_dict'])
            self.optimizer.load_state_dict(checkpoint.adapter_state['optimizer_state'])

            # Restore constraints
            self.constraints = ConstraintStore(baseline_embeddings=self.baseline_embeddings)
            for ml in checkpoint.must_links:
                self.constraints.add_must_link(ml[0], ml[1])
            for cl in checkpoint.cannot_links:
                self.constraints.add_cannot_link(cl[0], cl[1])
            for flag in checkpoint.miscluster_flags:
                self.constraints.add_miscluster(flag)
            for split_id in checkpoint.cluster_splits:
                self.constraints.add_cluster_split(split_id)
            self.constraints.cluster_labels = dict(checkpoint.cluster_labels)
            self.constraints.emphasized_keywords = set(checkpoint.emphasized_keywords)

            # Recompute clustering from restored model and constraints
            self._update_clustering()

            # Keep only the initial checkpoint in history.
            self.state_history = [copy.deepcopy(checkpoint)]
            self.step_counter = 1
            print("[Engine] Session reset to initial state")
        else:
            print("[Engine] No initial state to reset to")

    # ------------------------------------------------------------------
    # Memory monitoring and profiling
    # ------------------------------------------------------------------

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        stats = {
            'current_memory_mb': process.memory_info().rss / 1024 / 1024,
            'memory_percent': process.memory_percent(),
            'num_texts': len(self.texts),
            'embedding_shape': list(self.h.shape) if self.h is not None else None,
            'adapted_embedding_shape': list(self.z.shape) if self.z is not None else None,
            'n_clusters': self.n_clusters,
            'device': str(self.device),
        }

        # Add component-specific stats
        if hasattr(self.encoder, 'get_memory_stats'):
            stats['encoder_memory'] = self.encoder.get_memory_stats()

        if self.ann_search is not None:
            stats['ann_memory'] = self.ann_search.get_memory_stats()

        # Estimate total memory usage breakdown
        if self.h is not None:
            stats['embedding_memory_mb'] = self.h.numel() * self.h.element_size() / 1024 / 1024

        if self.z is not None:
            stats['adapted_memory_mb'] = self.z.numel() * self.z.element_size() / 1024 / 1024

        return stats

    def print_memory_report(self) -> None:
        """Print a detailed memory usage report."""
        stats = self.get_memory_stats()

        print("\n" + "="*60)
        print("MEMORY USAGE REPORT")
        print("="*60)

        print(f"Current Memory Usage: {stats['current_memory_mb']:.1f} MB ({stats['memory_percent']:.1f}%)")
        print(f"Dataset Size: {stats['num_texts']:,} texts")
        print(f"Clusters: {stats['n_clusters']}")
        print(f"Device: {stats['device']}")

        if 'embedding_memory_mb' in stats:
            print(f"Base Embeddings: {stats['embedding_memory_mb']:.1f} MB")
        if 'adapted_memory_mb' in stats:
            print(f"Adapted Embeddings: {stats['adapted_memory_mb']:.1f} MB")

        if 'encoder_memory' in stats:
            enc = stats['encoder_memory']
            print(f"Encoder Peak Memory: {enc.get('peak_memory_gb', 0):.2f} GB")

        if 'ann_memory' in stats:
            ann = stats['ann_memory']
            print(f"ANN Index Size (Est): {ann.get('index_memory_history', [0])[-1]:.1f} MB")

        total_estimated = sum([
            stats.get('embedding_memory_mb', 0),
            stats.get('adapted_memory_mb', 0)
        ])
        print(f"Estimated Core Memory: {total_estimated:.1f} MB")
        print("="*60)
