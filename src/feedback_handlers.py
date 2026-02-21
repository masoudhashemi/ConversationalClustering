from typing import List, Protocol, Any
from src.schema import FeedbackAction, FeedbackType
from src.constraints import ConstraintStore
import numpy as np

class FeedbackHandler(Protocol):
    def handle(self, action: FeedbackAction, store: ConstraintStore, context: Any = None):
        """Apply the action to the store."""
        pass

class MustLinkHandler:
    def handle(self, action: FeedbackAction, store: ConstraintStore, context: Any = None):
        if not action.item_ids or len(action.item_ids) < 2:
            return
        # Create pairwise links for all items in the list
        for i in range(len(action.item_ids)):
            for j in range(i + 1, len(action.item_ids)):
                store.add_must_link(action.item_ids[i], action.item_ids[j])

class CannotLinkHandler:
    def handle(self, action: FeedbackAction, store: ConstraintStore, context: Any = None):
        if not action.item_ids or len(action.item_ids) < 2:
            return
        if len(action.item_ids) == 2:
            store.add_cannot_link(action.item_ids[0], action.item_ids[1])
        else:
            for i in range(len(action.item_ids)):
                for j in range(i + 1, len(action.item_ids)):
                    store.add_cannot_link(action.item_ids[i], action.item_ids[j])

class MisclusterHandler:
    def handle(self, action: FeedbackAction, store: ConstraintStore, context: Any = None):
        if not action.item_ids: return
        for idx in action.item_ids:
            store.add_miscluster(idx)

class MergeClustersHandler:
    def handle(self, action: FeedbackAction, store: ConstraintStore, context: Any = None):
        if not context or not action.cluster_ids or len(action.cluster_ids) < 2:
            return
        labels = context.labels
        c1 = action.cluster_ids[0]
        c2 = action.cluster_ids[1]
        
        idxs_c1 = np.where(labels == c1)[0]
        idxs_c2 = np.where(labels == c2)[0]
        if len(idxs_c1) == 0 or len(idxs_c2) == 0: return
        
        k = min(3, len(idxs_c1), len(idxs_c2))
        anchors_1 = np.random.choice(idxs_c1, k, replace=False)
        anchors_2 = np.random.choice(idxs_c2, k, replace=False)
        
        for i1, i2 in zip(anchors_1, anchors_2):
            store.add_must_link(int(i1), int(i2))

class RenameClusterHandler:
    def handle(self, action: FeedbackAction, store: ConstraintStore, context: Any = None):
        if not action.cluster_ids or not action.text_payload:
            return
        # Assign label to the first cluster ID in list
        store.set_cluster_label(action.cluster_ids[0], action.text_payload)

class EmphasizeFeatureHandler:
    import re as _re
    _SPLIT_RE = _re.compile(r'\s*(?:,|;|\band\b|\bor\b|\b&\b)\s*', _re.IGNORECASE)
    _NOISE_WORDS = _re.compile(
        r'\b(?:elements?|aspects?|features?|concepts?|themes?|keywords?|topics?|factors?)\b',
        _re.IGNORECASE,
    )

    @classmethod
    def _clean(cls, token: str) -> str:
        """Strip filler/noise words and collapse whitespace."""
        cleaned = cls._NOISE_WORDS.sub('', token).strip()
        cleaned = cls._re.sub(r'\s+', ' ', cleaned)
        return cleaned

    def handle(self, action: FeedbackAction, store: ConstraintStore, context: Any = None):
        if not action.text_payload:
            return
        raw = action.text_payload.strip()
        tokens = self._SPLIT_RE.split(raw)
        keywords = [self._clean(t) for t in tokens]
        keywords = [kw for kw in keywords if kw]
        if not keywords:
            keywords = [raw]
        for kw in keywords:
            store.add_emphasized_keyword(kw)

class RollbackHandler:
    def handle(self, action: FeedbackAction, store: ConstraintStore, context: Any = None):
        if not context or action.step_number is None:
            return
        # Context should be the engine with rollback capability
        context.rollback_to_step(action.step_number)

class ShowHistoryHandler:
    def handle(self, action: FeedbackAction, store: ConstraintStore, context: Any = None):
        if not context:
            return
        # This is more of a display action, handled by the engine
        pass

class ResetSessionHandler:
    def handle(self, action: FeedbackAction, store: ConstraintStore, context: Any = None):
        if not context:
            return
        context.reset_session()

class SubclusterHandler:
    def handle(self, action: FeedbackAction, store: ConstraintStore, context: Any = None):
        if not action.cluster_ids:
            return
        for cluster_id in action.cluster_ids:
            store.add_cluster_split(cluster_id)
            if context and hasattr(context, 'increase_n_clusters'):
                context.increase_n_clusters(1)

class AssignOutlierHandler:
    def handle(self, action: FeedbackAction, store: ConstraintStore, context: Any = None):
        if not action.item_ids or not action.cluster_ids or len(action.cluster_ids) != 1:
            return
        # For now, this is just a placeholder - the actual assignment happens during clustering
        # We could add a constraint to force assignment, but this is complex
        # For now, just mark it as a must-link to the cluster's items
        if context and hasattr(context, 'labels'):
            cluster_id = action.cluster_ids[0]
            cluster_items = [i for i, label in enumerate(context.labels) if label == cluster_id]
            if cluster_items:
                # Create must-links between the outlier and some cluster items
                item_id = action.item_ids[0]
                # Link to first few items in the target cluster
                for cluster_item in cluster_items[:3]:  # Link to up to 3 items
                    store.add_must_link(item_id, cluster_item)

class FeedbackExecutor:
    """
    Registry and dispatcher for feedback actions.
    """
    def __init__(self):
        self.handlers = {
            FeedbackType.MUST_LINK: MustLinkHandler(),
            FeedbackType.CANNOT_LINK: CannotLinkHandler(),
            FeedbackType.MISCLUSTER: MisclusterHandler(),
            FeedbackType.MERGE_CLUSTERS: MergeClustersHandler(),
            FeedbackType.RENAME_CLUSTER: RenameClusterHandler(),
            FeedbackType.EMPHASIZE_FEATURE: EmphasizeFeatureHandler(),
            FeedbackType.SUBCLUSTER: SubclusterHandler(),
            FeedbackType.ASSIGN_OUTLIER: AssignOutlierHandler(),
            FeedbackType.ROLLBACK: RollbackHandler(),
            FeedbackType.SHOW_HISTORY: ShowHistoryHandler(),
            FeedbackType.RESET_SESSION: ResetSessionHandler(),
        }

    def execute(self, actions: List[FeedbackAction], store: ConstraintStore, context: Any = None):
        for action in actions:
            handler = self.handlers.get(action.feedback_type)
            if handler:
                print(f"[Executor] Applying {action.feedback_type}...")
                handler.handle(action, store, context)
            else:
                print(f"[Executor] No handler for {action.feedback_type}")
