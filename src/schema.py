from enum import Enum
from typing import List, Optional, Union, Any, Dict
from pydantic import BaseModel, Field

class FeedbackType(str, Enum):
    MUST_LINK = "must_link"         # Items should be together
    CANNOT_LINK = "cannot_link"     # Items should be apart
    MISCLUSTER = "miscluster"       # Item is in the wrong cluster
    MERGE_CLUSTERS = "merge_clusters" # Two clusters are the same topic
    RENAME_CLUSTER = "rename_cluster"       # Assign a semantic label to a cluster
    EMPHASIZE_FEATURE = "emphasize_feature" # Focus on specific concepts/keywords
    SUBCLUSTER = "subcluster"               # Split a specific cluster
    ASSIGN_OUTLIER = "assign_outlier"       # Force assign an outlier to a cluster

    # System control actions
    ROLLBACK = "rollback"           # Go back to a previous step
    SHOW_HISTORY = "show_history"   # Display conversation history
    RESET_SESSION = "reset_session" # Start over completely

class FeedbackAction(BaseModel):
    feedback_type: FeedbackType
    item_ids: Optional[List[int]] = Field(default=None, description="IDs of documents involved")
    cluster_ids: Optional[List[int]] = Field(default=None, description="IDs of clusters involved")
    text_payload: Optional[str] = Field(default=None, description="Label text or keyword to emphasize")
    step_number: Optional[int] = Field(default=None, description="Step number for rollback")
    reason: Optional[str] = Field(default=None, description="User's reasoning")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

class ChatMessage(BaseModel):
    role: str # "user", "assistant", "system"
    content: str

class InteractionContext(BaseModel):
    """What the LLM sees about the current state"""
    cluster_summaries: List[dict] # id, keywords, sample_texts
    focused_item_ids: Optional[List[int]] = None
    focused_pair_ids: Optional[List[int]] = None

class SystemState(BaseModel):
    """Snapshot of system state for rollback functionality"""
    step_number: int
    timestamp: str
    description: str

    # Core state
    adapter_state: dict  # Adapter weights
    labels: List[int]    # Cluster assignments
    centroids: List[List[float]]  # Cluster centroids

    # Constraints
    must_links: List[List[int]]
    cannot_links: List[List[int]]
    miscluster_flags: List[int]
    cluster_splits: List[int]
    cluster_labels: Dict[int, str]
    emphasized_keywords: List[str]

    # Metadata
    cluster_metadata: Dict[int, dict]

class FeedbackResponse(BaseModel):
    """Output from the LLM Feedback Parser"""
    thought_process: str
    actions: List[FeedbackAction]
    reply_to_user: str # Natural language acknowledgement
