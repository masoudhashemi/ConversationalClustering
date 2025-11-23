import json
from typing import List, Dict, Any

import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

from src.schema import (
    FeedbackResponse,
    ChatMessage,
    InteractionContext,
)


class LLMOracle:
    """
    Handles interactions with the LLM for:
    1. Generating questions (Active Learning).
    2. Processing user chat (Chat Schema).
    """

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_llm_json(self, system_prompt: str, user_prompt: str) -> Dict:
        """Helper to call LiteLLM expecting JSON."""
        response = litellm.completion(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    def chat_with_feedback(self, history: List[ChatMessage], context: InteractionContext) -> FeedbackResponse:
        """
        Main entry point for the modular chat schema.
        Takes history + context -> Returns FeedbackResponse (actions + reply).
        """
        system_prompt = """
        You are an intelligent assistant for a text clustering system.
        Your goal is to understand the user's natural language feedback about text groupings and convert it into structured constraint actions.

        CRITICAL: Analyze the user's message and identify which type of constraint they're expressing.

        Available Constraint Actions:
        - must_link: User wants specific items to be grouped together (e.g., "items X and Y should be together", "X and Y are similar")
        - cannot_link: User wants specific items to be kept apart (e.g., "X should NOT be with Y", "separate X from Y")
        - miscluster: User indicates an item is in the wrong cluster (e.g., "item X is misplaced", "X doesn't belong here")
        - merge_clusters: User wants to combine clusters (e.g., "clusters A and B are the same topic", "merge clusters X and Y")
        - rename_cluster: User wants to give a cluster a semantic label (e.g., "call cluster X 'Action Movies'", "rename cluster Y to 'Thrillers'")
        - emphasize_feature: User wants to prioritize certain concepts (e.g., "focus on thriller elements", "emphasize mystery aspects")
        - subcluster: User wants to split a cluster (e.g., "cluster X is too mixed, split it", "divide cluster Y into subgroups")
        - assign_outlier: User wants to force an outlier into a specific cluster (e.g., "move outlier Z to cluster A")

        Context Information:
        - Current cluster summaries show existing groupings with titles and keywords
        - Item IDs and cluster IDs are provided for reference
        - Focus on understanding user intent from natural language

        Response Format - JSON object:
        {
            "thought_process": "Step-by-step reasoning about what constraint the user is expressing",
            "actions": [
                {
                    "feedback_type": "exact_action_name_from_list_above",
                    "item_ids": [list of item IDs if applicable],
                    "cluster_ids": [list of cluster IDs if applicable],
                    "text_payload": "text content for labels/keywords if needed",
                    "reason": "brief explanation of the constraint",
                    "confidence": 0.0-1.0
                }
            ],
            "reply_to_user": "Natural language confirmation of what you understood"
        }

        Guidelines:
        - Only generate actions when user expresses actual constraints
        - Use appropriate IDs from context
        - Be confident (0.8-1.0) when intent is clear
        - Multiple actions can be generated if user expresses multiple constraints
        """

        context_str = context.json()
        history_str = "\n".join([f"{msg.role.upper()}: {msg.content}" for msg in history])

        user_prompt = f"""
        Current System State:
        {context_str}
        
        Conversation History:
        {history_str}
        
        Analyze the last user message. Extract any clustering constraints or feedback.
        If the user is just chatting, return empty actions.
        """

        print("[LLM] Processing chat message...")
        try:
            data = self._call_llm_json(system_prompt, user_prompt)
            return FeedbackResponse(**data)
        except Exception as e:
            print(f"[LLM] Error in chat: {e}")
            return FeedbackResponse(
                thought_process="Error",
                actions=[],
                reply_to_user="I'm sorry, I had trouble processing that. Could you rephrase?",
            )

    def propose_questions(self, candidates: List[Dict], context_str: str) -> List[Dict]:
        """
        Given a list of candidate items/pairs and context, select the best ones to ask.
        Returns a list of structured question objects.
        """
        system_prompt = """You are an expert at active learning for text clustering. 
        Your goal is to select the most informative questions to ask a human to improve the clustering.
        
        Strategy for selection:
        1. Reduce Uncertainty: Prioritize items where the model is unsure (e.g., on the boundary between clusters).
        2. Increase Homogeneity: Check items that are outliers in their assigned cluster.
        3. Clarify Boundaries: Ask about pairs that are close in distance but in different clusters, or far apart but in the same cluster.
        4. Mutual Information: Prefer questions that resolve ambiguity for entire groups of items, not just isolated cases.

        CRITICAL: Perform a "Common Sense Check" on every candidate.
        - Some pairs might be geometrically close but semantically absurd (e.g., "Action Movie" vs "RomCom").
        - DO NOT ask questions that a human would find obviously silly or unrelated.
        - Only propose questions where there is a GENUINE ambiguity or a reasonable chance the user considers them related.
        
        Return your response as a valid JSON object with a key 'questions' containing the list of questions.
        Each question should clarify *why* it is being asked (e.g. "These are close but in different clusters")."""

        user_prompt = f"""
        Context (Current Clusters): 
        {context_str}
        
        Candidates for feedback (with scores/metadata):
        {json.dumps(candidates, indent=2)}
        
        Task: Select up to 3 most impactful questions from the candidates to ask the user.
        Include the full text or IDs in the question so the user knows what they are judging.
        
        Output format:
        {{
            "questions": [
                {{
                    "query_id": "unique_id",
                    "type": "pair_same_or_diff" OR "miscluster_flag",
                    "item_ids": [id1, id2] OR [id1],
                    "question_text": "Natural language question for the user...",
                    "reason": "Brief explanation of why this question is important AND why it passes the common sense check."
                }}
            ]
        }}
        """
        
        print(f"[LLM] Generating questions for {len(candidates)} candidates...")
        try:
            data = self._call_llm_json(system_prompt, user_prompt)
            return data.get("questions", [])
        except Exception as e:
            print(f"[LLM] Error generating questions: {e}")
            return []
