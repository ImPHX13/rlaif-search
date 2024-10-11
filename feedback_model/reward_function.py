from feedback_model.feedback_agent import FeedbackAgent
import numpy as np

def calculate_reward(feedback_agent: FeedbackAgent, query: str, ranked_documents: list):
    """
    This function calculates the reward for a given query and ranked documents.
    It uses the Discounted Cumulative Gain (DCG) and Normalized DCG (NDCG) metrics.
    It also adds a bonus for improving over the initial ranking.
    """
    if not ranked_documents:
        return 0
    relevance_scores = [feedback_agent.predict_relevance(query, doc) for doc in ranked_documents]
    dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_scores))
    ndcg = dcg / idcg if idcg > 0 else 0
    
    # Add a bonus for improving over the initial ranking
    initial_ndcg = calculate_ndcg(feedback_agent, query, ranked_documents[::-1])  # Assume worst initial ranking
    improvement = ndcg - initial_ndcg
    
    return ndcg + max(0, improvement)  # Ensure non-negative reward

def calculate_ndcg(feedback_agent, query, documents):
    """
    This is a helper function to calculate the Normalized DCG (NDCG) for a given query and documents.
    """
    relevance_scores = [feedback_agent.predict_relevance(query, doc) for doc in documents]
    dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_scores))
    return dcg / idcg if idcg > 0 else 0