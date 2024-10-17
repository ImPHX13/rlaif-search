import numpy as np

def ndcg(relevance_scores, k=None):
    if k is None:
        k = len(relevance_scores)
    
    dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevance_scores[:k]))
    idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(sorted(relevance_scores, reverse=True)[:k]))
    
    return dcg / idcg if idcg > 0 else 0

def precision_at_k(relevance_scores, k):
    return sum(1 for score in relevance_scores[:k] if score > 0) / k