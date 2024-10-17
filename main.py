from search_engine.elasticsearch_setup import setup_elasticsearch
from search_engine.bm25_search import bm25_search
from feedback_model.feedback_agent import FeedbackAgent
from rl_agent.environment import SearchEnvironment
from rl_agent.training_loop import train_ppo
from evaluation.metrics import ndcg, precision_at_k
import torch
import numpy as np

def main():
    # Setup
    es = setup_elasticsearch(purge=True)
    if es is None:
        print("Failed to set up Elasticsearch. Exiting.")
        return

    feedback_agent = FeedbackAgent()
    env = SearchEnvironment(es, feedback_agent)
    
    # Training
    training_queries = [
        "artificial intelligence", "machine learning", "deep learning",
        "neural networks", "natural language processing", "computer vision",
        "theory of relativity", "quantum computing", "blockchain",
        "renewable energy", "climate change", "genetic algorithms",
        "CRISPR", "dark matter", "fusion power", "quantum entanglement",
        "human microbiome", "anthropocene", "cybersecurity", "nanotechnology",
        "virtual reality", "internet of things", "big data", "robotics",
        "3D printing", "gene editing", "autonomous vehicles", "space exploration",
        "artificial neural networks", "machine vision"
    ]
    
    print("Training on comprehensive query set...")
    trained_agent = train_ppo(env, num_episodes=1000, batch_size=64, training_queries=training_queries)
    
    # Evaluation
    test_queries = [
        "artificial intelligence", "theory of relativity", "machine learning",
        "climate change", "renaissance art", "quantum physics",
        "cybersecurity", "evolution", "nanotechnology", "virtual reality"
    ]
    for query in test_queries:
        bm25_results = bm25_search(es, query, top_k=10)
        
        if not bm25_results:
            print(f"No results found for query: {query}")
            continue
        
        state = env.reset(query)
        if isinstance(state, float) or len(state) == 0:
            print(f"No valid state for query: {query}")
            continue
        
        print(f"Evaluation state shape: {state.shape}")
        
        # Apply RL agent's actions
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        actions, _, _ = trained_agent.get_action(state_tensor)
        state, _, _, _ = env.step(actions.squeeze().numpy())
        
        rl_results = env.get_current_ranking()
        
        bm25_relevance = [feedback_agent.predict_relevance(query, doc) for doc in bm25_results]
        rl_relevance = [feedback_agent.predict_relevance(query, doc) for doc in rl_results]
        
        print(f"Query: {query}")
        print(f"BM25 ranking: {bm25_results[:5]}")
        print(f"RL ranking: {rl_results[:5]}")
        print(f"BM25 relevance scores: {bm25_relevance}")
        print(f"RL relevance scores: {rl_relevance}")
        print(f"BM25 NDCG: {ndcg(bm25_relevance):.4f}")
        print(f"RL NDCG: {ndcg(rl_relevance):.4f}")
        print(f"BM25 P@5: {precision_at_k(bm25_relevance, 5):.4f}")
        print(f"RL P@5: {precision_at_k(rl_relevance, 5):.4f}")
        print()

if __name__ == "__main__":
    main()