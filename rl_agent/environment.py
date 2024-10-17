import gym
from gym import spaces
import numpy as np
from search_engine.bm25_search import bm25_search
from feedback_model.reward_function import calculate_reward
from evaluation.metrics import ndcg

class SearchEnvironment(gym.Env):
    """
    A gym environment for search engine optimization.
    
    Attributes:
        es (str): Elasticsearch instance.
        feedback_agent (object): Feedback agent for predicting document relevance.
        top_k (int): Number of top documents to consider.
        action_space (gym.Space): Discrete space for actions (document indices).
        observation_space (gym.Space): Box space for observations (document relevance).
        query (str): Current search query.
        documents (list): List of documents retrieved for the query.
        initial_relevance (np.array): Initial relevance scores for the documents.
        current_ranking (list): Current ranking of documents.
        initial_ndcg (float): Initial NDCG score.
    """
    def __init__(self, es, feedback_agent, top_k=10):
        super().__init__()
        self.es = es
        self.feedback_agent = feedback_agent
        self.top_k = top_k
        
        self.action_space = spaces.Discrete(top_k)
        self.observation_space = spaces.Box(low=0, high=1, shape=(top_k,), dtype=np.float32)
        
        self.query = None
        self.documents = None
        self.initial_relevance = None
        self.current_ranking = None
        self.initial_ndcg = None
    
    def reset(self, query):
        """
        Resets the environment with a new query.
        
        Args:
            query (str): The search query.
        
        Returns:
            np.array: The initial relevance scores for the documents.
        """
        self.query = query
        self.documents = bm25_search(self.es, query, self.top_k)
        if not self.documents:
            return np.zeros(self.top_k)
        self.initial_relevance = np.array([self.feedback_agent.predict_relevance(query, doc) for doc in self.documents])
        # Pad the initial_relevance to always have top_k elements
        self.initial_relevance = np.pad(self.initial_relevance, (0, max(0, self.top_k - len(self.initial_relevance))), 'constant')
        self.current_ranking = list(range(len(self.documents)))
        self.initial_ndcg = ndcg(self.initial_relevance)
        return self.initial_relevance
    
    def step(self, action):
        """
        Takes an action in the environment.
        
        Args:
            action (int): The index of the document to move to the top.
        
        Returns:
            tuple: A tuple containing the new relevance scores, reward, done flag, and info.
        """
        if not self.documents or action >= len(self.current_ranking):
            return self.initial_relevance, 0, True, {}
        
        # Move the selected document to the top
        self.current_ranking.insert(0, self.current_ranking.pop(action))
        
        reranked_documents = [self.documents[i] for i in self.current_ranking]
        new_relevance = np.array([self.feedback_agent.predict_relevance(self.query, doc) for doc in reranked_documents])
        # Pad new_relevance to always have top_k elements
        new_relevance = np.pad(new_relevance, (0, max(0, self.top_k - len(new_relevance))), 'constant')
        
        new_ndcg = ndcg(new_relevance)
        reward = new_ndcg - self.initial_ndcg
        
        done = (action == 0)  # Episode ends if the agent chooses the top document
        
        return new_relevance, reward, done, {}

    def get_current_ranking(self):
        """
        Returns the current ranking of documents.
        
        Returns:
            list: A list of documents in their current ranking order.
        """
        return [self.documents[i] for i in self.current_ranking]