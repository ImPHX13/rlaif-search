# RL-Powered Search Engine Optimization

This repository implements a Reinforcement Learning (RL) framework to optimize search engine results using Proximal Policy Optimization (PPO). The system integrates with Elasticsearch to dynamically improve search rankings based on real-time feedback from a feedback agent.

## Overview

The project consists of the following key components:

1. **Search Environment**: Custom environment that simulates the search process, allowing an RL agent to interact with it by re-ranking search results.
2. **BM25 Search Engine**: Utilizes Elasticsearch to index documents and retrieve search results based on the BM25 ranking algorithm.
3. **Feedback Agent**: BERT-based model that evaluates the relevance of search results and provides feedback to the RL agent.
4. **Reinforcement Learning Agent**: PPO-based agent trained to learn optimal ranking strategies based on the feedback received.

## Installation

To set up the project, ensure you have Python 3.10 or higher installed. Then, clone the repository and install the required packages:

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
```

## Setup Elasticsearch

Before running the project, you need to set up Elasticsearch. Follow these steps:

1. Install Elasticsearch on your machine. You can find installation instructions on the [Elasticsearch website](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html).
2. Start the Elasticsearch service.
3. Run the following command to index the sample documents:

```bash
python -m search_engine.elasticsearch_setup
```

This will create an index named `documents` and load sample documents from `data/documents.json`. These documents are only a sample and do not represent the entire corpus of documents that will be used for training or evaluation.

## Running the Training

To train the RL agent, execute the following command:

```bash
python main.py
```

This will set up the environment, initialize the feedback agent, and start the training process using a predefined set of queries. The agent will learn to optimize the ranking of search results over multiple episodes.

## Evaluation

After training, the agent will evaluate its performance on a separate set of test queries. The results will be printed to the console, including the rankings from both the BM25 algorithm and the RL agent, along with their respective relevance scores and evaluation metrics (NDCG and Precision@K).

## Key Components

- **Search Environment**: Located in `rl_agent/environment.py`, this class defines the interaction between the RL agent and the search results.
- **BM25 Search**: Implemented in `search_engine/bm25_search.py`, this function retrieves documents based on the BM25 ranking.
- **Feedback Agent**: Found in `feedback_model/feedback_agent.py`, this class predicts the relevance of documents based on the query.
- **PPO Agent**: The RL agent is defined in `rl_agent/ppo_agent.py`, which implements the PPO algorithm for training.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.