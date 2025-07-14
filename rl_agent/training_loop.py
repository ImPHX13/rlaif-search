import torch
from torch.optim import Adam
from rl_agent.ppo_agent import PPOAgent
import random
from tqdm import tqdm
import time
import numpy as np

def train_ppo(env, num_episodes, batch_size, training_queries):
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    
    agent = PPOAgent(input_dim, output_dim)
    optimizer = Adam(agent.parameters(), lr=3e-4)
    
    best_reward = float('-inf')
    no_improvement_count = 0
    
    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        query = random.choice(training_queries)
        state = env.reset(query)
        
        # print(f"Episode {episode}, Initial state shape: {state.shape}")  # Debug print
        
        episode_reward = 0
        states, actions, log_probs, values, rewards = [], [], [], [], []
        
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state)
            
            epsilon = max(0.05, 1.0 - episode / (num_episodes * 0.8))
            action, log_prob, value = agent.get_action(state_tensor, epsilon)
            next_state, reward, done, _ = env.step(action.item())
            
            # print(f"Episode {episode}, Step state shape: {next_state.shape}")  # Debug print
            
            states.append(state)
            actions.append(action.item())
            log_probs.append(log_prob.item())
            values.append(value.item())
            rewards.append(reward)
            
            episode_reward += reward
            state = next_state
        
        # Compute returns and advantages
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        running_return = 0
        running_advantage = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + 0.99 * running_return
            running_advantage = rewards[t] + 0.99 * running_advantage - values[t]
            returns[t] = running_return
            advantages[t] = running_advantage
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        old_log_probs = torch.FloatTensor(np.array(log_probs))
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # PPO update
        for _ in range(10):  # Increase number of optimization epochs
            new_action_logits, new_values = agent(states)
            new_log_probs = torch.log_softmax(new_action_logits, dim=-1)
            new_log_probs = new_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            new_values = new_values.squeeze(1)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (returns - new_values).pow(2).mean()
            
            loss = actor_loss + 0.5 * critic_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= 50:  # Early stopping
            print(f"Early stopping at episode {episode}")
            break
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Query: {query}, Reward: {episode_reward:.4f}, Epsilon: {epsilon:.4f}")
    
    return agent