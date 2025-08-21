import torch
from torch.optim import Adam
from rl_agent.ppo_agent import PPOAgent
import random
from tqdm import tqdm
import numpy as np


def train_ppo(env, num_episodes, batch_size, training_queries, seed: int = 42):
    """Train a :class:`~rl_agent.ppo_agent.PPOAgent` with PPO.

    Parameters
    ----------
    env : gym.Env
        Environment providing ``reset`` and ``step`` interfaces.
    num_episodes : int
        Number of episodes to sample from the environment.
    batch_size : int
        Number of transitions to collect before each PPO update.
    training_queries : list
        Initial queries used to reset the environment each episode.
    seed : int, optional
        Random seed for ``random``, ``numpy`` and ``torch`` modules. Defaults to
        ``42``.

    Returns
    -------
    PPOAgent
        The trained agent.

    Notes
    -----
    Transitions are accumulated across episodes until ``batch_size`` is
    reached. A PPO update is then performed and the buffers are cleared for the
    next batch.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    print(f"Input dim: {input_dim}, Output dim: {output_dim}")

    agent = PPOAgent(input_dim, output_dim)
    optimizer = Adam(agent.parameters(), lr=3e-4)

    best_reward = float("-inf")
    no_improvement_count = 0

    states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []

    def ppo_update():
        """Run a PPO update using the collected transitions."""
        returns = []
        running_return = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            running_return = r + 0.99 * running_return * (1 - d)
            returns.insert(0, running_return)
        returns_arr = np.array(returns)
        advantages_arr = returns_arr - np.array(values)

        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(np.array(actions))
        old_log_probs_t = torch.FloatTensor(np.array(log_probs))
        returns_t = torch.FloatTensor(returns_arr)
        advantages_t = torch.FloatTensor(advantages_arr)

        for _ in range(10):  # Increase number of optimization epochs
            new_action_logits, new_values = agent(states_t)
            new_log_probs = torch.log_softmax(new_action_logits, dim=-1)
            new_log_probs = new_log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            new_values = new_values.squeeze(1)

            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages_t
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (returns_t - new_values).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        states.clear()
        actions.clear()
        log_probs.clear()
        values.clear()
        rewards.clear()
        dones.clear()

    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        query = random.choice(training_queries)
        state = env.reset(query)

        episode_reward = 0
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state)

            epsilon = max(0.05, 1.0 - episode / (num_episodes * 0.8))
            action, log_prob, value = agent.get_action(state_tensor, epsilon)
            next_state, reward, done, _ = env.step(action.item())

            states.append(state)
            actions.append(action.item())
            log_probs.append(log_prob.item())
            values.append(value.item())
            rewards.append(reward)
            dones.append(done)

            episode_reward += reward
            state = next_state

            if len(states) >= batch_size:
                ppo_update()

        if episode_reward > best_reward:
            best_reward = episode_reward
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= 50:  # Early stopping
            print(f"Early stopping at episode {episode}")
            break

        if episode % 10 == 0:
            print(
                f"Episode {episode}, Query: {query}, Reward: {episode_reward:.4f}, "
                f"Epsilon: {epsilon:.4f}"
            )

    if states:
        ppo_update()

    return agent
