"""
Training loops for Q-Learning and SARSA on Cliff Walking environment.
"""

import numpy as np
from environment import CliffWalkingEnv
from algorithms import QLearning, SARSA


def train_qlearning(env, agent, n_episodes=500, seed=42):
    """
    Train Q-Learning agent.
    
    Returns:
        episode_rewards: list of total reward per episode
        episode_td_errors: list of mean TD error per episode
    """
    np.random.seed(seed)
    episode_rewards = []
    episode_td_errors = []

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        td_errors = []
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            td_err = agent.update(state, action, reward, next_state, done)
            td_errors.append(td_err)

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        episode_td_errors.append(np.mean(td_errors))

    return episode_rewards, episode_td_errors


def train_sarsa(env, agent, n_episodes=500, seed=42):
    """
    Train SARSA agent.
    
    SARSA requires choosing next_action BEFORE the update,
    since it is on-policy and must use the actual next action.
    
    Returns:
        episode_rewards: list of total reward per episode
        episode_td_errors: list of mean TD error per episode
    """
    np.random.seed(seed)
    episode_rewards = []
    episode_td_errors = []

    for ep in range(n_episodes):
        state = env.reset()
        action = agent.choose_action(state)  # Choose first action
        total_reward = 0
        td_errors = []
        done = False

        while not done:
            next_state, reward, done = env.step(action)
            next_action = agent.choose_action(next_state)  # Choose NEXT action NOW

            td_err = agent.update(state, action, reward, next_state, next_action, done)
            td_errors.append(td_err)

            state = next_state
            action = next_action  # Carry forward to next step
            total_reward += reward

        episode_rewards.append(total_reward)
        episode_td_errors.append(np.mean(td_errors))

    return episode_rewards, episode_td_errors


def smooth(data, window=10):
    """Apply moving average smoothing."""
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def extract_path(env, agent, max_steps=200):
    """
    Extract the greedy path (no exploration) using the learned policy.
    Returns list of (row, col) positions.
    """
    env_copy = CliffWalkingEnv(env.height, env.width)
    state = env_copy.reset()
    path = [env_copy.get_grid_position(state)]
    visited = {state}

    for _ in range(max_steps):
        action = np.argmax(agent.Q[state])
        next_state, reward, done = env_copy.step(action)
        pos = env_copy.get_grid_position(next_state)
        path.append(pos)

        if done:
            break
        # Prevent infinite loops
        if next_state in visited:
            break
        visited.add(next_state)
        state = next_state

    return path
