"""
Q-Learning and SARSA Algorithm Implementations
"""

import numpy as np


class QLearning:
    """
    Q-Learning: Off-policy TD control algorithm.
    
    Update rule:
        Q(s, a) <- Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]
    
    The target uses the GREEDY action in next state (regardless of actual behavior).
    """
    
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate (ε-greedy)
        
        # Initialize Q-table with zeros
        self.Q = np.zeros((n_states, n_actions))
    
    def choose_action(self, state):
        """ε-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Q-Learning update (off-policy):
        Uses max Q(s', a') regardless of what action will actually be taken.
        """
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])
        
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        return abs(td_error)
    
    def get_policy(self):
        """Return greedy policy: argmax Q(s, a) for each state."""
        return np.argmax(self.Q, axis=1)
    
    def reset(self):
        """Reset Q-table."""
        self.Q = np.zeros((self.n_states, self.n_actions))


class SARSA:
    """
    SARSA: On-policy TD control algorithm.
    
    Update rule:
        Q(s, a) <- Q(s, a) + α * [r + γ * Q(s', a') - Q(s, a)]
    
    Where a' is the ACTUAL action chosen (by the same ε-greedy policy).
    This makes SARSA sensitive to the exploration behavior.
    """
    
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-table with zeros
        self.Q = np.zeros((n_states, n_actions))
    
    def choose_action(self, state):
        """ε-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, next_action, done):
        """
        SARSA update (on-policy):
        Uses the ACTUAL next action chosen by the policy.
        """
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.Q[next_state, next_action]
        
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        return abs(td_error)
    
    def get_policy(self):
        """Return greedy policy: argmax Q(s, a) for each state."""
        return np.argmax(self.Q, axis=1)
    
    def reset(self):
        """Reset Q-table."""
        self.Q = np.zeros((self.n_states, self.n_actions))
