"""
Cliff Walking Environment
A 4x12 GridWorld where the agent must navigate from start (bottom-left)
to goal (bottom-right) while avoiding the cliff (bottom row, middle cells).
"""

import numpy as np

class CliffWalkingEnv:
    """
    Cliff Walking Environment (4x12 GridWorld)
    
    Layout:
        Row 0 (top)    : normal cells
        Row 1          : normal cells
        Row 2          : normal cells
        Row 3 (bottom) : S [CLIFF...CLIFF] G
    
    Actions: 0=Up, 1=Down, 2=Left, 3=Right
    """
    
    def __init__(self, height=4, width=12):
        self.height = height
        self.width = width
        self.n_states = height * width
        self.n_actions = 4  # Up, Down, Left, Right
        
        # Define start and goal positions (row, col)
        self.start = (height - 1, 0)            # bottom-left
        self.goal = (height - 1, width - 1)     # bottom-right
        
        # Cliff positions: bottom row, columns 1 to width-2
        self.cliff = set()
        for col in range(1, width - 1):
            self.cliff.add((height - 1, col))
        
        # Action mapping: (row_delta, col_delta)
        self.action_deltas = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1),   # Right
        }
        
        self.action_names = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        
        self.state = None
        self.reset()
    
    def reset(self):
        """Reset environment to start position."""
        self.state = self.start
        return self._state_to_index(self.state)
    
    def step(self, action):
        """
        Take an action and return (next_state_index, reward, done).
        """
        row, col = self.state
        dr, dc = self.action_deltas[action]
        
        # Apply action with boundary clipping
        new_row = max(0, min(self.height - 1, row + dr))
        new_col = max(0, min(self.width - 1, col + dc))
        new_pos = (new_row, new_col)
        
        # Check cliff
        if new_pos in self.cliff:
            self.state = self.start
            return self._state_to_index(self.start), -100, False
        
        # Check goal
        self.state = new_pos
        if new_pos == self.goal:
            return self._state_to_index(new_pos), -1, True
        
        return self._state_to_index(new_pos), -1, False
    
    def _state_to_index(self, state):
        row, col = state
        return row * self.width + col
    
    def _index_to_state(self, index):
        row = index // self.width
        col = index % self.width
        return (row, col)
    
    def is_cliff(self, state_index):
        return self._index_to_state(state_index) in self.cliff
    
    def get_grid_position(self, state_index):
        return self._index_to_state(state_index)
    
    def __repr__(self):
        grid = []
        for row in range(self.height):
            line = []
            for col in range(self.width):
                pos = (row, col)
                if pos == self.state:
                    line.append(' A ')
                elif pos == self.start:
                    line.append(' S ')
                elif pos == self.goal:
                    line.append(' G ')
                elif pos in self.cliff:
                    line.append(' C ')
                else:
                    line.append(' . ')
            grid.append(''.join(line))
        return '\n'.join(grid)
