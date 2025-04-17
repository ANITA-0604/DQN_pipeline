import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from collections import deque
import numpy as np
import random


class DQN(nn.Module):
    def __init__(self, n_obsevations, n_actions) -> None:
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_obsevations, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, n_actions)

    def forward(self, x) -> torch.Tensor:
        x = functional.relu(self.layer1(x)) # pass data into the first layer
        x = functional.relu(self.layer2(x))
        return self.layer3(x)
    

class DQN_agent:
    def __init__(self, state_size: int, action_size: int, gamma: float=0.99, epsilon: float=1.0, lr: float = 1e-3) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory = deque(maxlen=1000) #  store past experience as (state, action, reward, next_state, done)
        # Replay memory allows the agent to break correlation between sequential experiences and improve stability.
        self.model = DQN(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr) # updates weights to minimize loss
        self.criterion = nn.MSELoss() #  loss function to measure how close predicted Q-values are to targets

    def act(self, state: np.ndarray) -> int:
        # goal: choose the action that can maximize the total reward over time
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.action_size))
            # if the random float is in the range between 0 and epsilon, then pick an action to apply randomly
            # if epsilon = 1.0, 100% random — fully exploring
		    # if epsilon = 0.1, 10% random, 90% greedy (based on Q-values)
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()
            # choosing the action that seems best based on what the model has learned.
        
    def remember(self, s: np.ndarray, a: int, r: float, s_: np.ndarray, done: bool) -> None: # (state, action, reward, next_state, done)
        self.memory.append((s, a, r, s_, done))

    def train(self, batch_size: int = 32) -> None:
        if( len(self.memory) < batch_size):
            return
    
        batch = random.sample(self.memory, batch_size)
        for s, a, r, s_, done in batch:
            s = torch.tensor(s, dtype=torch.float32)
            s_ = torch.tensor(s_, dtype=torch.float32)
            target  = r 
            if not done:
                target += self.gamma * torch.max(self.model(s_)).item()

            pred = self.model(s)[a]
            loss = loss = self.criterion(pred, torch.tensor(target, dtype=torch.float32))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() # Backpropagate and update weights










        
        