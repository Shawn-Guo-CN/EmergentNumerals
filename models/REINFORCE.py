import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils.ReplayMemory import Transition


class REINFORCE(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=60):
        super(REINFORCE, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        for m in self.modules():
            # in case add more modules later
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        self.saved_log_probs = []
        self.rewards = []
        self.eps = np.finfo(np.float32).eps.item()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        value = F.softmax(self.fc2(x), dim=1)
        return value

    def get_action(self, state):
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample(probs)
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def train(self, optimiser, gamma=0.9):
        R = 0
        policy_loss = []

        returns = []
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        optimiser.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimiser.step()
        del self.rewards[:]
        del self.saved_log_probs[:]

