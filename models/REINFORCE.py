import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random

from utils.ReplayMemory import Transition


class REINFORCE(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=60):
        super(REINFORCE, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.policy_network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(hidden_dim, self.output_dim),
            nn.Softmax(dim=-1))

        for m in self.modules():
            # in case add more modules later
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        self.saved_log_probs = []
        self.rewards = []
        self.eps = np.finfo(np.float32).eps.item()

    def forward(self, x):
        return self.policy_network(x)

    def get_action(self, state, epsilon=0.1):
        if random.random() > epsilon:
            probs = self.forward(state)
            # for debugging
            # print('PROBS', probs.unsqueeze(0).detach().numpy()[0])
            m = Categorical(probs)
            action = m.sample()
            self.saved_log_probs.append(m.log_prob(action))
            return action.item()
        else:
            probs = self.forward(state)
            action = random.randint(0, self.output_dim - 1)
            log_prob = torch.log(probs[0, action]).view(-1)
            self.saved_log_probs.append(log_prob)
            return action

    def train_episode(self, optimiser, gamma=0.9, verbose=False):
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        optimiser.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        if verbose:
            print('[REINFORCE train loss]', policy_loss.to(torch.device("cpu")).unsqueeze(0).detach().numpy()[0])
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 50)
        optimiser.step()
        del self.rewards[:]
        del self.saved_log_probs[:]


class REINFORCE_BASELINE(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=60):
        super(REINFORCE_BASELINE, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.policy_network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(hidden_dim, self.output_dim),
            nn.Softmax(dim=-1))

        self.value_network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(hidden_dim, 1))
        
        for m in self.modules():
            # in case add more modules later
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []
        self.eps = np.finfo(np.float32).eps.item()

    def forward(self, x):
        policy = self.policy_network(x)
        value = self.value_network(x)
        return policy, value

    def get_action(self, state):
        probs, value = self.forward(state)
        print('PROBS', probs.unsqueeze(0).detach().numpy()[0])
        print('VALUES', value.unsqueeze(0).detach().numpy()[0])
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        self.saved_values.append(value)
        return action.item()

    def train_episode(self, optimiser, gamma=0.9, verbose=False):
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        for v, r in zip(self.saved_values, returns):
            r = r - v

        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        optimiser.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        if verbose:
            print('[REINFORCE train loss]', policy_loss.to(torch.device("cpu")).unsqueeze(0).detach().numpy()[0])
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 50)
        optimiser.step()
        del self.rewards[:]
        del self.saved_log_probs[:]

