import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils.ReplayMemory import Transition
import numpy as np

class AdvantageActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=20):
        super(AdvantageActorCritic, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # TODO: try use same hidden layers later
        self.actor_network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(hidden_dim, self.output_dim),
            nn.Softmax(dim=-1)
        )

        self.critic_network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.8),
            nn.Linear(hidden_dim, 1)
        )

        for m in self.modules():
            # in case add more modules later
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        self.saved_log_probs = []
        self.saved_rewards = []
        self.saved_values = []
        self.saved_entropy = []

    def forward(self, x):
        policy = self.actor_network(x)
        value = self.critic_network(x)
        return policy, value
    
    def get_action(self, state):
        policy, value = self.forward(state)
        m = Categorical(policy)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        self.saved_values.append(value)
        self.saved_entropy.append(m.entropy().mean().view(-1))
        return action.item()

    def add_saved_reward(self, reward):
        self.saved_rewards.append(reward)
    
    def clear_buffer(self):
        del self.saved_rewards[:]
        del self.saved_log_probs[:]
        del self.saved_values[:]
        del self.saved_entropy[:]

    def train_trajectory(self, last_state, optimiser, gamma=0.9, verbose=False):
        # calculate returns
        R = self.critic_network(last_state)
        returns = []
        for r in self.saved_rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).detach()
        
        # calculate loss
        log_probs   = torch.cat(self.saved_log_probs)
        # returns     = torch.cat(returns).detach()
        values      = torch.cat(self.saved_values)
        entropy     = torch.cat(self.saved_entropy).sum()
        advantage   = returns - values
        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss        = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        if verbose:
            print('[REINFORCE train loss]', loss.to(torch.device("cpu")).unsqueeze(0).detach().numpy()[0])

        # back propogate loss
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 50)
        optimiser.step()

        # clear buffer
        self.clear_buffer()
