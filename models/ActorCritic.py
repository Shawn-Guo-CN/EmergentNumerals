#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/20 12:44
# @Author  : Shawn
# @File    : ActorCritic.py
# @Classes :
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=20):
        super(ActorCritic, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc_hidden = nn.Linear(input_dim, hidden_dim)
        self.fc_actor = nn.Linear(hidden_dim, output_dim)
        self.fc_critic = nn.Linear(hidden_dim, 1)

        for m in self.modules():
            # in case add more modules later
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = F.relu(self.fc_hidden(x))
        policy = F.softmax(self.fc_actor(x), dim=-1)
        value = self.fc_critic(x)
        return policy, value

    @classmethod
    def train_model(cls, model, transition, optimizer, gamma=1.0):
        state, next_state, action, reward, mask = transition

        policy, value = model(state)
        policy, value = policy.view(-1, model.output_dim), value.view(-1, 1)
        _, next_value = model(next_state)
        next_value = next_value.view(-1, 1)

        target_plus_value = reward + mask * gamma * next_value[0]
        target = target_plus_value - value[0]

        log_policy = torch.log(policy[0])[action]
        loss_policy = -log_policy * target.detach()
        loss_value = F.mse_loss(target_plus_value.detach(), value[0])

        loss = (loss_policy + loss_value).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, state, epsilon):
        policy, _ = self.forward(state)
        m = Categorical(policy)
        if np.random.rand() > epsilon:
            action = m.sample()
        else:
            action = torch.randint(high=self.output_dim, size=(1,))
        return action[0]

