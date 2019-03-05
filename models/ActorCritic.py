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
from utils.ReplayMemory import Transition


class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=20):
        super(ActorCritic, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc_hidden = nn.Linear(input_dim, hidden_dim)
        self.fc_actor = nn.Linear(hidden_dim, output_dim)
        self.fc_critic = nn.Linear(hidden_dim, 1)
        self.fc_critic_target = nn.Linear(hidden_dim, 1)

        for m in self.modules():
            # in case add more modules later
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        self.update_counter = 0
        self.update_interval = 1

    def forward(self, x):
        x = F.relu(self.fc_hidden(x))
        policy = F.softmax(self.fc_actor(x), dim=-1)
        value = self.fc_critic(x)
        return policy, value

    def eval_target_value(self, x):
        x = F.relu(self.fc_hidden(x))
        value = self.fc_critic_target(x)
        return value

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

    def train_(self, transitions, optimiser, gamma=1.0):
        if self.update_counter % self.update_interval == 0:
            self.fc_critic_target.load_state_dict(self.fc_critic.state_dict())
            for param in self.fc_critic_target.parameters():
                param.requires_grad = False

        self.update_counter += 1

        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action).view(-1, 1)
        reward_batch = torch.cat(batch.reward).view(-1, 1)
        non_final_mask = torch.tensor(batch.mask, device=state_batch.device, dtype=torch.uint8)

        state_policies, state_values = self.forward(state_batch)
        state_policies = state_policies.gather(1, action_batch)
        next_state_values = torch.zeros((len(transitions),1), device=state_batch.device)
        next_state_values[non_final_mask] = self.eval_target_value(next_state_batch[non_final_mask]).detach()

        expected_state_values = reward_batch + (gamma * next_state_values)
        loss_policy = - torch.log(state_policies) * (expected_state_values - state_values)
        loss_value = F.mse_loss(expected_state_values, state_values)
        loss = (loss_policy + loss_value).mean()
        optimiser.zero_grad()
        loss.backward()
        for param in self.fc_actor.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.fc_hidden.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.fc_critic.parameters():
            param.grad.data.clamp_(-1, 1)
        optimiser.step()

        return loss

    def get_action(self, state, epsilon):
        policy, _ = self.forward(state)
        m = Categorical(policy)
        if np.random.rand() > epsilon:
            action = m.sample()
        else:
            action = torch.randint(high=self.output_dim, size=(1,), device=state.device)
        return action
