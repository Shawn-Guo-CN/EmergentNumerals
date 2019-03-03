#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 27/02/19 15:20
# @Author  : Shawn
# @File    : train_AC_perfect_info_pytorch.py
# @Goal    : train AC model with pytorch version environment

import configparser
import torch.optim as optim

from models.ActorCritic import *
from env import FoodGatherEnv_GPU
from utils.ReplayMemory import ReplayMemory


device = torch.device("cuda")
lr = 1e-3
test_interval = 20
batch_size = 32
replay_pool = ReplayMemory(1000)


def convert_state2onehot(state, state_dim):
    state_index = state + torch.ones_like(state)
    state_one_hot = torch.zeros((len(state), state_dim),
                                device=state.device).scatter_(1, state_index.view(-1, 1), 1).view(1,-1)
    return state_one_hot


def train_ActorCritic_perfect_info(env):
    state_dim = env.max_capacity + 2
    model = ActorCritic(state_dim * env.num_food_types, env.num_actions)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    for e in range(6000):
        state = env.reset()
        # print('-----------------------------------------------------------------------------------------------------')
        # print('reset game:', env.warehouse_num, env.knapsack_num)
        expected = env.expected_num - env.warehouse_num
        # print('expected num:', expected)
        # print(state, end='|')
        state = expected - state
        # print('transferred:', state)
        state.clamp_(-1, env.max_capacity)
        # print('state clamped', state)

        terminate = False
        running_loss = 0.
        running_steps = 0.
        # create an episode
        while not terminate:
            state_one_hot = convert_state2onehot(state, state_dim=state_dim)
            # print(state_one_hot)
            action = model.get_action(state_one_hot, epsilon=0.01)
            # print('action:', action)
            next_state, reward, terminate = env.step(action[0])
            next_state = expected - next_state
            next_state.clamp_(-1, env.max_capacity)
            # print(reward)
            # print('----------------------------------------------')
            next_state_one_hot = convert_state2onehot(next_state, state_dim=state_dim)

            mask = 0 if terminate else 1

            # action_one_hot = torch.zeros((env.num_actions,), device=state.device).scatter_()
            # action_one_hot[action] = 1
            # transition = [state_onehot, next_state_onehot, action, reward, mask]
            replay_pool.push(state_one_hot, next_state_one_hot, action, reward, mask)

            state = next_state
            if len(replay_pool) >= batch_size:
                loss = model.train_(replay_pool.sample(batch_size), optimizer)
                running_loss += loss
                running_steps += 1

                print('[loss]episode %d: %.2f' % (e, running_loss / running_steps))

        if e % test_interval == 0 and (not e == 0):
            scores = []
            model.eval()
            for i in range(100):
                terminate = False
                state = env.reset()
                expected = env.expected_num - env.warehouse_num
                score = 0
                while not terminate:
                    state = expected - state
                    state.clamp_(-1, env.max_capacity)
                    state_one_hot = convert_state2onehot(state, state_dim=state_dim)
                    action = model.get_action(state_one_hot, epsilon=0.001)
                    next_state, reward, terminate = env.step(action[0])
                    score += reward
                    state = next_state
                scores.append(score)
            model.train()
            print('[test score]episode %d: %.2f'
                  % (e, torch.cat(scores, 0).mean().to(torch.device("cpu")).unsqueeze(0).numpy()[0]))


if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read('../game.conf')
    env = FoodGatherEnv_GPU(int(cf.defaults()['num_food_types']),
                            int(cf.defaults()['max_capacity']))
    env.to(device)
    train_ActorCritic_perfect_info(env)

