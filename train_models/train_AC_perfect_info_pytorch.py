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


device = torch.device("cpu")
lr = 1e-3
test_interval = 20


def convert_state2onehot(state, state_dim):
    state_min = -1 * np.ones(state_dim, dtype=int)
    state_max = 6 * np.ones(state_dim, dtype=int)
    state = np.clip(state, state_min, state_max)
    state_idx = state + np.ones(state_dim, dtype=int)
    state_one_hot = np.zeros(8 * state_dim)
    for i in range(state_dim):
        state_one_hot[state_idx[i] + 8 * i] = 1.
    state_one_hot = torch.Tensor(state_one_hot).to(device).unsqueeze(0)
    return state_one_hot


def train_ActorCritic_perfect_info(env):
    model = ActorCritic(8 * env.num_food_types, env.num_actions)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    for e in range(6000):
        state = env.reset()
        # print('reset game:', env.warehouse_num, env.knapsack_num)
        expected = env.expected_num - env.warehouse_num
        state = expected - state

        terminate = False
        running_loss = 0.
        running_steps = 0.
        # create an episode
        while not terminate:
            # print(state, end=' | ')
            state_onehot = convert_state2onehot(state, state_dim=env.num_food_types)
            # print(state_onehot)
            action = model.get_action(state_onehot, epsilon=0.01).numpy().tolist()
            # print(action)
            next_state, reward, terminate = env.step(action)
            # print(reward)
            next_state = expected - next_state
            next_state_onehot = convert_state2onehot(next_state, state_dim=env.num_food_types)

            mask = 0 if terminate else 1

            action_one_hot = torch.zeros(env.num_actions)
            action_one_hot[action] = 1
            transition = [state_onehot, next_state_onehot, action, reward, mask]

            state = next_state
            loss = model.train_model(model, transition, optimizer)
            running_loss += loss
            running_steps += 1

        # print('[loss]episode %d: %.2f' % (e, running_loss / running_steps))

        if e % test_interval == 0 and (not e == 0):
            scores = []
            model.eval()
            for i in range(100):
                terminate = False
                state = env.reset()
                expected = env.expected_num - env.warehouse_num
                state = expected - state
                score = 0
                while not terminate:
                    state_onehot = convert_state2onehot(state, state_dim=env.num_food_types)
                    action = model.get_action(state_onehot, epsilon=0.001).numpy().tolist()
                    next_state, reward, terminate = env.step(action)
                    next_state = expected - next_state
                    score += reward
                    state = next_state
                scores.append(score)
            model.train()
            print('[test score]episode %d: %.2f' % (e, np.mean(np.asarray(scores))))


if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read('../game.conf')
    env = FoodGatherEnv_GPU(int(cf.defaults()['num_food_types']),
                            int(cf.defaults()['max_capacity']))
    train_ActorCritic_perfect_info(env)

