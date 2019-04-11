#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/11 17:39
# @Author  : Shawn
# @File    : train_AC_perfect_info_pytorch_multigame.py
# @Goal    : train AC model with pytorch version environment

import configparser
import torch.optim as optim

from models.ActorCritic import *
from env import FoodGatherEnv_GPU
from utils.ReplayMemory import ReplayMemory


device = torch.device("cuda")
lr = 1e-3
test_interval = 20
decay_interval = 50
batch_size = 64
replay_pool = ReplayMemory(batch_size)
torch.manual_seed(1234)


def convert_state2onehot(state, state_dim):
    state_index = state + torch.ones_like(state)
    state_one_hot = torch.zeros((len(state), state_dim),
                                device=state.device).scatter_(1, state_index.view(-1, 1), 1).view(1,-1)
    return state_one_hot


def train_ActorCritic_perfect_info():
    cf = configparser.ConfigParser()
    cf.read('../game.conf')

    envs = []
    for _ in range(batch_size):
        env = FoodGatherEnv_GPU(int(cf.defaults()['num_food_types']),
                                int(cf.defaults()['max_capacity']))
        env.to(device)
        envs.append(env)

    state_dim = envs[0].max_capacity + 2
    model = ActorCritic(envs[0].num_food_types * state_dim, envs[0].num_actions)

    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    epsilon = 0.05

    for e in range(6000):
        if e % decay_interval == 0 and not (e == 0):
            epsilon *= 0.1

        states = []
        for idx, env in enumerate(envs):
            states.append((idx, env.reset()))

        keep_going = True
        while keep_going:
            keep_going = False

            next_states = []
            replay_pool.reset()
            for (idx, state) in states:
                expected = envs[idx].expected_num - envs[idx].warehouse_num
                state = expected - state
                state.clamp_(-1, envs[idx].max_capacity)
                state_one_hot = convert_state2onehot(state, state_dim=state_dim)

                action = model.get_action(state_one_hot, epsilon=epsilon)

                next_state, reward, terminate = envs[idx].step(action[0])
                next_state = expected - next_state
                next_state.clamp_(-1, envs[idx].max_capacity)
                next_state_one_hot = convert_state2onehot(next_state, state_dim=state_dim)

                mask = 0 if terminate else 1

                if not terminate:
                    keep_going = True
                    replay_pool.push(state_one_hot, next_state_one_hot, action, reward, mask)
                    next_states.append((idx, next_state))

            states = next_states

            if not len(replay_pool) == 0:
                loss = model.train_(replay_pool.pop_all(), optimizer)
            # print(loss)

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
    train_ActorCritic_perfect_info()
