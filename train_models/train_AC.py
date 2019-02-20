#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/20 12:54
# @Author  : Shawn
# @File    : train_AC.py
# @Classes :
import configparser
import torch.optim as optim

from models.ActorCritic import *
from env import FoodGatherEnv


device = torch.device("cpu")
lr = 1e-6
test_interval = 20


def convert_state2onehot(state, m):
    state_one_hot = np.zeros(45)
    for i in range(3):
        state_one_hot[m[i] + 5 * i] = 1.
        state_one_hot[state[i] + 10 * i + 14] = 1.
    state_one_hot = torch.Tensor(state_one_hot).to(device).unsqueeze(0)
    return state_one_hot


def train_ActorCritic_perfect_message(env):
    model = ActorCritic(45, env.num_actions)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    for e in range(6000):
        state = env.reset()
        m = env.warehouse_num

        terminate = False
        running_loss = 0.
        running_steps = 0.
        # create an episode
        while not terminate:
            state_onehot = convert_state2onehot(state, m)
            action = model.get_action(state_onehot)
            next_state, reward, terminate = env.step(action)
            next_state_onehot = convert_state2onehot(next_state, m)

            mask = 0 if terminate else 1

            action_one_hot = torch.zeros(4)
            action_one_hot[action] = 1
            transition = [state_onehot, next_state_onehot, action, reward, mask]

            state = next_state
            loss = model.train_model(model, transition, optimizer)
            running_loss += loss
            running_steps += 1

        print('[loss]episode %d: %.2f' % (e, running_loss / running_steps))

        if e % test_interval == 0 and (not e == 0):
            scores = []
            model.eval()
            for i in range(100):
                terminate = False
                state = env.reset()
                m = env.warehouse_num
                score = 0
                while not terminate:
                    state_onehot = convert_state2onehot(state, m)
                    action = model.get_action(state_onehot)
                    next_state, reward, terminate = env.step(action)
                    score += reward
                    state = next_state
                scores.append(score)
            model.train()
            print('[test score]episode %d: %.2f' % (e, np.mean(np.asarray(scores))))


if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read('../game.conf')
    env = FoodGatherEnv(int(cf.defaults()['num_food_types']),
                        int(cf.defaults()['max_capacity']))
    train_ActorCritic_perfect_message(env)
