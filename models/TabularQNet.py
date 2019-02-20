#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/20 16:19
# @Author  : Shawn
# @File    : TabularQNet.py
# @Classes :
from collections import defaultdict, namedtuple
import numpy as np
import configparser
from env import FoodGatherEnv
np.random.seed(1234)

test_interval = 100
decay_interval = 10000
Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward'))
state_min = -1 * np.ones(3, dtype=int)
state_max = 5 * np.ones(3, dtype=int)
global epsilon
epsilon = 0.1
global alpha
alpha = 0.5


class QNet(object):
    def __init__(self, num_actions=4, epsilon=1e-2):
        self.epsilon = epsilon
        self.num_actions = num_actions

        self.q_est = defaultdict(lambda: np.zeros(self.num_actions))

    def get_action(self, state, epsilon=1e-2):
        state = tuple(state)
        if np.random.uniform() > epsilon:
            return np.argmax(self.q_est[state])
        else:
            return np.random.randint(self.num_actions)

    @classmethod
    def train_by_sarsa(cls, model, transition, gamma=1.0, alpha=0.5):
        state, next_state, action, reward = transition
        state = tuple(state)
        next_state = tuple(next_state)
        next_action = model.get_action(state)

        model.q_est[state][action] += alpha * (reward + gamma * model.q_est[next_state][next_action]
                                               - model.q_est[state][action])


def convert_knapsack_state(state, expected):
    state = expected - state
    return np.clip(state, state_min, state_max)


def train_sarsa_perfect_info(env, model):
    for e in range(60000):
        state = env.reset()
        # print('reset game:', env.warehouse_num, env.knapsack_num)
        expected = env.expected_num - env.warehouse_num
        state = convert_knapsack_state(state, expected)

        global alpha
        global epsilon

        if e % decay_interval == 0 and not e == 0:
            alpha = alpha * 0.1
            epsilon = epsilon * 0.1

        terminate = False
        running_reward = 0.
        # create an episode
        while not terminate:
            action = model.get_action(state, epsilon=epsilon)
            next_state, reward, terminate = env.step(action)
            next_state = convert_knapsack_state(next_state, expected)
            transition = [state, next_state, action, reward]
            state = next_state
            running_reward += reward

            model.train_by_sarsa(model, transition, alpha=alpha)

        # print(e, running_reward)

        if e % test_interval == 0 and not e == 0:
            print(q_net.q_est[(0, 0, 0)])
            print(q_net.q_est[(1, 0, 0)])
            print(q_net.q_est[(0, 1, 0)])
            print(q_net.q_est[(0, 0, 1)])
            print(q_net.q_est[(-1, 0, 0)])
            rewards = []
            for _ in range(100):
                terminate = False
                total_reward = 0.
                state = env.reset()
                expected = env.expected_num - env.warehouse_num
                state = convert_knapsack_state(state, expected)
                while not terminate:
                    action = model.get_action(state, epsilon=1e-3)
                    next_state, reward, terminate = env.step(action)
                    next_state = convert_knapsack_state(next_state, expected)
                    total_reward += reward
                    state = next_state
                rewards.append(total_reward)

            rewards = np.asarray(rewards)
            print(e, np.mean(rewards))


if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read('../game.conf')
    env = FoodGatherEnv(int(cf.defaults()['num_food_types']),
                        int(cf.defaults()['max_capacity']))
    q_net = QNet(4, 0.1)
    train_sarsa_perfect_info(env, q_net)
