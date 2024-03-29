#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/20 16:19
# @Author  : Shawn
# @File    : TabularQNet.py
# comment: this version can only run with CPU environment
from collections import defaultdict, namedtuple
import numpy as np
import configparser
from env import FoodGatherEnv
np.random.seed(1234)

test_interval = 50
decay_interval = 100
Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward'))
global epsilon
epsilon = 0.1
global alpha
alpha = 0.5


class QNet(object):
    def __init__(self, num_actions=4):
        self.num_actions = num_actions

        self.q_est = defaultdict(lambda: np.zeros(self.num_actions))

    def get_action(self, state, epsilon=1e-2, greedy=False):
        state = tuple(state)
        if np.random.uniform() >= epsilon or greedy:
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

    @classmethod
    def train_by_q_learning(cls, model, transition, gamma=1.0, alpha=0.5):
        state, next_state, action, reward = transition
        state = tuple(state)
        next_state = tuple(next_state)
        next_action = model.get_action(state, greedy=True)

        model.q_est[state][action] += alpha * (reward + gamma * model.q_est[next_state][next_action]
                                               - model.q_est[state][action])


def convert_knapsack_state(state, expected, state_dim, max_capacity):
    state_min = -1 * np.ones(state_dim, dtype=int)
    state_max = max_capacity * np.ones(state_dim, dtype=int)
    state = expected - state
    return np.clip(state, state_min, state_max)


def test_model(env, model):
    # print(q_net.q_est[(0)])
    # print(q_net.q_est[(1)])
    # print(q_net.q_est[(0, 1)])
    # print(q_net.q_est[(-1, 0)])
    print(q_net.q_est)
    rewards = []
    for _ in range(100):
        terminate = False
        total_reward = 0.
        state = env.reset()
        expected = env.expected_num - env.warehouse_num
        state = convert_knapsack_state(state, expected,
                                       state_dim=env.num_food_types, max_capacity=env.max_capacity)
        while not terminate:
            action = model.get_action(state, epsilon=1e-3)
            next_state, reward, terminate = env.step(action)
            next_state = convert_knapsack_state(next_state, expected,
                                                state_dim=env.num_food_types, max_capacity=env.max_capacity)
            total_reward += reward
            state = next_state
        rewards.append(total_reward)

    rewards = np.asarray(rewards)
    return rewards


def train_sarsa_perfect_info(env, model):
    for e in range(60000):
        state = env.reset()
        # print('reset game:', env.warehouse_num, env.knapsack_num)
        expected = env.expected_num - env.warehouse_num
        state = convert_knapsack_state(state, expected,
                                       state_dim=env.num_food_types, max_capacity=env.max_capacity)

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
            next_state = convert_knapsack_state(next_state, expected,
                                       state_dim=env.num_food_types, max_capacity=env.max_capacity)
            transition = [state, next_state, action, reward]
            state = next_state
            running_reward += reward

            model.train_by_sarsa(model, transition, alpha=alpha)

        if e % test_interval == 0 and not e == 0:
            rewards = test_model(env, model)
            print(e, np.mean(rewards))


def train_q_learning_perfect_info(env, model):
    global alpha
    global epsilon

    for e in range(60000):
        state = env.reset()
        expected = env.expected_num - env.warehouse_num
        state = convert_knapsack_state(state, expected,
                                       state_dim=env.num_food_types, max_capacity=env.max_capacity)

        if e % decay_interval == 0 and not e == 0:
            alpha = alpha * 0.1
            epsilon = epsilon * 0.1

        terminate = False
        # create an episode
        while not terminate:
            action = model.get_action(state, epsilon=epsilon)
            next_state, reward, terminate = env.step(action)
            next_state = convert_knapsack_state(next_state, expected,
                                                state_dim=env.num_food_types, max_capacity=env.max_capacity)
            transition = [state, next_state, action, reward]
            state = next_state

            model.train_by_q_learning(model, transition, alpha=alpha)

        if e % test_interval == 0 and not e == 0:
            rewards = test_model(env, model)
            print(e, np.mean(rewards))


if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read('../game.conf')
    env = FoodGatherEnv(int(cf.defaults()['num_food_types']),
                        int(cf.defaults()['max_capacity']))
    q_net = QNet(env.num_actions)
    train_sarsa_perfect_info(env, q_net)
    # train_q_learning_perfect_info(env, q_net)
