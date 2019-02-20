#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/20 12:31
# @Author  : Shawn
# @File    : TabularTD0.py
# @Classes : QNetTabular
import numpy as np


class QNetTabular(object):
    def __init__(self, shape=[4, 12], num_actions=4, epsilon=1e-2):
        self.epsilon = epsilon

        self.q_est = [[[] for _ in range(shape[1])] for __ in range(shape[0])]
        for i in range(shape[0]):
            for j in range(shape[1]):
                for a in range(num_actions):
                    self.q_est[i][j].append(0.)

    def get_action(self, state, epsilon=1e-2, test=False):
        state = state.tolist()
        if np.random.uniform() > epsilon or test:
            return np.argmax(self.q_est[state[0]][state[1]])
        else:
            return np.random.randint(4)

    @classmethod
    def train_by_sarsa(cls, model, transition, gamma=1.0, alpha=0.5):
        state, next_state, action, reward = transition
        next_action = model.get_action(state)

        model.q_est[state[0]][state[1]][action] += alpha * (reward
                                                            + gamma * model.q_est[next_state[0]][next_state[1]][
                                                                next_action]
                                                            - model.q_est[state[0]][state[1]][action])

    @classmethod
    def train_by_q_learning(cls, model, transition, gamma=1.0, alpha=0.5):
        state, next_state, action, reward = transition
        next_action = model.get_action(state, test=True)

        model.q_est[state[0]][state[1]][action] += alpha * (reward
                                                            + gamma * model.q_est[next_state[0]][next_state[1]][
                                                                next_action]
                                                            - model.q_est[state[0]][state[1]][action])
