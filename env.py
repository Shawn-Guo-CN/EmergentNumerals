#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/10 17:59
# @Author  : Shawn
# @File    : env.py

import numpy as np
import configparser


class FoodGatherEnv(object):
    def __init__(self, num_food_types=3, max_capacity=5):
        """
        This is the game environment.
        :param max_capacity: int, the maximum capacity for specific kind of food
        """
        self.max_capacity = max_capacity
        self.num_food_types = num_food_types

        self.warehouse_num = np.random.randint(self.max_capacity + 1, size=self.num_food_types)
        self.knapsack_num = np.zeros(self.num_food_types, dtype=int)
        self.expected_num = np.asarray([self.max_capacity] * self.num_food_types)

        self.num_actions = num_food_types + 1
        self.action2shift = {}
        for i in range(self.num_food_types):
            shift = np.zeros(self.num_food_types, dtype=int)
            shift[i] = 1
            self.action2shift[i] = shift
        self.action2shift[self.num_food_types] = np.zeros(self.num_food_types, dtype=int)

        self.knapsack_max = 10 * np.ones(3, dtype=int)
        self.knapsack_min = np.zeros(3, dtype=int)

    def step(self, action):
        self.knapsack_num += self.action2shift[action]
        self.knapsack_num = np.clip(self.knapsack_num, self.knapsack_min, self.knapsack_max)
        reward = -1.
        if action == self.num_food_types:
            reward = 10. if np.array_equal(self.expected_num,
                                           self.knapsack_num + self.warehouse_num) else -10.
            return self.knapsack_num, reward, True
        else:
            return self.knapsack_num, reward, False

    def reset(self):
        self.warehouse_num = np.random.randint(self.max_capacity + 1, size=self.num_food_types)
        self.knapsack_num = np.zeros(self.num_food_types, dtype=int)
        return self.knapsack_num


def test_food_gather_env_by_hand(env):
    print(env.warehouse_num)
    terminate = False
    while not terminate:
        print("Knapsack Status", env.knapsack_num)
        action = int(input("Please input your action:"))
        _, reward, terminate = env.step(action)
        print("reward:", reward)


if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read('./game.conf')
    env = FoodGatherEnv(int(cf.defaults()['num_food_types']),
                        int(cf.defaults()['max_capacity']))
    test_food_gather_env_by_hand(env)
