#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/10 17:59
# @Author  : Shawn
# @File    : env.py

import numpy as np
import configparser
import torch
import torch.nn as nn


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

        self.knapsack_max = 10 * np.ones(self.num_food_types, dtype=int)
        self.knapsack_min = np.zeros(self.num_food_types, dtype=int)

    def step(self, action):
        self.knapsack_num += self.action2shift[action]
        self.knapsack_num = np.clip(self.knapsack_num, self.knapsack_min, self.knapsack_max)
        reward = -1.
        if action == self.num_food_types:
            result = self.expected_num - (self.knapsack_num + self.warehouse_num)
            multiplier1 = np.count_nonzero(result == 0)
            multiplier2 = np.count_nonzero(result)
            reward = multiplier1 * 100. - multiplier2 * 100.
            return self.knapsack_num, reward, True
        else:
            return self.knapsack_num, reward, False

    def reset(self):
        self.warehouse_num = np.random.randint(self.max_capacity + 1, size=self.num_food_types)
        self.knapsack_num = np.zeros(self.num_food_types, dtype=int)
        return self.knapsack_num


class FoodGatherEnv_GPU(nn.Module):
    def __init__(self, num_food_types=3, max_capacity=5):
        """
        This is the game environment.
        :param max_capacity: int, the maximum capacity for specific kind of food
        """
        super(FoodGatherEnv_GPU, self).__init__()
        self.max_capacity = max_capacity
        self.num_food_types = num_food_types

        warehouse_num = torch.randint(0, self.max_capacity + 1, (self.num_food_types,), dtype=torch.int64)
        knapsack_num = torch.zeros((self.num_food_types,), dtype=torch.int64)
        expected_num = self.max_capacity * torch.ones((self.num_food_types,), dtype=torch.int64)

        self.num_actions = num_food_types + 1
        action2shift_eye = torch.eye(self.num_food_types, dtype=torch.int64)
        action2shift_end = torch.zeros((1, self.num_food_types), dtype=torch.int64)
        action2shift = torch.cat((action2shift_eye, action2shift_end), dim=0)

        self.knapsack_max = 10

        self.register_buffer('warehouse_num', warehouse_num)
        self.register_buffer('knapsack_num', knapsack_num)
        self.register_buffer('expected_num', expected_num)
        self.register_buffer('action2shift', action2shift)

    def step(self, action):
        return self.forward(action)

    def forward(self, action):
        self.knapsack_num += self.action2shift[action]
        self.knapsack_num = torch.clamp(self.knapsack_num, max=self.knapsack_max)
        reward = -1 * torch.ones((1,), dtype=torch.float, device=self.knapsack_num.device)
        if action == self.num_food_types:
            if torch.equal(self.expected_num, self.knapsack_num + self.warehouse_num):
                reward = 100 * torch.ones((1,), dtype=torch.float, device=self.knapsack_num.device)
            else:
                reward = -100 * torch.ones((1,), dtype=torch.float, device=self.knapsack_num.device)
            return self.knapsack_num, reward, True
        else:
            return self.knapsack_num, reward, False

    def reset(self):
        self.warehouse_num = torch.randint(0, self.max_capacity + 1, (self.num_food_types,),
                                           dtype=torch.int64, device=self.knapsack_num.device)
        self.knapsack_num = torch.zeros((self.num_food_types,), dtype=torch.int64, device=self.knapsack_num.device)
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
    # env = FoodGatherEnv(int(cf.defaults()['num_food_types']),
    #                     int(cf.defaults()['max_capacity']))
    env = FoodGatherEnv_GPU(int(cf.defaults()['num_food_types']),
                            int(cf.defaults()['max_capacity']))
    env.to(torch.device("cuda"))
    test_food_gather_env_by_hand(env)
