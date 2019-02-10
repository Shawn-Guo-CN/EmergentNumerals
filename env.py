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
        self.knapsack_num = np.zeros(self.num_food_types)

        self.action2shift = {}
        for i in range(self.num_food_types):
            shift = np.zeros(self.num_food_types, dtype=int)
            shift[i] = 1
            self.action2shift[i] = shift
        self.action2shift[self.num_food_types] = np.zeros(self.num_food_types, dtype=int)

    def _build_transition_matrix(self):
        pass

    def step(self):
        pass


if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read('./game.conf')
    env = FoodGatherEnv(int(cf.defaults()['num_food_types']),
                        int(cf.defaults()['max_capacity']))
    print(env.action2shift)
