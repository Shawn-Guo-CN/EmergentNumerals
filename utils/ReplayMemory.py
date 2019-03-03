#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/2 17:48
# @Author  : Shawn
# @File    : ReplayMemory.py
# @Classes :
from collections import namedtuple
import random

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
