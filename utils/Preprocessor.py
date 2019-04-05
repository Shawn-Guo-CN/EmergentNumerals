import torch
import configparser

class Preprocessor(object):
    def __init__(self):
        cf = configparser.ConfigParser()
        cf.read('../game.conf')
        num_types = int(cf.defaults()['num_food_types']),
        max_capacity = int(cf.defaults()['max_capacity'])
    
    def env_state_process(self, state):
        pass
