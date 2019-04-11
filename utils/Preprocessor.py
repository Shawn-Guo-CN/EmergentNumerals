import torch
import numpy as np
import configparser
from env import FoodGatherEnv_GPU


class Preprocessor(object):
    def __init__(self):
        super(Preprocessor, self).__init__()
        cf = configparser.ConfigParser()
        cf.read('./game.conf')
        self.num_types = int(cf.defaults()['num_food_types'])
        self.max_capacity = int(cf.defaults()['max_capacity'])
        self.knapsack_max = int(cf.defaults()['knapsack_max'])

    def env_state_process_one_hot(self, state):
        # comments are for debugging
        # print('--------------------------------------------------------')
        # print('state', state)
        # max_cap + 1 is due to we need to take 0 into consideration
        state_one_hot = torch.zeros((len(state), self.knapsack_max + 1),
                                    device=state.device).scatter_(1, state.view(-1, 1), 1).view(1,-1)
        return state_one_hot
    
    def env_warehouse2message_nhot(self, state):
        """
        :param state: status of warehouse
        """
        # ret = torch.zeros((self.num_types, self.max_capacity), device=state.device)
        # for i, s in enumerate(state):
        #     ret[i, :s] = 1.
        
        # although the following code is elegant, it would reture a tensor with dtype uint8, as there is 
        # a judgement in the code
        ret = torch.stack([torch.arange(self.max_capacity, device=state.device) \
            < a_ for a_ in state]).view(-1).type(torch.float)
        return ret.view(1, -1)

    def env_warehouse2message_onehot(self, state):
        """
        :parama state: status of warehouse
        """
        # max_capacity + 1 is due to that we need to take 0 into consideration
        state_one_hot = torch.zeros((len(state), self.max_capacity + 1),
                                    device=state.device).scatter_(1, state.view(-1, 1), 1).view(1,-1)
        return state_one_hot


if __name__ == '__main__':
    device = torch.device("cpu")
    cf = configparser.ConfigParser()
    cf.read('./game.conf')
    env = FoodGatherEnv_GPU()
    env.to(device)
    preprocessor = Preprocessor()


    k_state = env.reset()
    print('k_state:', k_state)
    print(preprocessor.env_state_process_one_hot(k_state))
    w_state = env.warehouse_num
    print('w_state:', w_state)
    print('n_hot:', preprocessor.env_warehouse2message_nhot(w_state))
    print('1_hot:', preprocessor.env_warehouse2message_onehot(w_state))
    
