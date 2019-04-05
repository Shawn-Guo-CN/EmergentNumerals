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
    
    def env_state_process_ones(self, state):
        ret = torch.zeros((self.num_types, self.max_capacity), dtype=torch.int64, device=state.device)
        for i, s in enumerate(state):
            ret[i, :s] = 1.
        
        # although the following code is elegane, it would reture a tensor with dtype uint8, as there is 
        # a judgement in the code
        # ret = torch.stack([torch.arange(self.max_capacity, device=state.device, dtype=torch.int64) \
        #     < a_ for a_ in state]).view(-1)
        return ret.view(-1)


if __name__ == '__main__':
    device = torch.device("cpu")
    cf = configparser.ConfigParser()
    cf.read('./game.conf')
    env = FoodGatherEnv_GPU(int(cf.defaults()['num_food_types']),
                            int(cf.defaults()['max_capacity']))
    env.to(device)
    preprocessor = Preprocessor()


    k_state = env.reset()
    print('k_state', k_state)
    print(preprocessor.env_state_process_ones(k_state))
    w_state = env.warehouse_num
    print('w_state', w_state)
    print(preprocessor.env_state_process_ones(w_state))
    
