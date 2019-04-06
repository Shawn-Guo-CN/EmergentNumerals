import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

import configparser

from utils.ReplayMemory import *
from models.REINFORCE import REINFORCE
from env import FoodGatherEnv_GPU
from utils.Preprocessor import Preprocessor

device = torch.device("cpu")
lr = 1e-3
test_interval = 20
decay_interval = 50
replay_pool = ReplayMemory(5000)
torch.manual_seed(1234)
epsilon = 0.1
gamma = 0.9
episode_num = 6000


def test(model, env):
    preprocessor = Preprocessor()
    model.eval()
    returns = []
    for e in range(100):
        # TODO: turn on the test mode in the future
        state = env.reset()
        info = preprocessor.env_state_process_ones(env.warehouse_num)
        terminate = False
        rewards = 0
        
        while not terminate:
            state = preprocessor.env_state_process_one_hot(state, env.max_capacity)
            state = torch.cat(state, info)
            action = model.get_action(state)
            next_state, reward, terminate = env.step(action[0])
            rewards += reward
            state = next_state
        
        returns.append(rewards)
    
    print('[Test Performance]', np.mean(returns))
    model.train()


def train(env):
    state_dim = env.num_food_types * env.max_capacity
    model = REINFORCE(state_dim, env.num_actions)
    model.to(device)
    model.reset()
    model.train()

    optimiser = optim.Adam(model.parameters(), lr=lr)
    preprocessor = Preprocessor()

    for e in range(1, episode_num+1):
        if e % decay_interval == 0:
            epsilon *= 0.1
        
        state = env.reset()
        # TODO: may need to try other encoding methods later
        info = preprocessor.env_state_process_ones(env.warehouse_num)

        terminate = False

        while not terminate:
            state = preprocessor.env_state_process_one_hot(state, env.max_capacity)
            state = torch.cat(state, info)
            action = model.get_action(state)
            next_state, reward, terminate = env.step(action[0])
            model.rewards.append(reward)
            state = next_state
        
        model.train_episode(optimiser, gamma=gamma, verbose=True)
        
        if e % test_interval == 0:
            test(model, env)


if __name__ == '__main__':
    cf = configparser.ConfigParser()
    cf.read('./game.conf')
    env = FoodGatherEnv_GPU(int(cf.defaults()['num_food_types']),
                            int(cf.defaults()['max_capacity']))
    env.to(device)
    train(env)
