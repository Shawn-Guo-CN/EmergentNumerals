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
lr = 1e-8
test_interval = 50
decay_interval = 500
replay_pool = ReplayMemory(5000)
torch.manual_seed(1234)
np.random.seed(1234)
gamma = 0.99
episode_num = 30000
verbose = False


def test(model, env):
    preprocessor = Preprocessor()
    model.eval()
    returns = []
    for e in range(100):
        # TODO: turn on the test mode in the future
        state = env.reset()
        message = preprocessor.env_warehouse2message_onehot(env.warehouse_num)
        terminate = False
        rewards = 0
        
        while not terminate:
            state = preprocessor.env_state_process_one_hot(state)
            state = torch.cat((state, message), 1)
            action = model.get_action(state)
            next_state, reward, terminate = env.step(action)
            rewards += reward
            state = next_state
        
        returns.append(rewards)
    
    print('[Test Performance]', np.mean(returns))
    model.train()


def train(env):
    # state_dim consists of number of foods in warehouse and knapsack
    state_dim = env.num_food_types * (env.max_capacity + 1) + env.num_food_types * (env.knapsack_max + 1)

    # load model configure
    cf = configparser.ConfigParser()
    cf.read('./game.conf')
    model = REINFORCE(state_dim, env.num_actions, int(cf['MODEL']['hidden_dim']))
    # model = REINFORCE_BASELINE(state_dim, env.num_actions, int(cf['MODEL']['hidden_dim']))
    model.to(device)
    model.train()

    optimiser = optim.Adam(model.parameters(), lr=lr)
    preprocessor = Preprocessor()

    for e in range(1, episode_num+1):
        state = env.reset()
        # TODO: may need to try other encoding methods later
        message = preprocessor.env_warehouse2message_onehot(env.warehouse_num)

        terminate = False

        while not terminate:
            # max_cap + 1 is due to we need to take 0 into consideration
            state = preprocessor.env_state_process_one_hot(state)
            state = torch.cat((state, message), 1)
            action = model.get_action(state)
            next_state, reward, terminate = env.step(action)
            model.rewards.append(reward)
            state = next_state
        
        model.train_episode(optimiser, gamma=gamma, verbose=verbose)
        
        if e % test_interval == 0:
            test(model, env)


if __name__ == '__main__':
    env = FoodGatherEnv_GPU()
    env.to(device)
    train(env)
