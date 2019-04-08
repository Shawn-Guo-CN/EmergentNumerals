import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

import configparser

from models.A2C import AdvantageActorCritic
from env import FoodGatherEnv_GPU
from utils.Preprocessor import Preprocessor

device = torch.device("cpu")
lr = 1e-8
test_interval = 50
torch.manual_seed(1234)
np.random.seed(1234)
gamma = 0.9
episode_num = 30000
update_steps = 5
verbose = False


def test(model, env):
    preprocessor = Preprocessor()
    model.eval()

    returns = []
    for e in range(100):
        # TODO: turn on the test mode in the future
        state = env.reset()
        info = preprocessor.env_warehouse2message_nhot(env.warehouse_num)
        terminate = False
        rewards = 0
        
        while not terminate:
            state = preprocessor.env_state_process_one_hot(state)
            state = torch.cat((state, info), 1)
            action = model.get_action(state)
            next_state, reward, terminate = env.step(action)
            rewards += reward
            state = next_state
        returns.append(rewards)
    
    print('[Test Performance]', np.mean(returns))
    
    model.clear_buffer()
    model.train()


def train(env):
    # state_dim consists of number of foods in warehouse and knapsack
    state_dim = env.num_food_types * env.max_capacity + env.num_food_types * (env.knapsack_max + 1)

    # load model configure
    cf = configparser.ConfigParser()
    cf.read('./game.conf')
    model = AdvantageActorCritic(state_dim, env.num_actions, int(cf['MODEL']['hidden_dim']))
    model.to(device)
    model.train()

    optimiser = optim.Adam(model.parameters(), lr=lr)
    preprocessor = Preprocessor()

    for e in range(1, episode_num+1):
        state = env.reset()
        # TODO: may need to try other encoding methods later
        info = preprocessor.env_warehouse2message_nhot(env.warehouse_num)

        terminate = False
        step = 0

        while not terminate:
            step += 1
            state = preprocessor.env_state_process_one_hot(state)
            state = torch.cat((state, info), 1)

            # note that current state is actually the next_state for last loop step
            if step % update_steps == 0:
                step = 0
                model.train_trajectory(state, optimiser, gamma=gamma, verbose=verbose)
            
            action = model.get_action(state)
            next_state, reward, terminate = env.step(action)
            model.add_saved_reward(reward)
            state = next_state
        
        model.train_trajectory('done', optimiser, gamma=gamma, verbose=verbose)
        
        if e % test_interval == 0:
            test(model, env)


if __name__ == '__main__':
    env = FoodGatherEnv_GPU()
    env.to(device)
    train(env)
