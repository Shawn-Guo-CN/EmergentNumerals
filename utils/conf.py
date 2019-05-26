import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# device for training model
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# for generating and loading data
DATA_FILE_PATH = './data/fix_len_data.txt'
TRAIN_FILE_PATH = './data/train.txt'
DEV_FILE_PATH = './data/dev.txt'
TEST_FILE_PATH = './data/test.txt'

# for preprocessing sequences
SOS_token = 0  # Start-of-sequence token
EOS_token = 4  # End-of-sequence token

# hyperparameters for model
HIDDEN_SIZE = 32
