import random
import numpy as np
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# parameters for training model
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4 # learning rate
CLIP = 50.0 # max after clipping gradients
TEACHER_FORCING_RATIO = 1.0
DECODER_LEARING_RATIO = 5.0
NUM_ITERS = 4000
PRINT_EVERY = 1
SAVE_EVERY = 500
OPTIMISER = optim.Adam

# for generating and loading data
DATA_FILE_PATH = './data/fix_len_data.txt'
TRAIN_FILE_PATH = './data/train.txt'
DEV_FILE_PATH = './data/dev.txt'
TEST_FILE_PATH = './data/test.txt'

# for preprocessing sequences
SOS_TOKEN = 0  # Start-of-sequence token
EOS_TOKEN = 4  # End-of-sequence token
MAX_LENGTH = 15

# hyperparameters for model
HIDDEN_SIZE = 32
BATCH_SIZE = 64
