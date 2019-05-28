import random
import numpy as np
import math
import os

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# parameters for training model
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4 # learning rate
DROPOUT_RATIO = 0.2
CLIP = 50.0 # max after clipping gradients
TEACHER_FORCING_RATIO = 0.3
DECODER_LEARING_RATIO = 5.0
NUM_ITERS = 4000
PRINT_EVERY = 1
SAVE_EVERY = 100
OPTIMISER = optim.Adam
LOSS_FUNCTION = nn.CrossEntropyLoss()

# for saving and loading params of models
SAVE_DIR = './params/'

# for generating and loading data
DATA_FILE_PATH = './data/fix_len_data.txt'
TRAIN_FILE_PATH = './data/sample_train.txt'
DEV_FILE_PATH = './data/sample_dev.txt'
TEST_FILE_PATH = './data/sample_test.txt'

# for preprocessing sequences
SOS_TOKEN = 0  # Start-of-sequence token
EOS_TOKEN = 4  # End-of-sequence token
MAX_LENGTH = 15

# hyperparameters for model
HIDDEN_SIZE = 32
BATCH_SIZE = 64
