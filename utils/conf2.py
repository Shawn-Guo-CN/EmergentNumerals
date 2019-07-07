import random
import numpy as np
import math
import os
import itertools

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch import autograd

'''
for training model
'''
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4 # learning rate
DROPOUT_RATIO = 0.2
CLIP = 50.0 # max after clipping gradients
TEACHER_FORCING_RATIO = 0.5
DECODER_LEARING_RATIO = 5.0
NUM_ITERS = 15000
PRINT_EVERY = 1
SAVE_EVERY = 100000
EVAL_EVERY = 100
OPTIMISER = optim.Adam
LOSS_FUNCTION = nn.CrossEntropyLoss(reduce=False)

'''
for testing model
'''
TEST_MODE = False


'''
for saving and loading params of models
'''
SAVE_DIR = './tmp/'
PARAM_FILE = None

'''
for generating and loading data
'''
DATA_FILE_PATH = './data/2/all_data.txt'
TRAIN_FILE_PATH = './data/2/train.txt'
DEV_FILE_PATH = './data/2/dev.txt'
TEST_FILE_PATH = './data/2/test.txt'

'''
for preprocessing sequences
'''
PAD_TOKEN = 'PAD' # Padding token
PAD_INDEX = 0 # PAD token index
SOS_TOKEN = 'SOS' # Start-of-sequence token
SOS_INDEX = 1  # SOS token index
EOS_TOKEN = 'EOS' # End-of-sequence token
EOS_INDEX = 2 # EOS token index
NUM_WORD = 2 # Number of different characters
MAX_LEN_WORD = 5 # Maximum length of a single kind of word
MAX_LENGTH = NUM_WORD * MAX_LEN_WORD # Max length of whole sequence

'''
hyperparameters of model
'''
HIDDEN_SIZE = 256
BATCH_SIZE = 2048
MSG_MAX_LEN = NUM_WORD + 2
# Size of vocabulary this is available for communication
MSG_VOCSIZE = 10
MSG_MODE = 'GUMBEL' # 'SOFTMAX', 'GUMBEL' or 'REINFORCE'
MSG_TAU = 1. # Hyperparam for gumbel softmax
MSG_HARD = True # Discretized as one-hot vectors
