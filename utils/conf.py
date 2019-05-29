import random
import numpy as np
import math
import os

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

'''
for training model
'''
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4 # learning rate
DROPOUT_RATIO = 0.1
CLIP = 50.0 # max after clipping gradients
TEACHER_FORCING_RATIO = 1.0
DECODER_LEARING_RATIO = 5.0
NUM_ITERS = 4000
PRINT_EVERY = 1
SAVE_EVERY = 10
EVAL_EVERY = 10
OPTIMISER = optim.Adam
LOSS_FUNCTION = nn.CrossEntropyLoss()

'''
for saving and loading params of models
'''
SAVE_DIR = './params/'
PARAM_FILE = './params/standard_seq2seq_32/100_checkpoint.tar'

'''
for generating and loading data
'''
DATA_FILE_PATH = './data/all_data.txt'
TRAIN_FILE_PATH = './data/train.txt'
DEV_FILE_PATH = './data/dev.txt'
TEST_FILE_PATH = './data/test.txt'

'''
for preprocessing sequences
'''
SOS_TOKEN = 0  # Start-of-sequence token
NUM_WORD = 6 # Number of different characters
EOS_TOKEN = 4  # End-of-sequence token
MAX_LEN_WORD = 9 # Maximum length of a single kind of word
MAX_LENGTH = NUM_WORD * MAX_LEN_WORD # Max length of whole sequence

'''
hyperparameters of model
'''
HIDDEN_SIZE = 256
BATCH_SIZE = 64
