import argparse
import random

import torch
import torch.nn as nn
from torch import optim

'''
default values for argparser
'''
defaults = {
    'DEVICE': 1,
    'LEARNING_RATE':1e-4,
    'DROPOUT_RATIO': 0.2,
    'CLIP': 50.0,
    'TEACHER_FORCING_RATIO': 0.5,
    'SPEAKER_LEARNING_RATIO': 5.0,
    'LISTENER_LEARNING_RATIO': 0.0,
    'NUM_ITERS': 40000,
    'PRINT_EVERY': 1,
    'SAVE_EVERY': 10000,
    'EVAL_EVERY': 10,
    'OPTIMISER': optim.Adam,
    'LOSS_FUNCTION': nn.CrossEntropyLoss(reduce=False),
    'TEST_MODE': False,
    'SAVE_DIR': './params/',
    'PARAM_FILE': '',
    'DATA_FILE_PATH': './data/2/all_data.txt',
    'TRAIN_FILE_PATH': './data/2/train.txt',
    'DEV_FILE_PATH': './data/2/dev.txt',
    'TEST_FILE_PATH': './data/2/test.txt',
    'PAD_TOKEN': 'PAD', # Padding token
    'PAD_INDEX': 0, # PAD token index
    'SOS_TOKEN': 'SOS', # Start-of-sequence token
    'SOS_INDEX': 1,  # SOS token index
    'EOS_TOKEN': 'EOS', # End-of-sequence token
    'EOS_INDEX': 2, # EOS token index
    'NUM_WORD': 2, # Number of different characters
    'MAX_LEN_WORD': 5,
    'HIDDEN_SIZE': 256,
    'BATCH_SIZE': 1024,
    'MSG_MODE': 'GUMBEL', # 'SOFTMAX', 'GUMBEL', 'SCST' or 'REINFORCE'
    'MSG_TAU': 1.,
    'L_RESET_FREQ': -1,
    'SIM_CHK_FREQ': 1000,
    'SIM_CHK_K': 50,
    'NUM_DISCTRACTOR': 4,
    'NUM_GENERATION': 50,
    'NUM_PLAYITER': 250,
    'NUM_SPKLEARNITER': 20,
    'NUM_LWARMUPITER':20,
    'EARLAY_STOP_THRESHOLD': 0.96,
}

'''
for preprocessing sequences
'''
# Max length of whole sequence, extra 1 for EOS
MAX_LENGTH = defaults['NUM_WORD'] * defaults['MAX_LEN_WORD'] + 1 

'''
hyperparameters of model
'''
MSG_MAX_LEN = defaults['NUM_WORD'] + 2
MSG_VOCSIZE = defaults['MAX_LEN_WORD'] + 5 # Consider 0 and EOS for MSG


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', type=int, default=defaults['DEVICE'], choices=[0, 1, 2, 3, -1],
        help="which gpu to use, -1 for cpu")
parser.add_argument('-lr', '--learning-rate', type=float, default=defaults['LEARNING_RATE'],
        help="learning rate for optimisers")
parser.add_argument('-dpr', '--dropout-ratio', type=float, default=defaults['DROPOUT_RATIO'],
        help="dropout ratio in model's modules")
parser.add_argument('--clip', type=float, default=defaults['CLIP'],
        help="the maximum of gradients after clipped")
parser.add_argument('--teacher-ratio', type=float, default=defaults['TEACHER_FORCING_RATIO'],
        help='teacher-forcing ratio for training last sequence decoder')
parser.add_argument('--speaker-ratio', type=float, default=defaults['SPEAKER_LEARNING_RATIO'],
        help='learning rate ratio for training speaker')
parser.add_argument('--listener-ratio', type=float, default=defaults['LISTENER_LEARNING_RATIO'],
        help='learning rate ratio for training listener')
parser.add_argument('-i', '--iter_num', type=int, default=defaults['NUM_ITERS'],
        help='maximum number of iterations')
parser.add_argument('--print-freq', type=int, default=defaults['PRINT_EVERY'],
        help='frequency of output training information')
parser.add_argument('--eval-freq', type=int, default=defaults['EVAL_EVERY'], 
        help='frequency of evaluation during training')
parser.add_argument('--save-freq', type=int, default=defaults['SAVE_EVERY'],
        help='frequency of saving parameters during training')
parser.add_argument('--optimiser', type=str, default='adam',
        help='the optimiser for training model, currently only ADAM is supported')
parser.add_argument('-loss', '--loss-function', type=str, default='cross_entropy',
        help='loss function for training model')
parser.add_argument('-test', '--test', action="store_true",
        help='choose mode between TRAIN and TEST')
parser.add_argument('--save-dir', type=str, default=defaults['SAVE_DIR'],
        help='directory for saving parameters during training')
parser.add_argument('--param-file', type=str, default=defaults['PARAM_FILE'],
        help='path to the saved parameter file')
parser.add_argument('--data-file', type=str, default=defaults['DATA_FILE_PATH'],
        help='path to the file containing all data')
parser.add_argument('--train-file', type=str, default=defaults['TRAIN_FILE_PATH'],
        help='path to the train set file')
parser.add_argument('--dev-file', type=str, default=defaults['DEV_FILE_PATH'],
        help='path to the dev set file')
parser.add_argument('--test-file', type=str, default=defaults['TEST_FILE_PATH'],
        help='path to the test set file')
parser.add_argument('--pad-token', type=str, default=defaults['PAD_TOKEN'],
        help='token for padding sequences')
parser.add_argument('--sos-token', type=str, default=defaults['SOS_TOKEN'],
        help='token for Start-Of-Sequences')
parser.add_argument('--eos-token', type=str, default=defaults['EOS_TOKEN'],
        help='token for End-Of-Sequences')
parser.add_argument('--pad-index', type=int, default=defaults['PAD_INDEX'],
        help='index for padding tokens')
parser.add_argument('--sos-index', type=int, default=defaults['SOS_INDEX'],
        help='index for SOS tokens')
parser.add_argument('--eos-index', type=int, default=defaults['EOS_INDEX'],
        help='index for EOS tokens')
parser.add_argument('--num-words', type=int, default=defaults['NUM_WORD'],
        help='number of different kinds of words')
parser.add_argument('--max-len-word', type=int, default=defaults['MAX_LEN_WORD'],
        help='maximum length of a single kind of word')
parser.add_argument('--hidden-size', type=int, default=defaults['HIDDEN_SIZE'],
        help='hidden size of the models')
parser.add_argument('-b', '--batch-size', type=int, default=defaults['BATCH_SIZE'],
        help='batch size during traing and testing')
parser.add_argument('--soft', action='store_true',
        help='false discritisation of the messages')
parser.add_argument('--max-seq-len', type=int, default=MAX_LENGTH,
        help='maximum length of the i/o sequences')
parser.add_argument('--max-msg-len', type=int, default=MSG_MAX_LEN,
        help='maximum length of the messages')
parser.add_argument('--msg-vocsize', type=int, default=MSG_VOCSIZE,
        help='size of vocabulary this is available for communication')
parser.add_argument('-t', '--tau', type=float, default=defaults['MSG_TAU'],
        help='tau in GUMBEL softmax')
parser.add_argument('-m', '--msg-mode', type=str, default=defaults['MSG_MODE'],
        help='mode of message generation')
parser.add_argument('--l-reset-freq', type=int, default=defaults['L_RESET_FREQ'],
        help='frequence of resetting listener, -1 for no reset')
parser.add_argument('--sim-chk-freq', type=int, default=defaults['SIM_CHK_FREQ'],
        help='frequence of checking topological similarity during traing')
parser.add_argument('--sim-chk-k', type=int, default=defaults['SIM_CHK_K'],
        help='number of sample data used to check topological similarity')
parser.add_argument('--num-distractors', type=int, default=defaults['NUM_DISCTRACTOR'],
        help='number of distractors in a training sample')
parser.add_argument('--num-generation', type=int, default=defaults['NUM_GENERATION'],
        help='number of training generations')
parser.add_argument('--num-play-iter', type=int, default=defaults['NUM_PLAYITER'],
        help='number of playing game iterations')
parser.add_argument('--num-spklearn-iter', type=int, default=defaults['NUM_SPKLEARNITER'],
        help='number of speaker learning iterations')
parser.add_argument('--num-lwarmup-iter', type=int, default=defaults['NUM_LWARMUPITER'],
        help='number of listener warming up iterations')
parser.add_argument('--early-stop', type=float, default=defaults['EARLAY_STOP_THRESHOLD'],
        help='threshold for early stopping during game playing phase')
args = parser.parse_args()

args.device = torch.device("cuda:" + str(args.device) \
    if torch.cuda.is_available() and not args.device == -1 else "cpu")
args.optimiser = defaults['OPTIMISER'] if args.optimiser == 'adam' else None
args.loss_function = defaults['LOSS_FUNCTION'] if args.loss_function == 'cross_entropy' else None
args.param_file = None if len(args.param_file) == 0 else args.param_file


def set_random_seed(seed:int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
