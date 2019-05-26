from utils.conf import *


class Voc:
    def __init__(self):
        self.word2index = {'A': 1, 'B': 2, 'C': 3}
        self.index2word = {SOS_token: "SOS", EOS_token: "EOS",
                           1: 'A', 2: 'B', 3:'C'}
        self.num_words = 5  # Count SOS, EOS

def load_sequences(file_path):
    f = open(file_path, 'r')

    pair_set = []
    for line in f.readlines:
        items = line.strip().split('\t')
        pair_set.append([items[0], items[1]])
    
    return pair_set

def load_train_dev_test(train_file_path=TRAIN_FILE_PATH,
                        dev_file_path=DEV_FILE_PATH,
                        test_file_path=TEST_FILE_PATH):
    train_set = load_sequences(train_file_path)
    dev_set = load_sequences(dev_file_path)
    test_set = load_sequences(test_file_path)
    return train_set, dev_set, test_set
