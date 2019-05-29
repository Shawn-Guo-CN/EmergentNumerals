from utils.conf import *


def load_sequences(file_path):
    f = open(file_path, 'r')

    string_set = []
    for line in f.readlines():
        string_set.append(line.strip())
    
    return string_set


def load_train_dev_test(train_file_path=TRAIN_FILE_PATH,
                        dev_file_path=DEV_FILE_PATH,
                        test_file_path=TEST_FILE_PATH):
    train_set = load_sequences(train_file_path)
    dev_set = load_sequences(dev_file_path)
    test_set = load_sequences(test_file_path)
    return train_set, dev_set, test_set


def load_train_set(train_file_path=TRAIN_FILE_PATH):
    return load_sequences(TRAIN_FILE_PATH)


def load_dev_set(dev_file_path=DEV_FILE_PATH):
    return load_sequences(DEV_FILE_PATH)


def load_test_set(test_file_path=TEST_FILE_PATH):
    return load_sequences(TEST_FILE_PATH)
