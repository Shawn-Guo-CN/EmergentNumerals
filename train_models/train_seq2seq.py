from utils.conf import *
from models.Seq2Seq import *
from utils.DatasetLoader import *


def train():
    train_set, dev_set, test_set = load_train_dev_test()
    print(len(train_set), len(dev_set), len(test_set))
    print(random.choice(train_set))


if __name__ == '__main__':
    train()
