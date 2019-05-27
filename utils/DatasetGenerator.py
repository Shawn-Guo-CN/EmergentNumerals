import configparser
from utils.conf import *

def str_sort(s=''):
    # if len(s) <= 1:
    #     return [s]
    char_set = set([])
    for i in s:
        char_set.add(i)
    if len(char_set) == 1:
        return [s]

    str_list = set([])
    for i in range(len(s)):
        for j in str_sort(s[0:i] + s[i + 1:]):
            str_list.add(s[i] + j)
    return str_list

def generate_input_data(in_file='./data/all_targets.txt', out_path='./data/'):
    with open(in_file, mode='r') as f:
        for line in f.readlines():
            line = line.strip()
            line_list = str_sort(line)
            for item in line_list:
                print(item + '\t' + line, file=open(out_path + 'all_pairs.txt', mode='a'))

def generate_target_data(out_file='./data/', max_len=15, num_types=3):
    for a in range(0, max_len+1):
        for b in range(0, max_len+1-a):
            for c in range(0, max_len+1-a-b):
                # print(file=out_file+'all.txt')
                print('A' * a + 'B' * b + 'C' * c, file=open(out_file + 'all_targets.txt', mode='a'))


def generate_fixed_len_data(prefix='', length=15, out_file=open(DATA_FILE_PATH, mode='a')):
    if length == 1:
        for c in ['A', 'B', 'C']:
            print(prefix+c)
            target = ''.join(sorted(prefix+c))
            print(prefix+c+'\t'+target, file=out_file)
    else:
        for c in ['A', 'B', 'C']:
            generate_fixed_len_data(prefix=prefix+c, length=length-1)

def generate_train_dev_test_files(in_file=open(DATA_FILE_PATH, 'r'), 
                                  train_file=open(TRAIN_FILE_PATH, 'a'),
                                  dev_file=open(DEV_FILE_PATH, 'a'),
                                  test_file=open(TEST_FILE_PATH, 'a')):
    pairs_set = []
    for line in in_file.readlines():
        line = line.strip().split('\t')
        pairs_set.append([line[0], line[1]])
    
    random.shuffle(pairs_set)
    train_set = pairs_set[:int(0.8*len(pairs_set))]
    dev_set = pairs_set[int(0.8*len(pairs_set)):int(0.9*len(pairs_set))]
    test_set = pairs_set[int(0.9*len(pairs_set)):]

    def _print_set_(s, out_file):
        for pair in s:
            print(pair[0]+'\t'+pair[1], file=out_file)

    _print_set_(train_set, train_file)
    _print_set_(dev_set, dev_file)
    _print_set_(test_set, test_file)

    in_file.close()
    train_file.close()
    dev_file.close()
    test_file.close()


if __name__ == '__main__':
    # cf = configparser.ConfigParser()
    # cf.read('./game.conf')
    # generate_target_data(out_file=cf['DATA']['out_path'], 
    #     max_len=int(cf['DATA']['max_len']),
    #     num_types=int(cf.defaults()['num_food_types'])
    # )
    # generate_input_data()
    # generate_fixed_len_data('', 3)
    generate_train_dev_test_files()