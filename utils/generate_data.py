from utils.conf import args
import random
import os


def generate_all_combinations(prefix='', type_idx=1, out_file=open(args.data_file, 'a')):
    if type_idx == args.num_words:
        for i in range(0, args.max_len_word+1):
            target_str = chr(64+type_idx) * i
            print(prefix+target_str)
            print(prefix+target_str, file=out_file)
    else:
        for i in range(0, args.max_len_word+1):
            target_str = chr(64+type_idx) * i
            generate_all_combinations(prefix+target_str, type_idx+1, out_file=out_file)


def generate_compositional_lan_pair(io_file=open(args.data_file, 'r')):
    string_set = []
    for line in io_file.readlines():
        if len(line.strip()) == 0:
            continue
        msg = ''
        for i in range(args.num_words):
            msg += str(line.count(chr(65+i)))
        string_set.append(msg + '\t' + line.strip())
    
    io_file.close()
    os.remove(args.data_file)

    io_file = open(args.data_file, 'a')
    for string in string_set:
        print(string, file=io_file)
    io_file.close()


def generate_train_dev_test_files(in_file=open(args.data_file, 'r'),
                                  train_file=open(args.train_file, 'a'),
                                  dev_file=open(args.dev_file, 'a'),
                                  test_file=open(args.test_file, 'a')):
    string_set = []
    for line in in_file.readlines():
        if len(line.strip()) == 0:
            continue
        string_set.append(line.strip())
    
    random.shuffle(string_set)
    train_set = string_set[:int(0.9*len(string_set))]
    dev_set = string_set[int(0.9*len(string_set)):int(0.95*len(string_set))]
    test_set = string_set[int(0.95*len(string_set)):]

    def _print_set_(s, out_file):
        for string in s:
            print(string, file=out_file)

    _print_set_(train_set, train_file)
    _print_set_(dev_set, dev_file)
    _print_set_(test_set, test_file)

    in_file.close()
    train_file.close()
    dev_file.close()
    test_file.close()


def generate_fixed_len_data(prefix='', length=15, out_file=open(args.data_file, mode='a')):
    if length == 1:
        for c in ['A', 'B', 'C']:
            print(prefix+c)
            target = ''.join(sorted(prefix+c))
            print(prefix+c+'\t'+target, file=out_file)
    else:
        for c in ['A', 'B', 'C']:
            generate_fixed_len_data(prefix=prefix+c, length=length-1)

def generate_train_dev_test_files_bak(in_file=open(args.data_file, 'r'), 
                                  train_file=open(args.train_file, 'a'),
                                  dev_file=open(args.dev_file, 'a'),
                                  test_file=open(args.test_file, 'a')):
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
    generate_all_combinations()
    generate_compositional_lan_pair()
    generate_train_dev_test_files()
