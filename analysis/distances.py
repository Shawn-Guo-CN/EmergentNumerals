import numpy as np
import editdistance
from utils.conf import args


def instr2np_array(in_str):
    coordinate = []
    for i in range(args.num_words):
        coordinate.append(in_str.count(chr(65+i)))
    return np.asarray(coordinate)


def label2np_array(in_str):
    coordinate = [int(i) for i in in_str]
    return np.asarray(coordinate)


def in_ham_dis(in1, in2):
    return np.sum(in1 != in2)


def in_edit_dis(in1, in2):
    return np.abs(in1 - in2).sum()


def euclid_dis(in1, in2):
    return np.linalg.norm(in1 - in2)


def msg_ham_dis(msg1, msg2):
    return np.sum(msg1 != msg2)


def msg_bleu_dis(msg1, msg2):
    N = msg1.size(0) - 1
    dis = 0
    for n in range(1, N+1):
        dis += ngram_dis(msg1, msg2, N=n)
    return dis / N


def ngram_dis(msg1, msg2, N=1):
    msg1_grams = get_grams(msg1, N)
    msg2_grams = get_grams(msg2, N)
    return len(msg1_grams & msg2_grams) / len(msg1_grams | msg2_grams)


def get_grams(msg, N):
    grams = set([])
    for i in range(msg.size(0) + 1 - N):
        grams.add(msg[i:i+N])
    return grams


if __name__ == '__main__':
    in1 = 'AABB'
    vec1 = instr2np_array(in1)
    in2 = 'ABBBBB'
    vec2 = instr2np_array(in2)
    print(vec1)
    print(vec2)
    print(in_ham_dis(vec1, vec2))
    print(in_edit_dis(vec1, vec2))
    print(msg_ham_dis(vec1, vec2))