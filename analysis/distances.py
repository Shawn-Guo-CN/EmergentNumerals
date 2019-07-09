import numpy as np
import editdistance
from utils.conf import args


def instr2np_array(in_str):
    coordinate = []
    for i in range(args.num_words):
        coordinate.append(in_str.count(chr(65+i)))
    return np.asarray(coordinate)


def in_ham_dis(in1, in2):
    return np.sum(in1 != in2)


def in_edit_dis(in1, in2):
    return np.abs(in1 - in2).sum()


def euclid_dis(in1, in2):
    return np.linalg.norm(in1 - in2)


def msg_ham_dis(msg1, msg2):
    return np.sum(msg1 != msg2)


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