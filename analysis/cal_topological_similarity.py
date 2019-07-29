import pandas as pd
import numpy as np
import math
from scipy import stats
from utils.conf import args
from analysis.get_input_message_pairs import generate_perfect_in_msg_pairs
from analysis.get_input_message_pairs import output_perfect_in_msg_pairs
import editdistance

DATA_FILE = './data/all_data.txt'
MSG_FILE = './data/input_msg_pairs.txt'

def load_in_msg_pairs(file_path=MSG_FILE):
    in_file = open(file_path, 'r')
    
    im_pairs = []
    
    for line in in_file.readlines():
        line = line.strip()
        # in_str, msg, _ = line.split('\t')
        # msg = msg[:-1]
        msg, in_str = line.split('\t')
        im_pairs.append([in_str, msg])

    return im_pairs


def in_ham_dis(in1, in2):
    dis = 0
    for c in range(args.num_words):
        cur_char = chr(65+c)
        if not in1.count(cur_char) == in2.count(cur_char):
            dis += 1
    return dis


def in_edit_dis(in1, in2):
    dis = 0
    for c in range(args.num_words):
        cur_char = chr(65+c)
        dis += abs(in1.count(cur_char) - in2.count(cur_char))
    return dis


def in_euclid_dis(in1, in2):
    dis = 0
    for c in range(args.num_words):
        cur_char = chr(65+c)
        dis += (in1.count(cur_char) - in2.count(cur_char)) ** 2
    return math.sqrt(dis)


def msg_ham_dis(msg1, msg2):
    dis = 0
    dis += max(len(msg1), len(msg2)) - min(len(msg1), len(msg2))
    for i in range(min(len(msg1), len(msg2))):
        if not msg1[i] == msg2[i]:
            dis += 1
    return dis


def msg_bleu_dis(msg1, msg2, weights=[0.9, 0.1, 0.0]):
    assert len(msg1) == len(msg2)
    assert len(weights) == len(msg1) - 1
    N = len(msg1) - 1
    dis = 0
    for n in range(1, N+1):
        dis += weights[n-1] * ngram_dis(msg1, msg2, N=n)
    return 1. - dis


def ngram_dis(msg1, msg2, N=1):
    msg1_grams = get_grams(msg1, N)
    msg2_grams = get_grams(msg2, N)
    return len(msg1_grams & msg2_grams) / len(msg1_grams | msg2_grams)


def get_grams(msg, N):
    grams = set([])
    for i in range(len(msg) + 1 - N):
        grams.add(msg[i:i+N])
    return grams


def cal_topological_sim(
        msg_file_path=MSG_FILE, 
        in_dis_measure='hamming',
        msg_dis_measure='edit',
        corr_method='pearson'):
    in_msg_pairs = load_in_msg_pairs(msg_file_path)

    mean_distances = []
    symbol_distances = []

    len_pairs = len(in_msg_pairs)
    for i, im_pair in enumerate(in_msg_pairs):
        for j in range(i+1, len_pairs):
            if in_dis_measure == 'hamming':
                mean_distances.append(in_ham_dis(im_pair[0], in_msg_pairs[j][0]))
            elif in_dis_measure == 'edit':
                mean_distances.append(in_edit_dis(im_pair[0], in_msg_pairs[j][0]))
            elif in_dis_measure == 'euclidean':
                mean_distances.append(in_euclid_dis(im_pair[0], in_msg_pairs[j][0]))
            else:
                raise NotImplementedError
            
            if msg_dis_measure == 'edit':
                symbol_distances.append(editdistance.eval(im_pair[1], in_msg_pairs[j][1]))
            elif msg_dis_measure == 'hamming':
                symbol_distances.append(msg_ham_dis(im_pair[1], in_msg_pairs[j][1]))
            elif msg_dis_measure == 'bleu':
                symbol_distances.append(msg_bleu_dis(im_pair[1], in_msg_pairs[j][1]))
            else:
                raise NotImplementedError
    
    mean_distances = np.asarray(mean_distances)
    symbol_distances = np.asarray(symbol_distances)

    if symbol_distances.sum() == 0:
        symbol_distances = symbol_distances + 0.1
        symbol_distances[-1] -= 0.01

    # dis_table = pd.DataFrame({'MD': mean_distances, 'SD': symbol_distances})
    # corr = dis_table.corr(corr_method)['SD']['MD']
    if corr_method == 'pearson':
        corr, p = stats.pearsonr(mean_distances, symbol_distances)
    elif corr_method == 'spearman':
        corr, p = stats.spearmanr(mean_distances, symbol_distances)

    return corr, p


def main():
    sim, p = cal_topological_sim(
            msg_file_path='./data/rebuilt_language_2_refer_IL.txt',
            in_dis_measure='euclidean',
            msg_dis_measure='bleu',
            corr_method='spearman'
        )
    print(sim)
    print(p)

if __name__ == '__main__':
    main()