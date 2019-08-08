import pandas as pd
import numpy as np
import math
from scipy import stats
from collections import defaultdict
import random

from analysis.cal_topological_similarity import load_in_msg_pairs, msg_bleu_dis, msg_ham_dis
from analysis.test_swapped_pairs import build_dict_from_pairs
import editdistance


CORR_WEIGHTS = [0.9, 0.1, 0.0]


def scan_keys():
    # This is a temporary impolementation
    return [0, 1, 2, 3, 4, 5]


def generate_neighbor_keys_dis(key, all_keys):
    assert len(key) == 2
    keys = []
    key_dis = []

    k1 = key[0]
    k2 = key[1]

    for i in range(int(k1) + 1, max(all_keys)):
        keys.append(str(i) + k2)
        key_dis.append(i - int(k1))
    
    for j in range(int(k2) + 1, max(all_keys)):
        keys.append(k1 + str(j))
        key_dis.append(j - int(k2))

    return keys, key_dis


def test_swapped_pairs(
    msg_file_path,
    msg_dis_measure='bleu',
    corr_method='spearman'
):
    in_msg_pairs = load_in_msg_pairs(msg_file_path)
    im_dict = build_dict_from_pairs(in_msg_pairs)

    mean_distances = []
    msg_distances = []

    for key in im_dict.keys():
        all_keys = scan_keys()
        neighbor_keys, key_diss = generate_neighbor_keys_dis(key, all_keys)

        for idx, neighbor_key in enumerate(neighbor_keys):
            mean_distances.append(key_diss[idx])

            if msg_dis_measure == 'bleu':
                msg_distances.append(msg_bleu_dis(im_dict[key], im_dict[neighbor_key], weights=CORR_WEIGHTS))
            elif msg_dis_measure == 'edit':
                msg_distances.append(editdistance.eval(im_dict[key], im_dict[neighbor_key]))
            elif msg_dis_measure == 'hamming':
                msg_distances.append(msg_ham_dis(im_dict[key], im_dict[neighbor_key]))
            else:
                raise NotImplementedError

    mean_distances = np.asarray(mean_distances)
    msg_distances = np.asarray(msg_distances)

    if corr_method == 'pearson':
        corr, p = stats.pearsonr(mean_distances, msg_distances)
    elif corr_method == 'spearman':
        corr, p = stats.spearmanr(mean_distances, msg_distances)

    return corr, p


def main():
    sim, p = test_swapped_pairs(
            msg_file_path='./data/rebuilt_language_2_refer_IL.txt',
            msg_dis_measure='edit',
            corr_method='spearman'
        )
    print(sim)
    print(p)


if __name__ == '__main__':
    random.seed(1234)
    main()
