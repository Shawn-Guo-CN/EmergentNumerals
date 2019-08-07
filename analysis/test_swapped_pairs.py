import pandas as pd
import numpy as np
import math
from scipy import stats
from collections import defaultdict
import random

from analysis.cal_topological_similarity import load_in_msg_pairs, msg_bleu_dis, msg_ham_dis
import editdistance


CORR_WEIGHTS = [0.9, 0.1, 0.0]


def build_dict_from_pairs(im_pairs):
    ret = {}
    for im_pair in im_pairs:
        key = str(im_pair[0].count('A')) + str(im_pair[0].count('B'))
        value = im_pair[1]
        ret[key] = value
    return ret


def swap_key(key):
    assert len(key) == 2
    new_key = key[1] + key[0]
    return new_key


def distinguish_keys(k1, k2):
    assert len(k1) == 2
    assert len(k2) == 2
    if k1[0] == k2[0] or k1[0] == k2[1] or k1[1] == k2[0] or k1[1] == k2[1]:
        return False
    else:
        return True


def random_select_key(keys, key):
    random_key = random.choice(keys)
    while not distinguish_keys(random_key, key):
        random_key = random.choice(keys)
    return random_key


def test_swapped_pairs(
    msg_file_path,
    msg_dis_measure='bleu',
    corr_method='spearman'
):
    in_msg_pairs = load_in_msg_pairs(msg_file_path)
    im_dict = build_dict_from_pairs(in_msg_pairs)

    swapped_indicators = []
    msg_distances = []
    visit_flag = defaultdict(lambda: False)

    for key in im_dict.keys():
        if visit_flag[key]:
            continue

        swapped_key = swap_key(key)
        random_key = random_select_key(list(im_dict.keys()), key)

        swapped_indicators.append(0)
        swapped_indicators.append(1)
        if msg_dis_measure == 'bleu':
            msg_distances.append(msg_bleu_dis(im_dict[key], im_dict[swapped_key], weights=CORR_WEIGHTS))
            msg_distances.append(msg_bleu_dis(im_dict[key], im_dict[random_key], weights=CORR_WEIGHTS))
        elif msg_dis_measure == 'edit':
            msg_distances.append(editdistance.eval(im_dict[key], im_dict[swapped_key]))
            msg_distances.append(editdistance.eval(im_dict[key], im_dict[random_key]))
        elif msg_dis_measure == 'hamming':
            msg_distances.append(msg_ham_dis(im_dict[key], im_dict[swapped_key]))
            msg_distances.append(msg_ham_dis(im_dict[key], im_dict[random_key]))
        else:
            raise NotImplementedError
        
        visit_flag[key] = True
        visit_flag[swapped_key] = False

    swapped_indicators = np.asarray(swapped_indicators)
    msg_distances = np.asarray(msg_distances)

    if corr_method == 'pearson':
        corr, p = stats.pearsonr(swapped_indicators, msg_distances)
    elif corr_method == 'spearman':
        corr, p = stats.spearmanr(swapped_indicators, msg_distances)

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
