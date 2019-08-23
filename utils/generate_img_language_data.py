import os
import random
import numpy as np
from utils.conf import args


def generate_all_combinations(ret=[], prefix='', type_idx=0, max_num_objs=args.max_len_word):
    if type_idx == args.num_words - 1:
        for i in range(0, max_num_objs+1):
            target_str = str(i)
            ret.append(prefix+target_str)
    else:
        for i in range(0, max_num_objs+1):
            target_str = str(i)
            generate_all_combinations(ret, prefix+target_str, type_idx+1)


def generate_language_file(
    meanings,
    out_file_path,
    num_type_objs=args.num_words,
    max_num_objs=args.max_len_word+1,
    num_change=0,
):
    meaning_list = meanings
    message_list = []

    meaning_groups = []
    idx = 0
    g = []
    for m in meanings:
        idx += 1
        g.append(m)
        if idx % max_num_objs == 0:
            meaning_groups.append(g)
            g = []
            idx = 0

    assert num_change <= len(meaning_groups)
    g_ids = np.random.choice(range(len(meaning_groups)), size=num_change, replace=False)
    items_shuffle = []
    for idx, group in enumerate(meaning_groups):
        if idx in g_ids:
            items_shuffle += group

    random.shuffle(items_shuffle)
    replace_idx = 0
    for idx, group in enumerate(meaning_groups):
        if idx in g_ids:
            group = items_shuffle[replace_idx*len(group):(replace_idx+1)*len(group)]
            replace_idx += 1
        message_list += group
    
    assert len(meaning_list) == len(message_list)

    with open(out_file_path, 'a') as f:
        for idx in range(len(meaning_list)):
            print(meaning_list[idx]+'\t'+message_list[idx], file=f)

if __name__ == '__main__':
    combines = []
    generate_all_combinations(ret=combines)
    
    generate_language_file(combines, './data/img_languages/holistic0.txt', num_change=0)
