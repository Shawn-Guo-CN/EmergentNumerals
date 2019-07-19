import torch
import numpy as np
import pandas as pd
import math
import analysis.distances as distances

def sim_check(
    model, in_set, batch_set,
    in_dis_measure='euclidean',
    spk_hidden_measure='euclidean',
    msg_dis_measure='edit',
    lis_hidden_measure='euclidean',
    corr_measure='pearson'
):
    in_np_set = []
    for in_str in in_set:
        in_np_set.append(distances.instr2np_array(in_str))
    spk_hset = reproduce_spk_hidden_set(model, batch_set)
    msg_set = reproduce_msg_set(model, batch_set)
    lis_hset = reproduce_lis_hidden_set(model, batch_set)

    assert len(in_set) == len(batch_set)
    
    mean_distances = []
    spk_h_distances = []
    msg_distances = []
    lis_h_distances = []

    for i in range(len(in_set)-1):
        for j in range(i+1, len(in_set)):
            mean_distances.append(get_in_dis(in_np_set[i], in_np_set[j], measure=in_dis_measure))
            spk_h_distances.append(get_hidden_dis(spk_hset[i], spk_hset[j], measure=spk_hidden_measure))
            msg_distances.append(get_msg_dis(msg_set[i], msg_set[j], measure=msg_dis_measure))
            lis_h_distances.append(get_hidden_dis(lis_hset[i], lis_hset[j], measure=lis_hidden_measure))
    
    mean_distances = np.asarray(mean_distances)
    spk_h_distances = np.asarray(spk_h_distances, dtype=float)
    msg_distances = np.asarray(msg_distances, dtype=float)
    lis_h_distances = np.asarray(lis_h_distances, dtype=float)

    spk_h_distances = check_distance_set(spk_h_distances)
    msg_distances = check_distance_set(msg_distances)
    lis_h_distances = check_distance_set(lis_h_distances)

    dis_table = pd.DataFrame(
            {'MD': mean_distances, 'SHD': spk_h_distances, 'MSD': msg_distances, 'LHD': lis_h_distances}
        )
    
    mean_spkh_corr = dis_table.corr(corr_measure)['MD']['SHD']
    mean_msg_corr = dis_table.corr(corr_measure)['MD']['MSD']
    mean_lish_corr = dis_table.corr(corr_measure)['MD']['LHD']

    return mean_spkh_corr, mean_msg_corr, mean_lish_corr


def check_distance_set(distances):
    if distances.sum() == 0:
        distances += 0.1
        distances[-1] -= 0.01
    return distances


def reproduce_spk_hidden_set(model, batch_set):
    hidden_set = []

    for batch in batch_set:
        hidden = model.reproduce_speaker_hidden(batch)
        hidden_set.append(hidden.squeeze().detach().cpu().numpy())

    return hidden_set

def reproduce_msg_set(model, batch_set):
    msg_set = []

    for batch in batch_set:
        message = model.reproduce_message(batch)
        message = message.squeeze().detach().cpu().numpy()
        msg_array = []
        for r_idx in range(message.shape[0]):
            cur_v = np.argmax(message[r_idx])
            msg_array.append(cur_v)
        msg_set.append(np.asarray(msg_array))

    return msg_set

def reproduce_lis_hidden_set(model, batch_set):
    hidden_set = []

    for batch in batch_set:
        hidden = model.reproduce_listener_hidden(batch)
        hidden_set.append(hidden.squeeze().detach().cpu().numpy())

    return hidden_set

def get_hidden_dis(h1, h2, measure='euclidean'):
    if measure == 'euclidean':
        return distances.euclid_dis(h1, h2)
    else:
        raise NotImplementedError


def get_in_dis(in1, in2, measure='hamming'):
    if measure == 'hamming':
        return distances.in_ham_dis(in1, in2)
    elif measure == 'edit':
        return distances.in_edit_dis(in1, in2)
    elif measure == 'euclidean':
        return distances.euclid_dis(in1, in2)
    else:
        raise NotImplementedError


def get_msg_dis(m1, m2, measure='edit'):
    if measure == 'edit':
        return distances.editdistance.eval(m1, m2)
    elif measure == 'hamming':
        return distances.msg_ham_dis(m1, m2)
    elif measure == 'euclidean':
        return distances.euclid_dis(m1, m2)
    else:
        raise NotImplementedError
