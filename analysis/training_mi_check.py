import torch
import numpy as np
import pandas as pd
import math
import analysis.distances as distances


def mi_check(
    model,
    batch_set,
):
    in_msg_prob_matrix = reproduce_all_msg_distribution(model, batch_set)
    # TODO: check that the shape of all_msg_distribution is [N_in, N_msg]

    log_in_msg_prob_matrix = torch.log(in_msg_prob_matrix)
    # TODO: check the device of following operations
    log_in_msg_prob_matrix += math.log(in_msg_prob_matrix.shape[1])

    in_msg_mi = (in_msg_prob_matrix * log_in_msg_prob_matrix).sum() / in_msg_prob_matrix.shape[0]

    return in_msg_mi


def reproduce_all_msg_distribution(
    model,
    batch_set,
):
    all_msg_distribution = None

    for data_batch in batch_set:
        msg_probs = model.reproduce_msg_probs(data_batch)
        msg_distribution = batch_msg_probs_to_msg_distribution(msg_probs)

        if all_msg_distribution is None:
            all_msg_distribution = msg_distribution
        else:
            all_msg_distribution = torch.concat((all_msg_distribution, msg_distribution), 0)
    
    # shape of all_msg_distribution: [N_in, V^L]
    return all_msg_distribution


def batch_msg_probs_to_msg_distribution(batch_msg_probs):
    assert len(batch_msg_probs.shape) == 3

    msg_distribution = None
    for i in range(batch_msg_probs.shape[1]):
        if msg_distribution is None:
            msg_distribution = batch_msg_probs[:, i, :]
        else:
            msg_distribution = torch.einsum('bi,bj->bij', (msg_distribution, batch_msg_probs[:, i, :]))
    
    # shape of msg_distribution: [B, V^L] 
    return msg_distribution


if __name__ == '__main__':
    pass
