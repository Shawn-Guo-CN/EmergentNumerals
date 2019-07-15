import torch
import torch.nn as nn

def weight_init(m):
    if isinstance(m, nn.Parameter) or isinstance(m.weight, nn.Parameter):
        nn.init.xavier_uniform_(m.weight)