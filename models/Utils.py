import torch
import torch.nn as nn

def weight_init(m):
    if isinstance(m, nn.Parameter):
        nn.init.xavier_normal_(m.weight.data)