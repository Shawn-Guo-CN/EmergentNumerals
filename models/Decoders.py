import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.distributions.categorical import Categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical

from utils.conf import args


def cat_softmax(probs, mode, tau=1, hard=False, dim=-1):
    if mode == 'REINFORCE' or mode == 'SCST':
        cat_distr = OneHotCategorical(probs=probs)
        return cat_distr.sample()
    elif mode == 'SOFTMAX':
        y_soft = probs
    elif mode == 'GUMBEL':
        cat_distr = RelaxedOneHotCategorical(tau, probs=probs)
        y_soft = cat_distr.rsample()
    
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(probs, device=args.device).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    return ret


def weight_init(m):
    if isinstance(m, nn.Parameter):
        torch.nn.init.xavier_normal(m.weight.data)


def decoding_sampler(logits, mode, tau=1, hard=False, dim=-1):
    if mode == 'REINFORCE' or mode == 'SCST':
        cat_distr = OneHotCategorical(logits=logits)
        return cat_distr.sample()
    elif mode == 'SOFTMAX':
        y_soft = logits
    elif mode == 'GUMBEL':
        cat_distr = RelaxedOneHotCategorical(tau, logits=logits)
        y_soft = cat_distr.rsample()
    
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, device=args.device).scatter_(dim, index, 1.0)
        if mode == 'GUMBEL':
            ret = y_hard - y_soft.detach() + y_soft
        else: # if SOFTMAX mode
            ret = y_hard
    else:
        # Reparametrization trick.
        ret = y_soft

    return ret


class SeqDecoder(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            dropout=args.dropout_ratio,
            embedding=None
        ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

        self.init_input = nn.Parameter(torch.zeros(1, self.hidden_size, device=args.device))

        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Parameter(
                torch.randn(self.input_size, self.hidden_size, device=args.device)
            )

    def forward(self, encoder_hidden, encoder_cell, 
                max_len=args.max_seq_len,
                init_input=None,
                mode='SOFTMAX',
                sample_hard=True,
                eos_idx=args.eos_index
        ):
        batch_size = encoder_hidden.shape[0]
        
        predicts = []
        logits = []
        masks = []

        # Create initial decoder input (start with SOS tokens for each sentence)
        if init_input is None:
            decoder_input = self.init_input.expand(batch_size, -1)
        else:
            decoder_input = init_input

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        mask = torch.ones((1, batch_size), device=args.device)

        # Forward batch of sequences through decoder one time step at a time
        for t in range(max_len):
            masks.append(mask)

            decoder_hidden, decoder_cell = self.lstm(decoder_input, (decoder_hidden, decoder_cell))
            logit = self.out(decoder_hidden)
            logits.append(logit)

            if self.training:
                predict = decoding_sampler(logit, mode=mode, tau=args.tau, hard=sample_hard)
            else:
                predict = F.one_hot(torch.argmax(logit, dim=1), 
                                    num_classes=self.output_size).to(mask.dtype)
            
            predicts.append(predict)
            mask = mask * (1 - predict[:, eos_idx])
            
            decoder_input = torch.matmul(predict, self.embedding)

        # shape of predicts: Len * Batch Size * Voc Size
        predicts = torch.stack(predicts)
        # shape of outputs: Len * Batch Size * Voc Size
        logits = torch.stack(logits)
        # shape of masks: Len * Batch Size
        masks = torch.stack(masks)
        
        return predicts, logits, masks


class MSGGeneratorLSTM(nn.Module):
    """
    This class is used to generate messages.
    """
    def __init__(
            self,
            input_size=args.msg_vocsize,
            output_size=args.msg_vocsize,
            hidden_size=args.hidden_size, 
            dropout=args.dropout_ratio,
            msg_embedding=None
        ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(dropout)

        self.init_input = nn.Parameter(torch.zeros(1, self.hidden_size, device=args.device))

        if msg_embedding is not None:
            self.msg_embedding = msg_embedding
        else:
            self.msg_embedding = nn.Parameter(
                torch.randn(self.input_size, self.hidden_size, device=args.device)
            )

    def forward(self, encoder_hidden, encoder_cell, init_input=None):
        batch_size = encoder_hidden.shape[0]

        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        message = []
        mask = []
        digits = []

        if init_input is not None:
            decoder_input = init_input
        else:
            decoder_input = self.init_input.expand(batch_size, -1)

        _mask = torch.ones((1, batch_size), device=args.device)
        log_probs = 0.
        
        for _ in range(args.max_msg_len):
            mask.append(_mask)
            decoder_input = F.relu(decoder_input)
            decoder_hidden, decoder_cell = self.lstm(decoder_input, (decoder_hidden, decoder_cell))
            _probs = F.softmax(self.out(decoder_hidden), dim=1)
            digits.append(self.out(decoder_hidden))

            if self.training:
                predict = cat_softmax(_probs, mode=args.msg_mode, tau=args.tau, hard=(not args.soft), dim=1)
            else:
                predict = F.one_hot(torch.argmax(_probs, dim=1), 
                                    num_classes=self.output_size).to(_mask.dtype)
            
            log_probs += torch.log((_probs * predict).sum(dim=1)) # * _mask.squeeze()
            # _mask = _mask * (1 - predict[:, -1])

            message.append(predict)
            decoder_input = torch.matmul(predict, self.msg_embedding)
        
        message = torch.stack(message)
        mask = torch.stack(mask)
        digits = torch.stack(digits)

        return message, mask, digits, log_probs
