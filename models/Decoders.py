import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.distributions.categorical import Categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical

from utils.conf import args


def decoding_sampler(logits, mode, tau=1, hard=False, dim=-1):
    if mode == 'REINFORCE' or mode == 'SCST':
        cat_distr = OneHotCategorical(logits=logits)
        return cat_distr.sample()
    elif mode == 'GUMBEL':
        cat_distr = RelaxedOneHotCategorical(tau, logits=logits)
        y_soft = cat_distr.rsample()
    else: # mode == 'SOFTMAX':
        y_soft = logits
    
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

        self.embedding = embedding

        if embedding is None:
            self.lstm = nn.LSTMCell(output_size, hidden_size)
            self.init_input = nn.Parameter(torch.zeros(1, self.output_size, device=args.device))
        else:
            self.lstm = nn.LSTMCell(hidden_size, hidden_size)
            self.init_input = nn.Parameter(torch.zeros(1, self.hidden_size, device=args.device))
        
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_hidden, encoder_cell, 
                max_len=args.max_seq_len,
                init_input=None,
                mode='SOFTMAX',
                sample_hard=True,
                eos_idx=None
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

            if eos_idx is not None:
                mask = mask * (1 - predict[:, eos_idx]) # for variable lengths
            
            if self.embedding is not None:
                decoder_input = torch.matmul(predict, self.embedding)
            else:
                decoder_input = predict

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
    def __init__(self, io_size=args.msg_vocsize, hidden_size=args.hidden_size, dropout=args.drop_out):
        super().__init__()
        self.input_size = io_size
        self.hidden_size = hidden_size
        self.output_size = io_size

        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(dropout)

        self.init_input = nn.Parameter(torch.zeros(1, self.input_size, device=args.device))

    def forward(self, encoder_hidden, encoder_cell, eos_idx=-1):
        batch_size = encoder_hidden.size(0)
        decoder_input = self.init_input.expand(batch_size, -1)
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        message = []
        mask = []

        _mask = torch.ones((1, batch_size), device=args.device)
        log_probs = 0.
        
        for _ in range(args.msg_max_len):
            mask.append(_mask)
            decoder_hidden, decoder_cell = \
                self.lstm(decoder_input, (decoder_hidden, decoder_cell))
            probs = F.softmax(self.out(decoder_hidden), dim=1)

            if self.training:
                predict = cat_softmax(probs, mode=args.msg_mode, tau=args.tau, hard=(not args.soft), dim=1)
            else:
                predict = F.one_hot(torch.argmax(probs, dim=1), 
                                    num_classes=self.output_size).to(_mask.dtype)
            
            log_probs += torch.log((probs * predict).sum(dim=1)).dot(_mask.squeeze())
            _mask = _mask * (1 - predict[:, eos_idx])
            
            message.append(predict)
            deocder_input = predict
        
        message = torch.stack(message)
        mask = torch.stack(mask)

        return message, mask, log_probs
