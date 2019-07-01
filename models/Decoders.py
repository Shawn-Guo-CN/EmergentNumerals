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
    elif mode == 'GUMBEL':
        cat_distr = RelaxedOneHotCategorical(tau, probs=probs)
        y_soft = cat_distr.rsample()
    elif mode == 'SOFTMAX':
        y_soft = probs
    
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


class SeqDecoder(nn.Module):
    def __init__(self, output_size, hidden_size=args.hidden_size, dropout=args.dropout_ratio):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedding, target_var, target_max_len, \
                encoder_hidden, encoder_cell):
        batch_size = target_var.shape[1]
        # Initialize variables
        outputs = []
        masks = []

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = embedding(
            torch.LongTensor([args.sos_index for _ in range(batch_size)]).to(args.device)
        )

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < args.teacher_ratio \
                                    and self.training else False

        decoding_max_len = target_max_len if self.training else args.max_seq_len

        # Forward batch of sequences through decoder one time step at a time
        for t in range(decoding_max_len):
            decoder_hidden, decoder_cell = self.lstm(decoder_input, (decoder_hidden, decoder_cell))
            # Here we don't need to take Softmax as the CrossEntropyLoss later would
            # automatically take a Softmax operation
            decoder_output = self.out(decoder_hidden)
            outputs.append(decoder_output)
            # mask is the probabilities for predicting EOS token
            masks.append(F.softmax(decoder_output, dim=1)[:, args.eos_index])

            if use_teacher_forcing:
                decoder_input = embedding(target_var[t].view(1, -1)).squeeze()
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = embedding(
                    torch.LongTensor([topi[i][0] for i in range(batch_size)]).to(args.device)
                )

        # shape of outputs: Len * Batch Size * Voc Size
        outputs = torch.stack(outputs)
        # shape of masks: Len * Batch Size
        masks = torch.stack(masks)
        return outputs, masks

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=args.device)


class MSGGeneratorLSTM(nn.Module):
    """
    This class is used to generate messages.
    """
    def __init__(self, io_size=args.msg_vocsize, hidden_size=args.hidden_size, dropout=args.dropout_ratio):
        super().__init__()
        self.input_size = io_size
        self.hidden_size = hidden_size
        self.output_size = io_size

        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(dropout)

        self.init_input = nn.Parameter(torch.zeros(1, self.input_size, device=args.device))

    def forward(self, encoder_hidden, encoder_cell):
        batch_size = encoder_hidden.size(0)
        decoder_input = self.init_input.expand(batch_size, -1)
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        message = []
        mask = []

        _mask = torch.ones((1, batch_size), device=args.device)
        log_probs = 0.
        
        for _ in range(args.max_msg_len):
            mask.append(_mask)
            decoder_hidden, decoder_cell = \
                self.lstm(decoder_input, (decoder_hidden, decoder_cell))
            probs = F.softmax(self.out(decoder_hidden), dim=1)

            if self.training:
                predict = cat_softmax(probs, mode=args.msg_mode, tau=args.tau, hard=(not args.soft), dim=1)
            else:
                predict = F.one_hot(torch.argmax(probs, dim=1), 
                                    num_classes=self.output_size).to(_mask.dtype)
            
            log_probs += torch.log((probs * predict).sum(dim=1)) * _mask.squeeze()
            _mask = _mask * (1 - predict[:, -1])
            
            message.append(predict)
            decoder_input = predict
        
        message = torch.stack(message)
        mask = torch.stack(mask)

        return message, mask, log_probs
