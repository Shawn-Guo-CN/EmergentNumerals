import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
import torch.nn.functional as F
import random

from utils.conf import args


def mask_NLL_loss(prediction, golden_standard, mask, last_eq):
    n_total = mask.sum().item()
    loss = args.loss_function(prediction, golden_standard) * mask.to(prediction.dtype)
    eq_cur = prediction.topk(1)[1].squeeze(1).eq(golden_standard).to(prediction.dtype) \
         * mask.to(prediction.dtype)
    n_correct = eq_cur.sum().item()
    eq_cur = eq_cur + (1 - mask.to(prediction.dtype)) * last_eq
    return loss, eq_cur, n_correct, n_total


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


# Attention layer
class Attn(nn.Module):
    def __init__(self, hidden_size=args.hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, hidden, whole_input, input_mask):
        attn_weights = self.attn(
            torch.cat((hidden.unsqueeze(0).transpose(0, 1).expand(-1, whole_input.size(1), -1),
                      whole_input), 2)
        ).sigmoid()
        #  .tanh() is another feasible function

        attn_weights = input_mask.transpose(0, 1).unsqueeze(-1).to(attn_weights.dtype) \
                    * attn_weights

        return attn_weights.transpose(1, 2)

class SetEncoderLSTM(nn.Module):
    def __init__(self, voc_size, hidden_size=args.hidden_size):
        super(SetEncoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.attn = Attn(hidden_size)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        
        self.init_hidden = self.init_hidden_and_cell()
        self.init_cell = self.init_hidden_and_cell()

    def forward(self, embedded_input, input_mask):
        batch_size = embedded_input.shape[0]

        last_hidden = self.init_hidden.expand(batch_size, -1).contiguous()
        last_cell = self.init_cell.expand(batch_size, -1).contiguous()
        
        for t in range(args.num_words):
            attn_weights = self.attn(last_hidden, embedded_input, input_mask)
            r = torch.bmm(attn_weights, embedded_input).squeeze(1)
            lstm_hidden, lstm_cell = self.lstm(r, (last_hidden, last_cell))

        return lstm_hidden, lstm_cell

    def init_hidden_and_cell(self):
        return nn.Parameter(torch.zeros(1, self.hidden_size, device=args.device))


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


class SpeakingAgent(nn.Module):
    def __init__(self, embedding, voc_size, hidden_size=args.hidden_size, dropout=args.dropout_ratio):
        super().__init__()
        self.voc_size = voc_size
        self.hidden_size = hidden_size

        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)

        self.encoder = SetEncoderLSTM(self.voc_size, self.hidden_size)
        # The output size of decoder is the size of vocabulary for communication
        self.decoder = MSGGeneratorLSTM(args.msg_vocsize, self.hidden_size)

    def forward(self, embedded_input_var, input_mask):

        encoder_hidden, encoder_cell = self.encoder(embedded_input_var, input_mask)
        message, mask, log_msg_prob = self.decoder(encoder_hidden, encoder_cell)

        return message, mask, log_msg_prob


class MSGEncoderLSTM(nn.Module):
    def __init__(self, input_size=args.msg_vocsize, hidden_size=args.hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        # Initialize LSTM; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.init_hidden = self.init_hidden_and_cell()
        self.init_cell = self.init_hidden_and_cell()

    def forward(self, input_var, input_mask):
        max_len = input_var.shape[0]

        last_hidden = self.init_hidden.expand(input_var.shape[1], -1).contiguous()
        last_cell = self.init_cell.expand(input_var.shape[1], -1).contiguous()
        for t in range(max_len):
            hidden, cell = self.lstm(input_var[t], (last_hidden, last_cell))
            last_hidden = input_mask[t].t() * hidden + (1 - input_mask[t].t()) * last_hidden
            last_cell = input_mask[t].t() * cell + (1 - input_mask[t].t()) * last_cell

        return last_hidden, last_cell

    def init_hidden_and_cell(self):
        return nn.Parameter(torch.zeros(1, self.hidden_size, device=args.device))


class SeqDecoderLSTM(nn.Module):
    def __init__(self, output_size, hidden_size=args.hidden_size, dropout=args.dropout_ratio):
        super(SeqDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedding, target_var, target_max_len, \
                encoder_hidden, encoder_cell):
        batch_size = target_var.shape[1]
        outputs = []

        decoder_input = embedding(
            torch.LongTensor([args.sos_index for _ in range(batch_size)]).to(args.device)
        )
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < args.teacher_ratio \
                                    and self.training else False

        if self.training:
            # During training, decode at most as long as the longest seq in batch
            decoder_len = target_max_len
        else:
            # During valid, reproduce as long as possible
            decoder_len = args.max_seq_len

        # Forward batch of sequences through decoder one time step at a time
        for t in range(decoder_len):
            decoder_hidden, decoder_cell = self.lstm(decoder_input, (decoder_hidden, decoder_cell))
            # Here we don't need to take Softmax as the CrossEntropyLoss later would
            # automatically take a Softmax operation
            decoder_output = self.out(decoder_hidden)
            outputs.append(decoder_output)

            if use_teacher_forcing:
                decoder_input = embedding(target_var[t].view(1, -1)).squeeze()
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = embedding(
                    torch.LongTensor([topi[i][0] for i in range(batch_size)]).to(args.device)
                )

        # shape of outputs: Len * Batch Size * Voc Size
        outputs = torch.stack(outputs)
        return outputs

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=args.device)


class ListeningAgent(nn.Module):
    def __init__(self, voc_size, hidden_size=args.hidden_size, dropout=args.dropout_ratio):
        super().__init__()
        self.voc_size = voc_size
        self.hidden_size = hidden_size

        # encoder and decoder
        self.encoder = MSGEncoderLSTM()
        self.decoder = SeqDecoderLSTM(self.voc_size, self.hidden_size)

    def forward(self, embedding, message, msg_mask, \
                target_var, target_mask, target_max_len):
        batch_size = message.shape[1]

        # Initialize variables
        loss = 0
        print_losses = []
        n_correct_tokens = 0
        n_total_tokens = 0
        n_correct_seqs = 0

        encoder_hidden, encoder_cell = self.encoder(message, msg_mask)

        decoder_outputs = self.decoder(
            embedding,
            target_var,
            target_max_len,
            encoder_hidden,
            encoder_cell
        )

        seq_correct = torch.ones((1, batch_size), device=args.device)
        eq_vec = torch.ones((1, batch_size), device=args.device)
        for t in range(target_max_len):
            mask_loss, eq_vec, n_correct, n_total = mask_NLL_loss(
                decoder_outputs[t],
                target_var[t],
                target_mask[t],
                eq_vec
            )
            loss += mask_loss
            print_losses.append(mask_loss.mean().item())
            n_total_tokens += n_total
            n_correct_tokens += n_correct
            seq_correct = seq_correct * eq_vec

        n_correct_seqs = seq_correct.sum().item()

        return loss, print_losses, \
            n_correct_seqs, n_correct_tokens, n_total_tokens, decoder_outputs

    def reset_params(self):
        self.apply(weight_init)


class Set2Seq2Seq(nn.Module):
    def __init__(self, voc_size, msg_length=args.max_msg_len, msg_vocsize=args.msg_vocsize, 
                    hidden_size=args.hidden_size, dropout=args.dropout_ratio):
        super().__init__()
        self.voc_size = voc_size
        self.msg_length = msg_length
        self.msg_vocsize = msg_vocsize
        self.hidden_size = hidden_size
        self.dropout = dropout

        # For embedding inputs
        self.embedding = nn.Embedding(self.voc_size, self.hidden_size)

        # Speaking agent
        self.speaker = SpeakingAgent(self.embedding, self.voc_size, 
                                        self.hidden_size, self.dropout)
        # Listening agent
        self.listener = ListeningAgent(self.voc_size, self.hidden_size, self.dropout)
        

    def forward(self, data_batch):
        input_var = data_batch['input']
        input_mask = data_batch['input_mask']
        target_var = data_batch['target']
        target_mask = data_batch['target_mask']
        target_max_len = data_batch['target_max_len']

        speaker_input = self.embedding(input_var.t())
        message, msg_mask, log_msg_prob = self.speaker(speaker_input, input_mask)
        # message shape: [msg_max_len, batch_size, msg_voc_size]
        # msg_mask shape: [msg_max_len, 1, batch_size]

        loss, print_losses, n_correct_seqs, n_correct_tokens, n_total_tokens, outputs = \
            self.listener(self.embedding, message, msg_mask, target_var, target_mask, target_max_len)

        if self.training and args.msg_mode == 'SCST':
            self.speaker.eval()
            self.listener.eval()
            msg, msg_mask, _ = self.speaker(speaker_input, input_mask)
            baseline = self.listener(self.embedding, msg, msg_mask, 
                                        target_var, target_mask, target_max_len)[0]
            self.speaker.train()
            self.listener.train()
        else:
            baseline = 0.
        
        return loss, log_msg_prob, baseline, print_losses, \
                n_correct_seqs, n_correct_tokens, n_total_tokens, outputs
