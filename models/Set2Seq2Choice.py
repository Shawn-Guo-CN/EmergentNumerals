import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from utils.conf import args
from models.Losses import choice_cross_entropy_loss
from models.Encoders import SetEncoder, SeqEncoder
from models.Decoders import SeqDecoder
from models.Utils import weight_init
from models.Set2Seq2Seq import SpeakingAgent


class ListeningAgent(nn.Module):
    def __init__(
            self, voc_size, msg_vocsize, hidden_size, 
            dropout=args.dropout_ratio,
            embedding=None,
            msg_embedding=None
        ):
        super().__init__()
        self.voc_size = voc_size
        self.msg_vocsize = msg_vocsize
        self.hidden_size = hidden_size

        if embedding is None:
            self.embedding = nn.Embedding(self.voc_size, self.hidden_size)
        else:
            self.embedding = embedding

        if msg_embedding is None:
            self.msg_embedding = nn.Embedding(self.msg_vocsize, self.hidden_size).weight
        else:
            self.msg_embedding = msg_embedding
        
        self.msg_encoder = SeqEncoder(self.hidden_size, self.hidden_size)
        self.can_encoder = SetEncoder(self.hidden_size, self.hidden_size)

    def forward(self, message, msg_mask, candidates):
        batch_size = message.shape[1]

        msg_len = msg_mask.squeeze(1).sum(dim=0)
        message = message.transpose(0, 1)

        if self.msg_embedding is not None:
            message = F.relu(
                torch.bmm(message, self.msg_embedding.expand(batch_size, -1, -1))
            )

        _, msg_encoder_hidden, _ = self.msg_encoder(message, msg_len)
        msg_encoder_hidden = msg_encoder_hidden.transpose(0, 1).transpose(1, 2)

        can_encoder_hiddens = []
        for candidate in candidates:
            input_var = candidate['input']
            input_mask = candidate['input_mask']
            embedded_input = self.embedding(input_var.t())
            encoder_hidden, _ = self.can_encoder(embedded_input, input_mask)
            can_encoder_hiddens.append(encoder_hidden)

        can_encoder_hiddens = torch.stack(can_encoder_hiddens).transpose(0, 1)

        choose_logits = torch.bmm(can_encoder_hiddens, msg_encoder_hidden).squeeze(2)
        return choose_logits


class Set2Seq2Choice(nn.Module):
    def __init__(self, voc_size, msg_length=args.max_msg_len, msg_vocsize=args.msg_vocsize, 
                    hidden_size=args.hidden_size, dropout=args.dropout_ratio):
        super().__init__()
        self.voc_size = voc_size
        self.msg_length = msg_length
        self.msg_vocsize = msg_vocsize
        self.hidden_size = hidden_size
        self.dropout = dropout

        # For embedding inputs
        self.embedding = None
        self.msg_embedding = None
        
        # Speaking agent, msg_embedding needs to be set as self.msg_embedding.weight
        self.speaker = SpeakingAgent(
            self.voc_size, self.msg_vocsize, self.embedding, self.msg_embedding,
            self.hidden_size, self.dropout
        )
        # Listening agent, msg_embedding needs to be set as self.msg_embedding.weight
        self.listener = ListeningAgent(
            self.voc_size, self.msg_vocsize, self.hidden_size,
            self.dropout, self.embedding, self.msg_embedding,
        )
        

    def forward(self, data_batch):
        correct_data = data_batch['correct']
        candidates = data_batch['candidates']
        golden_label = data_batch['label']

        input_var = correct_data['input']
        input_mask = correct_data['input_mask']
        message, msg_logits, msg_mask = self.speaker(input_var, input_mask)
        
        spk_entropy = (F.softmax(msg_logits, dim=2) * msg_logits).sum(dim=2).sum(dim=0)
        if self.training:
            log_msg_prob = torch.sum(msg_logits * message, dim=2).sum(dim=0)
        else:
            log_msg_prob = 0.

        choose_logits = self.listener(message, msg_mask, candidates)
        log_choose_prob = torch.sum(choose_logits, dim=1)
        
        loss, print_loss, acc, c_correct = choice_cross_entropy_loss(choose_logits, golden_label)

        if self.training and args.msg_mode == 'SCST':
            self.speaker.eval()
            self.listener.eval()
            _msg_, _, _msg_mask_ = self.speaker(input_var, input_mask)
            _choose_logits = self.listener(_msg_, _msg_mask_, candidates)
            _, _, _, baseline = choice_cross_entropy_loss(_choose_logits, golden_label)
            self.speaker.train()
            self.listener.train()
        else:
            baseline = 0.
        
        return loss, print_loss, acc, c_correct, log_msg_prob, log_choose_prob, baseline, spk_entropy

    def reproduce_speaker_hidden(self, data_batch):
        self.eval()

        input_var = data_batch['input']
        input_mask = data_batch['input_mask']
        
        embedded_input = self.speaker.embedding(input_var.t())
        hidden, _ = self.speaker.encoder(embedded_input, input_mask)
        
        self.train()
        return hidden

    def reproduce_listener_hidden(self, data_batch):
        correct_data = data_batch['correct']

        input_var = correct_data['input']
        input_mask = correct_data['input_mask']
        message, _, msg_mask = self.speaker(input_var, input_mask)

        batch_size = message.shape[1]
        msg_len = msg_mask.squeeze(1).sum(dim=0)
        message = message.transpose(0, 1)

        if self.listener.msg_embedding is not None:
            message = F.relu(
                torch.bmm(message, self.listener.msg_embedding.expand(batch_size, -1, -1))
            )

        _, msg_encoder_hidden, _ = self.listener.msg_encoder(message, msg_len)

        return msg_encoder_hidden

    def reproduce_message(self, data_batch):
        self.eval()
        input_var = data_batch['input']
        input_mask = data_batch['input_mask']
        message, _, _ = self.speaker(input_var, input_mask)
        self.train()
        return message

    def reset_speaker(self):
        del self.speaker
        self.speaker = SpeakingAgent(
            self.voc_size, self.msg_vocsize, self.embedding, self.msg_embedding,
            self.hidden_size, self.dropout
        ).to(self.listener.msg_embedding.device)

    def reset_listener(self):
        del self.listener
        self.listener = ListeningAgent(
            self.voc_size, self.msg_vocsize, self.hidden_size,
            self.dropout, self.embedding, self.msg_embedding,
        ).to(self.speaker.embedding.weight.device)