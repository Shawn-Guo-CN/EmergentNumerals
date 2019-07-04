import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from utils.conf import args
from models.Losses import mask_NLL_loss, seq_cross_entropy_loss
from models.Encoders import SetEncoder, SeqEncoder
from models.Decoders import SeqDecoder, MSGGeneratorLSTM, weight_init


class SpeakingAgent(nn.Module):
    def __init__(self, embedding, voc_size, hidden_size=args.hidden_size, \
            dropout=args.dropout_ratio, msg_embedding=None):
        super().__init__()
        self.voc_size = voc_size
        self.hidden_size = hidden_size

        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)

        self.encoder = SetEncoder(self.voc_size, self.hidden_size)
        # The output size of decoder is the size of vocabulary for communication
        self.decoder = MSGGeneratorLSTM(args.msg_vocsize, self.hidden_size, msg_embedding=msg_embedding)

    def forward(self, embedded_input_var, input_mask):

        encoder_hidden, encoder_cell = self.encoder(embedded_input_var, input_mask)
        message, mask, msg_probs, log_msg_prob = self.decoder(encoder_hidden, encoder_cell)

        return message, mask, msg_probs, log_msg_prob


class ListeningAgent(nn.Module):
    def __init__(self, voc_size, msg_vocsize=args.msg_vocsize, \
            hidden_size=args.hidden_size, dropout=args.dropout_ratio, msg_embedding=None):
        super().__init__()
        self.voc_size = voc_size
        self.msg_vocsize = msg_vocsize
        self.hidden_size = hidden_size

        if msg_embedding is not None:
            self.msg_embedding = msg_embedding
        else:
            self.msg_embedding = nn.Parameter(
                torch.randn(self.msg_vocsize, self.hidden_size, device=args.device)
                )

        # encoder and decoder
        self.encoder = SeqEncoder(self.hidden_size, self.hidden_size)
        self.decoder = SeqDecoder(self.voc_size, self.hidden_size)

    def forward(self, embedding, message, msg_mask, \
                target_var, target_mask, target_max_len):
        batch_size = message.shape[1]

        msg_len = msg_mask.squeeze(1).sum(dim=0)
        message = message.transpose(0, 1)

        message = F.relu(
            torch.bmm(message, self.msg_embedding.expand(batch_size, -1, -1))
        )

        _, encoder_hidden, encoder_cell = self.encoder(message, msg_len)

        decoder_outputs, _ = self.decoder(
            embedding,
            target_var,
            target_max_len,
            encoder_hidden.squeeze(0),
            encoder_cell.squeeze(0)
        )

        return decoder_outputs

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
        self.msg_embedding = nn.Parameter(
                torch.randn(self.msg_vocsize, self.hidden_size, device=args.device)
            )

        # Speaking agent
        self.speaker = SpeakingAgent(self.embedding, self.voc_size, 
                                        self.hidden_size, self.dropout, self.msg_embedding)
        # Listening agent
        self.listener = ListeningAgent(self.voc_size, self.msg_vocsize,
                                self.hidden_size, self.dropout, self.msg_embedding)
        

    def forward(self, data_batch):
        input_var = data_batch['input']
        input_mask = data_batch['input_mask']
        target_var = data_batch['target']
        target_mask = data_batch['target_mask']
        target_max_len = data_batch['target_max_len']

        speaker_input = self.embedding(input_var.t())
        message, msg_mask, _msg_probs_, log_msg_prob = self.speaker(speaker_input, input_mask)
        # message shape: [msg_max_len, batch_size, msg_voc_size]
        # msg_mask shape: [msg_max_len, 1, batch_size]

        listener_outputs = \
            self.listener(self.embedding, message, msg_mask, target_var, target_mask, target_max_len)

        loss_max_len = min(listener_outputs.shape[0], target_var.shape[0])

        loss, print_losses, seq_correct, tok_acc, seq_acc\
            = seq_cross_entropy_loss(listener_outputs, target_var, target_mask, loss_max_len)

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
                seq_correct, tok_acc, seq_acc, listener_outputs
