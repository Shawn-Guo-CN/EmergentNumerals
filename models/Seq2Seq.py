import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from utils.conf import args
from models.Encoders import SeqEncoder
from models.Decoders import SeqDecoder
from models.Losses import seq_cross_entropy_loss


class Seq2Seq(nn.Module):
    def __init__(self, voc_size, hidden_size=args.hidden_size, dropout=args.dropout_ratio):
        super(Seq2Seq, self).__init__()
        self.voc_size = voc_size
        self.hidden_size=hidden_size

        # universal modules
        self.embedding = nn.Embedding(self.voc_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_embedding = nn.Embedding(self.voc_size, self.hidden_size)

        # encoder and decoder
        self.encoder = SeqEncoder(self.hidden_size, self.hidden_size)
        self.decoder = SeqDecoder(
            self.voc_size, self.hidden_size, self.voc_size,
            embedding=self.out_embedding.weight
            )

    def forward(self, data_batch):
        input_var = data_batch['input']
        input_lengths = data_batch['input_lens']
        target_var = data_batch['target']
        target_mask = data_batch['target_mask']
        target_max_len = data_batch['target_max_len']

        batch_size = input_var.shape[1]

        embedded_input = self.embedding(input_var.t())
        _, encoder_hidden, encoder_cell = self.encoder(embedded_input, input_lengths)
        encoder_hidden = encoder_hidden.squeeze(dim=0)
        encoder_cell = encoder_cell.squeeze(dim=0)

        if self.training:
            decoder_max_len = target_max_len
        else:
            decoder_max_len = args.max_seq_len

        _, decoder_logits, _ = self.decoder(
            encoder_hidden,
            encoder_cell,
            max_len=decoder_max_len
        )

        loss_max_len = min(decoder_logits.shape[0], target_var.shape[0])
        loss, print_losses, tok_correct, seq_correct, tok_acc, seq_acc\
            = seq_cross_entropy_loss(decoder_logits, target_var, target_mask, loss_max_len)

        return loss, print_losses, tok_correct, seq_correct, tok_acc, seq_acc
