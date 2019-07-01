import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from utils.conf import args
from models.Encoders import SeqEncoder
from models.Decoders import SeqDecoder
from models.Losses import mask_NLL_loss


class Seq2Seq(nn.Module):
    def __init__(self, voc_size, hidden_size=args.hidden_size, dropout=args.dropout_ratio):
        super(Seq2Seq, self).__init__()
        self.voc_size = voc_size
        self.hidden_size=hidden_size

        # universal modules
        self.embedding = nn.Embedding(self.voc_size, args.hidden_size)
        self.dropout = nn.Dropout(dropout)

        # encoder and decoder
        self.encoder = SeqEncoder(self.hidden_size, self.hidden_size)
        self.decoder = SeqDecoder(self.voc_size, self.hidden_size)

    def forward(self, data_batch):
        input_var = data_batch['input']
        input_lengths = data_batch['input_lens']
        target_var = data_batch['target']
        target_mask = data_batch['target_mask']
        target_max_len = data_batch['target_max_len']

        # Initialize variables
        loss = 0
        print_losses = []
        n_correct_tokens = 0
        n_total_tokens = 0
        n_correct_seqs = 0

        embedded_input = self.embedding(input_var.t())
        _, encoder_hidden, encoder_cell = self.encoder(embedded_input, input_lengths)
        encoder_hidden = encoder_hidden.squeeze()
        encoder_cell = encoder_cell.squeeze()

        decoder_outputs, _ = self.decoder(
            self.embedding,
            target_var,
            target_max_len,
            encoder_hidden,
            encoder_cell
        )

        seq_correct = torch.ones([input_var.shape[1]], device=args.device)
        eq_vec = torch.ones([input_var.shape[1]], device=args.device)
        for t in range(target_max_len):
            mask_loss, eq_vec, n_correct, n_total = mask_NLL_loss(
                decoder_outputs[t], 
                target_var[t], 
                target_mask[t],
                eq_vec
            )
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_total_tokens += n_total
            n_correct_tokens += n_correct
            seq_correct = seq_correct * eq_vec

        n_correct_seqs = seq_correct.sum().item()

        return loss, print_losses, n_correct_seqs, n_correct_tokens, n_total_tokens
