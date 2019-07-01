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
        self.encoder = SeqEncoder(self.hidden_size)
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

        encoder_input = self.embedding(input_var.t())
        _, encoder_hidden, encoder_cell = self.encoder(input_var, encoder_input, input_lengths)
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


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length=args.max_seq_len):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=args.device, dtype=torch.long) * args.sos_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([args.sos_token], device=args.device, dtype=torch.long)
        all_scores = torch.zeros([args.sos_token], device=args.device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores
