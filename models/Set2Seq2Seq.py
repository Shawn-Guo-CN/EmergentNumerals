import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from utils.conf import args
from models.Losses import mask_NLL_loss, seq_cross_entropy_loss
from models.Encoders import SetEncoder, SeqEncoder
from models.Decoders import SeqDecoder, weight_init


class SpeakingAgent(nn.Module):
    def __init__(
            self, voc_size, msg_vocsize, embedding, msg_embedding,
            hidden_size=args.hidden_size,
            dropout=args.dropout_ratio,
        ):
        super().__init__()
        self.voc_size = voc_size
        self.msg_vocsize = msg_vocsize
        self.hidden_size = hidden_size

        self.embedding = embedding
        self.msg_embedding = msg_embedding
        self.dropout = nn.Dropout(dropout)

        self.encoder = SetEncoder(self.voc_size, self.hidden_size)
        # The output size of decoder is the size of vocabulary for communication
        self.decoder = SeqDecoder(
            self.msg_vocsize, self.hidden_size, self.msg_vocsize,
            embedding=self.msg_embedding
        )

    def forward(self, input_var, input_mask):
        embedded_input = self.embedding(input_var.t())
        encoder_hidden, encoder_cell = self.encoder(embedded_input, input_mask)
        message, logits, mask = self.decoder(
                encoder_hidden, encoder_cell,
                mode=args.msg_mode,
                max_len=args.max_msg_len,
                eos_idx=-1,
            )

        return message, logits, mask


class ListeningAgent(nn.Module):
    def __init__(
            self, input_size, hidden_size, output_size,
            dropout=args.dropout_ratio,
            embedding=None,
            msg_embedding=None
        ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Parameter(
                torch.randn(self.output_size, self.hidden_size, device=args.device)
            )

        if msg_embedding is not None:
            self.msg_embedding = msg_embedding
        else:
            self.msg_embedding = nn.Parameter(
                    torch.randn(self.msg_vocsize, self.hidden_size, device=args.device)
                )

        # encoder and decoder
        self.encoder = SeqEncoder(self.hidden_size, self.hidden_size)
        self.decoder = SeqDecoder(
                self.output_size, self.hidden_size, self.output_size,
                embedding=self.embedding
            )

    def forward(self, message, msg_mask, target_max_len):
        batch_size = message.shape[1]

        msg_len = msg_mask.squeeze(1).sum(dim=0)
        message = message.transpose(0, 1)

        message = F.relu(
            torch.bmm(message, self.msg_embedding.expand(batch_size, -1, -1))
        )

        _, encoder_hidden, encoder_cell = self.encoder(message, msg_len)
        encoder_hidden = encoder_hidden.squeeze(0)
        encoder_cell = encoder_cell.squeeze(0)

        if self.training:
            decoder_max_len = target_max_len
        else:
            decoder_max_len = args.max_seq_len

        _, decoder_logits, _ = self.decoder(
            encoder_hidden,
            encoder_cell,
            max_len=decoder_max_len
        )

        return decoder_logits

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
        self.msg_embedding = nn.Embedding(self.voc_size, self.hidden_size)

        # Speaking agent
        self.speaker = SpeakingAgent(
            self.voc_size, self.msg_vocsize, self.embedding, self.msg_embedding.weight,
            self.hidden_size, self.dropout
        )
        # Listening agent
        self.listener = ListeningAgent(
            self.msg_vocsize, self.hidden_size, self.voc_size,
            self.dropout, self.embedding.weight, self.msg_embedding.wight
        )
        

    def forward(self, data_batch):
        input_var = data_batch['input']
        input_mask = data_batch['input_mask']
        target_var = data_batch['target']
        target_mask = data_batch['target_mask']
        target_max_len = data_batch['target_max_len']

        message, msg_logits, msg_mask = self.speaker(input_var, input_mask)

        log_msg_prob = torch.sum(msg_logits, dim=1)

        listener_outputs = self.listener(message, msg_mask, target_max_len)

        loss_max_len = min(listener_outputs.shape[0], target_var.shape[0])

        loss, print_losses, tok_correct, seq_correct, tok_acc, seq_acc\
            = seq_cross_entropy_loss(listener_outputs, target_var, target_mask, loss_max_len)

        if self.training and args.msg_mode == 'SCST':
            self.speaker.eval()
            self.listener.eval()
            msg, _, msg_mask = self.speaker(input_var, input_mask)
            l_outputs = self.listener(msg, msg_mask, args.max_seq_len)
            loss_max_len = min(listener_outputs.shape[0], target_var.shape[0])
            baseline = seq_cross_entropy_loss(l_outputs, target_var, target_mask, loss_max_len)[0]
            self.speaker.train()
            self.listener.train()
        else:
            baseline = 0.
        
        return loss, log_msg_prob, baseline, print_losses, \
                seq_correct, tok_acc, seq_acc, listener_outputs
