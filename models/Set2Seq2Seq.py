import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from utils.conf import args
from models.Losses import mask_NLL_loss, seq_cross_entropy_loss
from models.Encoders import SetEncoder, SeqEncoder
from models.Decoders import SeqDecoder
from models.Utils import weight_init


class SpeakingAgent(nn.Module):
    def __init__(
            self, voc_size, msg_vocsize, embedding, msg_embedding,
            hidden_size=args.hidden_size,
            dropout=args.dropout_ratio,
            msg_length=args.max_msg_len,
            msg_mode=args.msg_mode
        ):
        super().__init__()
        self.voc_size = voc_size
        self.msg_vocsize = msg_vocsize
        self.hidden_size = hidden_size
        self.msg_length = msg_length
        self.msg_mode = msg_mode

        if embedding is None:
            self.embedding = nn.Embedding(self.voc_size, self.hidden_size)
        else:
            self.embedding = embedding

        if msg_embedding is None:
            self.msg_embedding = nn.Embedding(self.msg_vocsize, self.hidden_size).weight
        else:
            self.msg_embedding = msg_embedding
        
        self.dropout = nn.Dropout(dropout)

        self.encoder = SetEncoder(self.voc_size, self.hidden_size)
        # The output size of decoder is the size of vocabulary for communication
        self.decoder = SeqDecoder(
            self.msg_vocsize, self.hidden_size, self.msg_vocsize,
            embedding=self.msg_embedding, role='msg'
        )

    def forward(self, input_var, input_mask, tau=1.0):
        embedded_input = self.embedding(input_var.t())
        encoder_hidden, encoder_cell = self.encoder(embedded_input, input_mask)
        message, logits, mask = self.decoder(
                encoder_hidden, encoder_cell, self.msg_length,
                mode=self.msg_mode, sample_tau=tau
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
        assert args.max_seq_len == args.num_words * args.max_len_word + 1
        self.max_out_len = args.max_seq_len

        if embedding is None:
            self.embedding = nn.Embedding(self.output_size, self.hidden_size).weight
        else:
            self.embedding = embedding

        if msg_embedding is None:
            self.msg_embedding = nn.Embedding(self.input_size, self.hidden_size).weight
        else:
            self.msg_embedding = msg_embedding
        
        self.encoder = SeqEncoder(self.hidden_size, self.hidden_size)

        self.decoder = SeqDecoder(
                self.output_size, self.hidden_size, self.output_size,
                embedding=self.embedding, role='out'
            )

    def forward(self, message, msg_mask, target_max_len):
        batch_size = message.shape[1]

        msg_len = msg_mask.squeeze(1).sum(dim=0)
        message = message.transpose(0, 1)

        if self.msg_embedding is not None:
            message = F.relu(
                torch.bmm(message, self.msg_embedding.expand(batch_size, -1, -1))
            )

        _, encoder_hidden, encoder_cell = self.encoder(message, msg_len)
        encoder_hidden = encoder_hidden.squeeze(0)
        encoder_cell = encoder_cell.squeeze(0)

        if self.training:
            decoder_max_len = target_max_len
        else:
            decoder_max_len = self.max_out_len

        _, decoder_logits, _ = self.decoder(
            encoder_hidden,
            encoder_cell,
            decoder_max_len
        )

        return decoder_logits


class Set2Seq2Seq(nn.Module):
    def __init__(self, voc_size, msg_length=args.max_msg_len, msg_vocsize=args.msg_vocsize, 
                    hidden_size=args.hidden_size, dropout=args.dropout_ratio, msg_mode=args.msg_mode):
        super().__init__()
        self.voc_size = voc_size
        self.msg_mode = msg_mode
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
            self.hidden_size, self.dropout, self.msg_length, self.msg_mode
        )
        # Listening agent, msg_embedding needs to be set as self.msg_embedding.weight
        self.listener = ListeningAgent(
            self.msg_vocsize, self.hidden_size, self.voc_size,
            self.dropout, self.embedding, self.msg_embedding,
        )
        

    def forward(self, data_batch, msg_tau=1.0):
        input_var = data_batch['input']
        input_mask = data_batch['input_mask']
        target_var = data_batch['target']
        target_mask = data_batch['target_mask']
        target_max_len = data_batch['target_max_len']

        message, msg_logits, msg_mask = self.speaker(input_var, input_mask, tau=msg_tau)
        
        spk_entropy = (F.softmax(msg_logits, dim=2) * msg_logits).sum(dim=2).sum(dim=0)
        if self.training:
            log_msg_prob = torch.sum(msg_logits * message, dim=2).sum(dim=0)
        else:
            log_msg_prob = 0.

        seq_logits = self.listener(message, msg_mask, target_max_len)
        if self.training:
            target_one_hot = F.one_hot(target_var, num_classes=self.voc_size).to(seq_logits.dtype)
            log_seq_prob = torch.sum(target_one_hot*seq_logits , dim=2).sum(dim=0)
        else:
            log_seq_prob = 0.

        loss_max_len = min(seq_logits.shape[0], target_var.shape[0])

        loss, print_losses, tok_correct, seq_correct, tok_acc, seq_acc\
            = seq_cross_entropy_loss(seq_logits, target_var, target_mask, loss_max_len)

        if self.training and self.msg_mode == 'SCST':
            self.speaker.eval()
            self.listener.eval()
            msg, _, msg_mask = self.speaker(input_var, input_mask)
            s_logits = self.listener(msg, msg_mask)
            loss_max_len = min(s_logits.shape[0], target_var.shape[0])
            baseline = seq_cross_entropy_loss(s_logits, target_var, target_mask, loss_max_len)[3]
            self.speaker.train()
            self.listener.train()
        else:
            baseline = 0.
        
        return loss, log_msg_prob, log_seq_prob, baseline, print_losses, \
                seq_correct, tok_acc, seq_acc, seq_logits, spk_entropy

    def reproduce_speaker_hidden(self, data_batch):
        if self.training:
            self.eval()
            resume_flag = True
        else:
            resume_flag = False

        input_var = data_batch['input']
        input_mask = data_batch['input_mask']
        
        embedded_input = self.speaker.embedding(input_var.t())
        hidden, _ = self.speaker.encoder(embedded_input, input_mask)
        
        if resume_flag:
            self.train()
        return hidden

    def reproduce_listener_hidden(self, data_batch):
        if self.training:
            self.eval()
            resume_flag = True
        else:
            resume_flag = False

        input_var = data_batch['input']
        input_mask = data_batch['input_mask']
        
        message, _, msg_mask = self.speaker(input_var, input_mask)

        batch_size = message.shape[1]
        msg_len = msg_mask.squeeze(1).sum(dim=0)
        message = message.transpose(0, 1)

        if self.listener.msg_embedding is not None:
            message = F.relu(
                torch.bmm(message, self.listener.msg_embedding.expand(batch_size, -1, -1))
            )

        _, hidden, _ = self.listener.encoder(message, msg_len)
        
        if resume_flag:
            self.train()

        return hidden

    def reproduce_message(self, data_batch):
        if self.training:
            self.eval()
            resume_flag = True
        else:
            resume_flag = False
        input_var = data_batch['input']
        input_mask = data_batch['input_mask']
        message, _, _ = self.speaker(input_var, input_mask)
        if resume_flag:
            self.train()
        return message

    def reset_speaker(self):
        del self.speaker
        self.speaker = SpeakingAgent(
            self.voc_size, self.msg_vocsize, self.embedding, self.msg_embedding,
            self.hidden_size, self.dropout, self.msg_length, self.msg_mode
        ).to(self.listener.msg_embedding.device)

    def reset_listener(self):
        del self.listener
        self.listener = ListeningAgent(
            self.msg_vocsize, self.hidden_size, self.voc_size,
            self.dropout, self.embedding, self.msg_embedding,
        ).to(self.speaker.embedding.weight.device)

    def reproduce_msg_probs(self, data_batch):
        if self.training:
            self.eval()
            resume_flag = True
        else:
            resume_flag = False

        input_var = data_batch['input']
        input_mask = data_batch['input_mask']
        
        _, msg_logits, _ = self.speaker(input_var, input_mask)

        if resume_flag:
            self.train()

        # Shape of return is [B, L_M, V_M] TODO: check this
        return F.softmax(msg_logits, dim=2)
