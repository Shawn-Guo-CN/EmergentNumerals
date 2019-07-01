import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from utils.conf import args


class Attn(nn.Module):
    def __init__(self, hidden_size=args.hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, hidden, whole_input, input_mask):
        # Calculate the attention weights (energies) based on the given method
        attn_weights = self.attn(
            torch.cat((hidden.unsqueeze(0).transpose(0, 1).expand(-1, whole_input.size(1), -1),
                      whole_input), 2)
        ).sigmoid()
        #  .tanh() is another feasible function

        attn_weights = input_mask.transpose(0, 1).unsqueeze(-1).to(attn_weights.dtype) \
                    * attn_weights

        # Tranpose the attention weights
        return attn_weights.transpose(1, 2)


class SeqEncoder(nn.Module):
    def __init__(self, in_size=args.hidden_size, hidden_size=args.hidden_size, dropout=args.dropout_ratio):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size

        # Initialize LSTM; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.lstm = nn.LSTM(in_size, hidden_size)
        self.init_hidden = self.init_hidden_and_cell()
        self.init_cell = self.init_hidden_and_cell()

    def forward(self, input_embedded, input_lengths):
        """
        inpug_embedded shape: [B, L, I]
        input_lengths shape: [B]
        """
        batch_size = input_embedded.shape[0]

        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(
            input_embedded, input_lengths, 
            batch_first=True, enforce_sorted=False)

        # Forward pass through LSTM
        h0 = self.init_hidden.expand(1, batch_size, -1).contiguous()
        c0 = self.init_cell.expand(1, batch_size, -1).contiguous()
        outputs, (hidden, cell) = self.lstm(packed, (h0, c0))

        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Return output and final hidden state
        return outputs, hidden, cell

    def init_hidden_and_cell(self):
        return nn.Parameter(torch.zeros(1, 1, self.hidden_size, device=args.device))


class SetEncoder(nn.Module):
    def __init__(self, voc_size, hidden_size=args.hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.attn = Attn(hidden_size)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        
        self.init_hidden = self.init_hidden_and_cell()
        self.init_cell = self.init_hidden_and_cell()

    def forward(self, embedded_input, input_mask):
        batch_size = embedded_input.shape[0]

        # Initialise the initial hidden and cell states for encoder
        last_hidden = self.init_hidden.expand(batch_size, -1).contiguous()
        last_cell = self.init_cell.expand(batch_size, -1).contiguous()
        
        # Forward pass through LSTM
        for t in range(args.num_words):
            # Calculate attention weights from the current LSTM input
            attn_weights = self.attn(last_hidden, embedded_input, input_mask)
            # Calculate the attention weighted representation
            r = torch.bmm(attn_weights, embedded_input).squeeze(1)
            # Forward through unidirectional LSTM
            lstm_hidden, lstm_cell = self.lstm(r, (last_hidden, last_cell))

        # Return hidden and cell state of LSTM
        return lstm_hidden, lstm_cell

    def init_hidden_and_cell(self):
        return nn.Parameter(torch.zeros(1, self.hidden_size, device=args.device))
