import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from utils.conf import args


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

        # Forward batch of sequences through decoder one time step at a time
        for t in range(target_max_len):
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