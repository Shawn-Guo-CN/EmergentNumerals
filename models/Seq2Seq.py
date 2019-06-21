import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from utils.conf import args


def mask_NLL_loss(prediction, golden_standard, mask, last_eq):
    n_total = mask.sum().item()
    loss = (args.loss_function(prediction, golden_standard) * mask.to(prediction.dtype)).mean()
    eq_cur = prediction.topk(1)[1].squeeze(1).eq(golden_standard).to(prediction.dtype) \
         * mask.to(prediction.dtype)
    n_correct = eq_cur.sum().item()
    eq_cur = eq_cur + (1 - mask.to(prediction.dtype)) * last_eq
    return loss, eq_cur, n_correct, n_total


class EncoderLSTM(nn.Module):
    def __init__(self, hidden_size=args.hidden_size, dropout=args.dropout_ratio):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        # Initialize LSTM; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.init_hidden = self.init_hidden_and_cell()
        self.init_cell = self.init_hidden_and_cell()

    def forward(self, input_var, input_embedded, input_lengths):
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(input_embedded, input_lengths, batch_first=True)

        # Forward pass through LSTM
        h0 = self.init_hidden.repeat(1, input_var.shape[1], 1)
        c0 = self.init_cell.repeat(1, input_var.shape[1], 1)
        outputs, (hidden, cell) = self.lstm(packed, (h0, c0))

        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Return output and final hidden state
        return outputs, hidden, cell

    def init_hidden_and_cell(self):
        return nn.Parameter(torch.zeros(1, 1, self.hidden_size, device=args.device))


class DecoderLSTM(nn.Module):
    def __init__(self, output_size, hidden_size=args.hidden_size, dropout=args.dropout_ratio):
        super(DecoderLSTM, self).__init__()
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


class Seq2Seq(nn.Module):
    def __init__(self, voc_size, hidden_size=args.hidden_size, dropout=args.dropout_ratio):
        super(Seq2Seq, self).__init__()
        self.voc_size = voc_size
        self.hidden_size=hidden_size

        # universal modules
        self.embedding = nn.Embedding(self.voc_size, args.hidden_size)
        self.dropout = nn.Dropout(dropout)

        # encoder and decoder
        self.encoder = EncoderLSTM(self.hidden_size)
        self.decoder = DecoderLSTM(self.voc_size, self.hidden_size)

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
