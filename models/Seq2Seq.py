from utils.conf import *


def mask_NLL_loss(prediction, golden_standard, mask):
    n_total = mask.sum()
    crossEntropy = -torch.log(torch.gather(prediction, 1, golden_standard.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(DEVICE)
    return loss, n_total.item()


class EncoderLSTM(nn.Module):
    def __init__(self, embedding, hidden_size=HIDDEN_SIZE, dropout=DROPOUT_RATIO):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.dropout = dropout

        # Initialize LSTM; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.init_hidden = self.init_hidden_and_cell()
        self.init_cell = self.init_hidden_and_cell()

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through LSTM
        h0 = self.init_hidden.repeat(1, input_seq.shape[1], 1)
        c0 = self.init_cell.repeat(1, input_seq.shape[1], 1)
        outputs, (hidden, cell) = self.lstm(packed, (h0, c0))
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Return output and final hidden state
        return outputs, hidden

    def init_hidden_and_cell(self):
        return nn.Parameter(torch.zeros(1, 1, self.hidden_size, device=DEVICE))


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, embedding, dropout=DROPOUT_RATIO):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = embedding
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, last_hidden):
        input = input.unsqueeze(0)
        # embedded size = [1, batch size, emb dim]
        embedded = self.dropout(self.embedding(input))
        
        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [batch size, hid dim]
        # cell = [batch size, hid dim]
        output, (hidden, cell) = self.lstm(embedded, last_hidden)
        # sent len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [batch size, hid dim]
        # cell = [batch size, hid dim]
        
        #prediction size = [batch size, output dim]
        prediction = self.out(output.squeeze(0))
        return prediction, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length=MAX_LENGTH):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=DEVICE, dtype=torch.long) * SOS_TOKEN
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([SOS_TOKEN], device=DEVICE, dtype=torch.long)
        all_scores = torch.zeros([SOS_TOKEN], device=DEVICE)
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
