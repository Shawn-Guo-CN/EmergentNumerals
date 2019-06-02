from utils.conf import *


def mask_NLL_loss(prediction, golden_standard, mask):
    n_total = mask.sum()
    crossEntropy = -torch.log(torch.gather(prediction, 1, golden_standard.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(DEVICE)
    return loss, n_total.item()


class EncoderLSTM(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, dropout=DROPOUT_RATIO):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        # Initialize LSTM; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.init_hidden = self.init_hidden_and_cell()
        self.init_cell = self.init_hidden_and_cell()

    def forward(self, input_var, input_embedded, input_lengths):
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(input_embedded, input_lengths)

        # Forward pass through LSTM
        h0 = self.init_hidden.repeat(1, input_var.shape[1], 1)
        c0 = self.init_cell.repeat(1, input_var.shape[1], 1)
        outputs, (hidden, cell) = self.lstm(packed, (h0, c0))

        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Return output and final hidden state
        return outputs, hidden, cell

    def init_hidden_and_cell(self):
        return nn.Parameter(torch.zeros(1, 1, self.hidden_size, device=DEVICE))


class DecoderLSTM(nn.Module):
    def __init__(self, output_size, hidden_size=HIDDEN_SIZE, dropout=DROPOUT_RATIO):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, last_input, last_hidden, last_cell):
        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [1, batch size, hid dim]
        # cell = [1, batch size, hid dim]
        output, (hidden, cell) = self.lstm(last_input, (last_hidden, last_cell))
        # sent len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [batch size, hid dim]
        # cell = [batch size, hid dim]
        output = output.squeeze(0)
        
        #prediction size = [batch size, output dim]
        prediction = F.softmax(self.out(output), dim=1)
        return prediction, hidden, cell

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


class Seq2Seq(nn.Module):
    def __init__(self, voc_size, hidden_size=HIDDEN_SIZE, dropout=DROPOUT_RATIO):
        super(Seq2Seq, self).__init__()
        self.voc_size = voc_size
        self.hidden_size=hidden_size

        # universal modules
        self.embedding = nn.Embedding(self.voc_size, HIDDEN_SIZE).to(DEVICE)
        self.dropout = nn.Dropout(dropout)

        # encoder and decoder
        self.encoder = EncoderLSTM(self.hidden_size)
        self.decoder = DecoderLSTM(self.voc_size, self.hidden_size)

    def forward(self, data_batch):
        input_var = data_batch['input']
        input_lens = data_batch['input_lens']
        target_var = data_batch['target']
        target_mask = data_batch['target_mask']
        target_max_len = data_batch['target_max_len']

        loss = 0
        print_losses = []
        n_totals = 0

        batch_size = input_var.shape[1]
        # forward pass through encoder
        input_embedded = self.embedding(input_var)
        encoder_outputs, encoder_hidden, encoder_cell = \
            self.encoder(input_var, input_embedded, input_lens)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = \
            self.embedding(torch.LongTensor([[SOS_INDEX for _ in range(batch_size)]]).to(DEVICE))

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO \
                                        and self.training else False

        # Forward batch of sequences through decoder one time step at a time
        for t in range(target_max_len):
            decoder_output, decoder_hidden, decoder_cell = \
               self.decoder(decoder_input, decoder_hidden, decoder_cell)
            
            if use_teacher_forcing:
                decoder_input = self.embedding(target_var[t].view(1, -1))
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = self.embedding(
                    torch.LongTensor([[topi[i][0] for i in range(batch_size)]]).to(DEVICE)
                )
            
            mask_loss, n_total = mask_NLL_loss(decoder_output, target_var[t], target_mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total

        return loss, print_losses, n_totals


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
