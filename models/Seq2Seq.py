from utils.conf import *


class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, embedding):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded)
        # shape of output is [length * batch_size * hidden_size]
        # shape of hidden is [1 * batch_size * hidden_size] (only contains h_n)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


class DecoderGRU(nn.Module):
    def __init__(self, hidden_size, output_size, embedding):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, init_hidden):
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.gru(embedded, init_hidden)
        
        #output = [sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #sent len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.out(output.squeeze(0))
        #prediction = [batch size, output dim]
        
        return prediction, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device=DEVICE):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, input, target, voc_size, teacher_forcing_ratio=TEACHER_FORCING_RATIO):
        #input = [src sent len, batch size]
        #target = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        #tensor to store decoder outputs
        outputs = torch.zeros(MAX_LENGTH, BATCH_SIZE, voc_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        _, hidden = self.encoder(input)
        
        #first input to the decoder is the <sos> tokens
        input = target[0,:]
        
        for t in range(1, MAX_LENGTH):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (target[t] if teacher_force else top1)
        
        return outputs


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
        all_tokens = torch.zeros([0], device=DEVICE, dtype=torch.long)
        all_scores = torch.zeros([0], device=DEVICE)
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
