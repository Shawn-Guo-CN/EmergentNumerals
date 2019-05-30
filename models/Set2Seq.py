from utils.conf import *


# Luong attention layer
class Attn(nn.Module):
    def __init__(self, method=ATTN_METHOD, hidden_size=HIDDEN_SIZE):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, whole_input):
        return torch.sum(hidden * whole_input, dim=2)

    def general_score(self, hidden, whole_input):
        energy = self.attn(whole_input)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, whole_input):
        energy = self.attn(torch.cat((hidden.expand(whole_input.size(0), -1, -1), whole_input), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, whole_input):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, whole_input)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, whole_input)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, whole_input)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class EncoderLSTM(nn.Module):
    def __init__(self, embedding, hidden_size=HIDDEN_SIZE, max_length=MAX_LENGTH+2):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.embedding = embedding
        self.memorising = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn = Attn(ATTN_METHOD, hidden_size)
        self.dropout = nn.Dropout(DROPOUT_RATIO)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        
        self.init_hidden = self.init_hidden_and_cell()
        self.init_cell = self.init_hidden_and_cell()

    def forward(self, whole_input, current_input, last_hidden, last_cell):
        # Embed the current input element
        embedded = self.dropout(self.embedding(current_input))

        # Calculate the memory vectors for the whole input sequence
        memoryised = self.dropout(self.memorising(self.embedding(whole_input)))
        # Calculate attention weights from the current LSTM input
        attn_weights = self.attn(embedded, memoryised)
        # Multiply attention weights to the memory vector to get new "weighted sum" memory vector
        r = attn_weights.bmm(memoryised.transpose(0, 1))
        # Forward through unidirectional LSTM
        lstm_output, (lstm_hidden, lstm_cell) = self.lstm(r, (last_hidden, last_cell))
        # Concatenate weighted context vector and LSTM output using Luong eq. 5

        # Return hidden and cell state of LSTM
        return lstm_hidden, lstm_cell

    def init_hidden_and_cell(self):
        return nn.Parameter(torch.zeros(1, 1, self.hidden_size, device=DEVICE))


class DecoderLSTM(nn.Module):
    def __init__(self, output_size, embedding, hidden_size=HIDDEN_SIZE, dropout=DROPOUT_RATIO):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = embedding
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, last_input, last_hidden, last_cell):
        # embedded size = [1, batch size, emb dim]
        embedded = self.dropout(self.embedding(last_input))
        
        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [1, batch size, hid dim]
        # cell = [1, batch size, hid dim]
        output, (hidden, cell) = self.lstm(embedded, (last_hidden, last_cell))
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

