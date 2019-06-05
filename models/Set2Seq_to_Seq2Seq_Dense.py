from utils.conf import *


def mask_NLL_loss(prediction, golden_standard, mask):
    n_total = mask.sum()
    crossEntropy = -torch.log(torch.gather(prediction, 1, golden_standard.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean().to(DEVICE)
    n_correct = prediction.topk(1)[1].squeeze(1).eq(golden_standard).masked_select(mask).sum()
    return loss, n_correct.item(), n_total.item()


# Attention layer
class Attn(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, hidden, whole_input):
        # Calculate the attention weights (energies) based on the given method
        attn_weights = self.attn(
            torch.cat((hidden.transpose(0, 1).expand(-1, whole_input.size(1), -1), whole_input), 2)
        ).sigmoid()
        #  .tanh() is another feasible function

        # Transpose the attention weights and return
        return attn_weights.transpose(1, 2)

class SetEncoderLSTM(nn.Module):
    """
    This class is used to encode an input set.
    """
    def __init__(self, voc_size, hidden_size=HIDDEN_SIZE, max_length=MAX_LENGTH+2):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.memorising = nn.Embedding(voc_size, self.hidden_size)
        self.attn = Attn(hidden_size)
        self.dropout = nn.Dropout(DROPOUT_RATIO)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        
        self.init_hidden = self.init_hidden_and_cell()
        self.init_cell = self.init_hidden_and_cell()

    def forward(self, embedded_input_var):
        batch_size = embedded_input_var.shape[0]

        # Initialise the initial hidden and cell states for encoder
        last_hidden = self.init_hidden.expand(-1, batch_size, -1).contiguous()
        last_cell = self.init_cell.expand(-1, batch_size, -1).contiguous()
        
        # Forward pass through LSTM
        for t in range(NUM_WORD):
            # Calculate attention weights from the current LSTM input
            attn_weights = self.attn(last_hidden, embedded_input_var)
            # Calculate the attention weighted representation
            r = attn_weights.bmm(embedded_input_var).transpose(0, 1)
            # Forward through unidirectional LSTM
            lstm_output, (lstm_hidden, lstm_cell) = self.lstm(r, (last_hidden, last_cell))

        # Return hidden and cell state of LSTM
        return lstm_hidden, lstm_cell

    def init_hidden_and_cell(self):
        return nn.Parameter(torch.zeros(1, 1, self.hidden_size, device=DEVICE))


class MSGGeneratorLSTM(nn.Module):
    """
    This class is used to generate messages.
    """
    def __init__(self, input_size, output_size, hidden_size=HIDDEN_SIZE, dropout=DROPOUT_RATIO):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(self.input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, last_input, last_hidden, last_cell):
        # sent len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [batch size, hid dim]
        # cell = [batch size, hid dim]
        output, (hidden, cell) = self.lstm(last_input, (last_hidden, last_cell))
        
        output = output.squeeze(0)
        
        # prediction size = [batch size, vocab size]
        # prediction = F.softmax(self.out(output), dim=1)
        prediction = self.out(output)
        return prediction, hidden, cell

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

    def init_input(self):
        return torch.zeros(1, 1, self.input_size, device=DEVICE)


class SpeakingAgent(nn.Module):
    def __init__(self, embedding, voc_size, hidden_size=HIDDEN_SIZE, dropout=DROPOUT_RATIO):
        super().__init__()
        self.voc_size = voc_size
        self.hidden_size = hidden_size

        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)

        self.encoder = SetEncoderLSTM(self.voc_size, self.hidden_size)
        # The output size of decoder is the size of vocabulary for communication
        self.decoder = MSGGeneratorLSTM(MSG_VOCSIZE, self.hidden_size)

        # the init input for decoder
        self.init_decoder_input = nn.Parameter(self.decoder.init_input())

    def forward(self, embedded_input_var):
        batch_size = embedded_input_var.shape[0]

        encoder_hidden, encoder_cell = self.encoder(embedded_input_var)

        message = torch.zeros(MSG_LENGTH, batch_size, MSG_VOCSIZE, device=DEVICE)

        # Create initial decoder variables
        decoder_input = self.init_decoder_input.expand(-1, batch_size, -1)
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        
        for t in range(MSG_LENGTH):
            decoder_output, decoder_hidden, decoder_cell = \
                self.decoder(decoder_input, decoder_hidden, decoder_cell)            
            message[t] = decoder_output
            deocder_input = decoder_output
        
        return message


class MSGEncoderLSTM(nn.Module):
    def __init__(self, input_size=MSG_VOCSIZE, hidden_size=HIDDEN_SIZE, dropout=DROPOUT_RATIO):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        # Initialize LSTM; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size)
        self.init_hidden = self.init_hidden_and_cell()
        self.init_cell = self.init_hidden_and_cell()

    def forward(self, input_var):
        h0 = self.init_hidden.expand(1, input_var.shape[1], 1)
        c0 = self.init_cell.expand(1, input_var.shape[1], 1)
        outputs, (hidden, cell) = self.lstm(input_var, (h0, c0))

        return outputs, hidden, cell

    def init_hidden_and_cell(self):
        return nn.Parameter(torch.zeros(1, 1, self.hidden_size, device=DEVICE))


class DecoderLSTM(nn.Module):
    def __init__(self, output_size, hidden_size=HIDDEN_SIZE, dropout=DROPOUT_RATIO):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

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


class ListeningAgent(nn.Module):
    def __init__(self, voc_size, hidden_size=HIDDEN_SIZE, dropout=DROPOUT_RATIO):
        super().__init__()
        self.voc_size = voc_size
        self.hidden_size=hidden_size

        # universal modules
        self.embedding = nn.Embedding(self.voc_size, HIDDEN_SIZE)
        self.dropout = nn.Dropout(dropout)

        # encoder and decoder
        self.encoder = MSGEncoderLSTM(self.hidden_size)
        self.decoder = DecoderLSTM(self.voc_size, self.hidden_size)

    def forward(self, message, target_var, target_mask, target_max_len):
        batch_size = message.shape[1]

        # Initialize return variables
        loss = 0
        print_losses = []
        n_corrects = 0
        n_totals = 0
        
        # forward pass through message encoder
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(message)

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
            
            mask_loss, n_correct, n_total = mask_NLL_loss(decoder_output, target_var[t], target_mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total
            n_corrects += n_correct

        return loss, print_losses, n_corrects, n_totals


class Set2Seq_Seq2Seq(nn.Module):
    def __init__(self, voc_size, msg_length=MSG_LENGTH, msg_vocsize=MSG_VOCSIZE, 
                    hidden_size=HIDDEN_SIZE, dropout=DROPOUT_RATIO):
        super().__init__()
        self.voc_size = voc_size
        self.msg_length = msg_length
        self.msg_vocsize = msg_vocsize
        self.hidden_size = hidden_size
        self.dropout = dropout

        # For embedding inputs
        self.embedding = nn.Embedding(self.voc_size, self.hidden_size)

        # Speaking agent
        self.speaker = SpeakingAgent(self.embedding, self.voc_size, 
                                        self.hidden_size, self.dropout)
        # Listening agent
        self.listener = ListeningAgent(self.voc_size, self.hidden_size, self.dropout)
        

    def forward(self, data_batch):
        input_var = data_batch['input']
        target_var = data_batch['target']
        target_mask = data_batch['target_mask']
        target_max_len = data_batch['target_max_len']

        batch_size = input_var.shape[1]

        speaker_input = self.embedding(input_var.t())
        message = self.speaker(speaker_input)

        loss, print_losses, n_corrects, n_totals = \
            self.listener(message, target_var, target_mask, target_max_len)
        
        return loss, print_losses, n_corrects, n_totals