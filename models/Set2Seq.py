from utils.conf import *


def mask_NLL_loss(prediction, golden_standard, mask):
    n_total = mask.sum().item()
    loss = (LOSS_FUNCTION(prediction, golden_standard) * mask.to(prediction.dtype)).mean()
    eq_vec = prediction.topk(1)[1].squeeze(1).eq(golden_standard).to(prediction.dtype) \
         * mask.to(prediction.dtype)
    n_correct = eq_vec.sum().item()
    return loss, eq_vec, n_correct, n_total


# Attention layer
class Attn(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE):
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

class EncoderLSTM(nn.Module):
    def __init__(self, voc_size, hidden_size=HIDDEN_SIZE, max_length=MAX_LENGTH+2):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.memorising = nn.Embedding(voc_size, self.hidden_size)
        self.attn = Attn(hidden_size)
        self.dropout = nn.Dropout(DROPOUT_RATIO)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        
        self.init_hidden = self.init_hidden_and_cell()
        self.init_cell = self.init_hidden_and_cell()

    def forward(self, embedded_input, input_mask):
        batch_size = embedded_input.shape[0]

        # Initialise the initial hidden and cell states for encoder
        last_hidden = self.init_hidden.expand(batch_size, -1).contiguous()
        last_cell = self.init_cell.expand(batch_size, -1).contiguous()
        
        # Forward pass through LSTM
        for t in range(NUM_WORD):
            # Calculate attention weights from the current LSTM input
            attn_weights = self.attn(last_hidden, embedded_input, input_mask)
            # Calculate the attention weighted representation
            r = torch.bmm(attn_weights, embedded_input).squeeze()
            # Forward through unidirectional LSTM
            lstm_hidden, lstm_cell = self.lstm(r, (last_hidden, last_cell))

        # Return hidden and cell state of LSTM
        return lstm_hidden, lstm_cell

    def init_hidden_and_cell(self):
        return nn.Parameter(torch.zeros(1, self.hidden_size, device=DEVICE))


class DecoderLSTM(nn.Module):
    def __init__(self, output_size, hidden_size=HIDDEN_SIZE, dropout=DROPOUT_RATIO):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedding, target_var, target_mask, target_max_len, encoder_hidden, encoder_cell):
        batch_size = target_var.shape[1]
        # Initialize variables
        outputs = []
        masks = []

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = embedding(
            torch.LongTensor([SOS_INDEX for _ in range(batch_size)]).to(DEVICE)
        )

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO \
                                    and self.training else False

        # Forward batch of sequences through decoder one time step at a time
        for t in range(target_max_len):
            decoder_hidden, decoder_cell = self.lstm(decoder_input, (decoder_hidden, decoder_cell))
            # Here we don't need to take Softmax as the CrossEntropyLoss later would
            # automatically take a Softmax operation
            decoder_output = self.out(decoder_hidden)
            outputs.append(decoder_output)
            # mask is the probabilities for predicting EOS token
            masks.append(F.softmax(decoder_output, dim=1)[:, EOS_INDEX])

            if use_teacher_forcing:
                decoder_input = embedding(target_var[t].view(1, -1)).squeeze()
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = embedding(
                    torch.LongTensor([topi[i][0] for i in range(batch_size)]).to(DEVICE)
                )

        # shape of outputs: Len * Batch Size * Voc Size
        outputs = torch.stack(outputs)
        # shape of masks: Len * Batch Size
        masks = torch.stack(masks)
        return outputs, masks

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


class Set2Seq(nn.Module):
    def __init__(self, voc_size, hidden_size=HIDDEN_SIZE, dropout=DROPOUT_RATIO):
        super(Set2Seq, self).__init__()
        self.voc_size = voc_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(self.voc_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.encoder = EncoderLSTM(self.voc_size, self.hidden_size)
        self.decoder = DecoderLSTM(self.voc_size, self.hidden_size)

    def forward(self, data_batch):
        input_var = data_batch['input']
        input_mask = data_batch['input_mask']
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
        encoder_hidden, encoder_cell = self.encoder(encoder_input, input_mask)

        decoder_outputs, _ = self.decoder(
            self.embedding,
            target_var,
            target_mask,
            target_max_len,
            encoder_hidden,
            encoder_cell
        )

        seq_correct = torch.ones([input_var.shape[1]], device=DEVICE)
        for t in range(target_max_len):
            mask_loss, eq_vec, n_correct, n_total = mask_NLL_loss(decoder_outputs[t], target_var[t], target_mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_total_tokens += n_total
            n_correct_tokens += n_correct
            seq_correct = seq_correct * eq_vec

        n_correct_seqs = seq_correct.sum().item()

        return loss, print_losses, n_correct_seqs, n_correct_tokens, n_total_tokens
