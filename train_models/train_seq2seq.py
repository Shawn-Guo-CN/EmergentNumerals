from utils.conf import *
from models.Seq2Seq import *
from utils.DatasetLoader import *
from utils.Preprocesses import *


def train_epoch(input_variable, target_variable, encoder, decoder, 
          encoder_optimizer, decoder_optimizer, clip=CLIP, max_length=MAX_LENGTH):

    batch_size = input_variable.shape[1]

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Initialize variables
    loss = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_TOKEN for _ in range(batch_size)]]).to(DEVICE)
    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden
    # Initilise the output tensor of decoder, where the EOS_TOKEN+1 is actually the size of voc
    decoder_outputs = torch.zeros(MAX_LENGTH+2, batch_size, EOS_TOKEN+1).to(DEVICE)
    
    #first input to the decoder is the <sos> tokens
    decoder_input = target_variable[0,:]
    
    # MAXLENGTH+1 caused by the fact that both SOS and EOS are contained
    for t in range(1, MAX_LENGTH+1):
        output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        decoder_outputs[t] = output
        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False
        top1 = output.max(1)[1]
        decoder_input = (target_variable[t] if use_teacher_forcing else top1)

    #target_variable = [trg sent len, batch size]
    #decoder_outputs = [trg sent len, batch size, output dim]
    
    decoder_outputs = decoder_outputs[1:].view(-1, decoder_outputs.shape[-1])
    target_variable = target_variable[1:].contiguous().view(-1)
    
    #trg = [(trg sent len - 1) * batch size]
    #output = [(trg sent len - 1) * batch size, output dim]

    # Calculate loss 
    loss = LOSS_FUNCTION(decoder_outputs, target_variable)
    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    nn.utils.clip_grad_norm_(encoder.parameters(), CLIP)
    nn.utils.clip_grad_norm_(decoder.parameters(), CLIP)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


def train():
    print('loading datasets...')
    train_set, dev_set, test_set = load_train_dev_test()
    print('\t', len(train_set), len(dev_set), len(test_set), 'items in total.')
    print('done')

    voc = Voc()

    print('converting datasets...')
    train_set = string_set2indices_set(voc, train_set)
    dev_set = string_set2indices_set(voc, dev_set)
    # test_set = string_set2indices_set(voc, test_set)
    print('done')

    print('generating batches...')
    train_input_batches, train_target_batches = indices_set2data_batches(train_set)
    dev_input_batches, dev_target_batches = indices_set2data_batches(dev_set)
    # test_input_batches, test_target_batches = indices_set2data_batches(test_set)
    print('done')
    
    print('building model...')
    embedding = nn.Embedding(voc.num_words, HIDDEN_SIZE)
    encoder = EncoderGRU(voc.num_words, HIDDEN_SIZE, embedding)
    decoder = DecoderGRU(HIDDEN_SIZE, voc.num_words, embedding)
    encoder_optimizer = OPTIMISER(encoder.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = OPTIMISER(decoder.parameters(), lr=LEARNING_RATE * DECODER_LEARING_RATIO)
    print('done')
    
    print('initiprint_lossesalising...')
    start_iteration = 1
    print_loss = 0
    print('done')

    print('training...')
    for iter in range(start_iteration, NUM_ITERS+1):
        for batch_index in range(0, len(train_input_batches)):
            input_batch = train_input_batches[batch_index]
            target_batch = train_target_batches[batch_index]

            loss = train_epoch(input_batch, target_batch, encoder, decoder, encoder_optimizer,  decoder_optimizer)
            print_loss += loss

        if iter % PRINT_EVERY == 0:
            print_loss_avg = print_loss / PRINT_EVERY
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(
                iter, iter / NUM_ITERS * 100, print_loss_avg))
            print_loss = 0


if __name__ == '__main__':
    train()
