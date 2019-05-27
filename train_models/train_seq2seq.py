from utils.conf import *
from models.Seq2Seq import *
from utils.DatasetLoader import *
from utils.Preprocesses import *


def train_epoch(input_variable, target_variable, encoder, decoder, 
          encoder_optimizer, decoder_optimizer, batch_size=BATCH_SIZE, clip=CLIP, max_length=MAX_LENGTH):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Initialize variables
    loss = 0
    print_losses = []

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_TOKEN for _ in range(batch_size)]]).to(DEVICE)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False

    # TODO: need to fill in the seq2seq model
    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), CLIP)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), CLIP)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses)


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
    # TODO: need to fill in the seq2seq model
    encoder_optimizer = OPTIMISER(encoder.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = OPTIMISER(decoder.parameters(), lr=LEARNING_RATE * DECODER_LEARING_RATIO)
    print('done')
    
    print('initialising...')
    start_iteration = 1
    print_loss = 0
    print('done')

    print('training...')
    for iter in range(start_iteration, NUM_ITERS+1):
        input_batch = train_input_batches[iter - 1]
        target_batch = train_target_batches[iter - 1]

        loss = train_epoch(input_batch, target_batch, encoder, decoder, encoder_optimizer, decoder_optimizer)
        print_loss += loss

        if iter % PRINT_EVERY == 0:
            print_loss_avg = print_loss / PRINT_EVERY
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(
                iter, iter / NUM_ITERS * 100, print_loss_avg))
            print_loss = 0


if __name__ == '__main__':
    train()
