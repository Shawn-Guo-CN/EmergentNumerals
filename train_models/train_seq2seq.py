from utils.conf import *
from models.Seq2Seq import *
from utils.DatasetLoader import *
from utils.Preprocesses import *


def train_epoch(input_variable, target_variable, encoder, decoder, 
          encoder_optimizer, decoder_optimizer, batch_size=BATCH_SIZE, clip=CLIP, max_length=MAX_LENGTH):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(DEVICE)
    target_variable = target_variable.to(DEVICE)

    # Initialize variables
    loss = 0
    print_losses = []

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_TOKEN for _ in range(batch_size)]])
    decoder_input = decoder_input.to(DEVICE)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            loss += -torch.log(torch.gather(decoder_output, 1, target_variable[t].view(-1, 1)).squeeze(1))
            print_losses.append(loss)
    else:
        for t in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(DEVICE)
            # Calculate and accumulate loss
            loss += -torch.log(torch.gather(decoder_output, 1, target_variable[t].view(-1, 1)).squeeze(1))
            print_losses.append(loss)

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

        if iter % SAVE_EVERY == 0:
            directory = os.path.join(SAVE_DIR, 'standard_seq2seq', str(HIDDEN_SIZE))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iter,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iter, 'checkpoint')))


if __name__ == '__main__':
    train()
