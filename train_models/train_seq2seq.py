from utils.conf import *
from models.Seq2Seq import *
from utils.DatasetLoader import *
from utils.Preprocesses import *


def train_epoch(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
          encoder_optimizer, decoder_optimizer, clip=CLIP):
    batch_size = input_variable.shape[1]
    
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden, encoder_cell = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_INDEX for _ in range(batch_size)]])
    decoder_input.to(DEVICE)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden
    decoder_cell = encoder_cell

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, decoder_cell = \
                decoder(decoder_input, decoder_hidden, decoder_cell)
            
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)

            # Calculate and accumulate loss
            mask_loss, n_total = mask_NLL_loss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, decoder_cell = \
                decoder(decoder_input, decoder_hidden, decoder_cell)
            
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(DEVICE)

            # Calculate and accumulate loss
            mask_loss, n_total = mask_NLL_loss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def dev_epoch(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, clip=CLIP):
    batch_size = input_variable.shape[1]

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden, encoder_cell = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_INDEX for _ in range(batch_size)]])
    decoder_input.to(DEVICE)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden
    decoder_cell = encoder_cell

    for t in range(max_target_len):
        decoder_output, decoder_hidden, decoder_cell = \
            decoder(decoder_input, decoder_hidden, decoder_cell)
        
        # No teacher forcing: next input is decoder's own current output
        _, topi = decoder_output.topk(1)
        decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
        decoder_input = decoder_input.to(DEVICE)

        # Calculate and accumulate loss
        mask_loss, n_total = mask_NLL_loss(decoder_output, target_variable[t], mask[t])
        loss += mask_loss
        print_losses.append(mask_loss.item() * n_total)
        n_totals += n_total

    return sum(print_losses) / n_totals


def train():
    print('loading datasets...')
    train_set, dev_set, test_set = load_train_dev_test()
    print('\t', len(train_set), len(dev_set), len(test_set), 'items in total.')
    print('done')

    voc = Voc()

    print('converting datasets...')
    train_set = string_set2indices_pair_set(voc, train_set)
    dev_set = string_set2indices_pair_set(voc, dev_set)
    # test_set = string_set2indices_set(voc, test_set)
    print('done')

    print('generating batches...')
    train_input_batches, train_input_lens_batches, \
        train_target_batches, train_target_mask_batches, \
            train_target_max_len_batches = indices_pair_set2data_batches(train_set)
    dev_input_batches, dev_input_lens_batches, \
        dev_target_batches, dev_target_mask_batches, \
            dev_target_max_len_batches = indices_pair_set2data_batches(dev_set)
    # test_input_batches, test_target_batches = indices_set2data_batches(test_set)
    print('done')
    
    print('building model...')
    embedding = nn.Embedding(voc.num_words, HIDDEN_SIZE).to(DEVICE)
    encoder = EncoderLSTM(embedding).to(DEVICE)
    decoder = DecoderLSTM(voc.num_words, embedding).to(DEVICE)
    encoder_optimizer = OPTIMISER(encoder.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = OPTIMISER(decoder.parameters(), lr=LEARNING_RATE * DECODER_LEARING_RATIO)
    if PARAM_FILE is not None:
        print('\tloading saved parameters from ' + PARAM_FILE + '...')
        checkpoint = torch.load(PARAM_FILE)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])
        embedding.load_state_dict(checkpoint['embedding'])
        voc = checkpoint['voc']
        print('\tdone')
    print('done')
    
    print('initiprint_lossesalising...')
    start_iteration = 1
    print_loss = 0
    print('done')

    print('training...')
    for iter in range(start_iteration, NUM_ITERS+1):
        for batch_index in range(0, len(train_input_batches)):
            input_batch = train_input_batches[batch_index]
            input_len_batch = train_input_lens_batches[batch_index]
            target_batch = train_target_batches[batch_index]
            target_mask_batch = train_target_mask_batches[batch_index]
            target_max_len_batch = train_target_max_len_batches[batch_index]

            loss = train_epoch(
                input_batch,
                input_len_batch,
                target_batch,
                target_mask_batch,
                target_max_len_batch,
                encoder,
                decoder,
                encoder_optimizer, 
                decoder_optimizer
            )
            print_loss += loss

        if iter % PRINT_EVERY == 0:
            print_loss_avg = print_loss / PRINT_EVERY
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(
                iter, iter / NUM_ITERS * 100, print_loss_avg))
            print_loss = 0

        if iter % SAVE_EVERY == 0:
            directory = os.path.join(SAVE_DIR, 'standard_seq2seq_' + str(HIDDEN_SIZE))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iter,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc': voc,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iter, 'checkpoint')))

        if iter % EVAL_EVERY == 0:
            encoder.eval()
            decoder.eval()
            loss = 0
            for batch_index in range(0, len(dev_input_batches)):
                input_batch = dev_input_batches[batch_index]
                input_len_batch = dev_input_lens_batches[batch_index]
                target_batch = dev_target_batches[batch_index]
                target_mask_batch = dev_target_mask_batches[batch_index]
                target_max_len_batch = dev_target_max_len_batches[batch_index]
                
                loss += dev_epoch(
                    input_batch,
                    input_len_batch,
                    target_batch,
                    target_mask_batch,
                    target_max_len_batch,
                    encoder,
                    decoder
                )

            print("[EVAL]Iteration: {}; Loss: {:.4f}".format(iter, loss/len(dev_input_batches)))
            encoder.train()
            decoder.train()


if __name__ == '__main__':
    train()
