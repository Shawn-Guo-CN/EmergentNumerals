from utils.conf import *
from models.Seq2Seq import Seq2Seq
from preprocesses.DataIterator import FruitSeqDataset
from preprocesses.Voc import Voc


def train_epoch(data_batch, model, param_optimiser, decoder_optimiser, clip=CLIP):
    # Zero gradients
    param_optimiser.zero_grad()
    decoder_optimiser.zero_grad()

    # Forward pass through model
    loss, print_losses, n_totals = model(data_batch)

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    nn.utils.clip_grad_norm_(model.parameters(), clip)

    # Adjust model weights
    param_optimiser.step()
    decoder_optimiser.step()

    return sum(print_losses) / n_totals

def train():
    print('building vocabulary...')
    voc = Voc()
    print('done')

    print('loading data and building batches...')
    train_set = FruitSeqDataset(voc, dataset_file_path=TRAIN_FILE_PATH)
    dev_set = FruitSeqDataset(voc, dataset_file_path=DEV_FILE_PATH)
    # test_set = FruitSeqDataset(voc, dataset_file_path=TEST_FILE_PATH)
    print('done')
    
    print('building model...')
    seq2seq = Seq2Seq(voc.num_words).to(DEVICE)
    param_optimizer = OPTIMISER(seq2seq.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = OPTIMISER(seq2seq.decoder.parameters(), \
                                    lr=LEARNING_RATE * DECODER_LEARING_RATIO)
    if PARAM_FILE is not None:
        print('\tloading saved parameters from ' + PARAM_FILE + '...')
        checkpoint = torch.load(PARAM_FILE)
        seq2seq.load_state_dict(checkpoint['model'])
        param_optimizer.load_state_dict(checkpoint['opt'])
        decoder_optimizer.load_state_dict(checkpoint['ddecoder_optimizere_opt'])
        voc = checkpoint['voc']
        print('\tdone')
    print('done')
    
    print('initiprint_lossesalising...')
    start_iteration = 1
    print_loss = 0
    print('done')

    print('training...')
    for iter in range(start_iteration, NUM_ITERS+1):
        for idx, data_batch in enumerate(train_set):
            loss = train_epoch(data_batch, seq2seq, param_optimizer, decoder_optimizer)
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
                'model': seq2seq.state_dict(),
                'opt': param_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc': voc
            }, os.path.join(directory, '{}_{}.tar'.format(iter, 'checkpoint')))

        if iter % EVAL_EVERY == 0:
            seq2seq.eval()

            loss = 0
            for idx, data_batch in enumerate(dev_set):
                _, print_losses, n_totals = seq2seq(data_batch)
                loss += sum(print_losses) / n_totals

            print("[EVAL]Iteration: {}; Loss: {:.4f}".format(iter, loss))
            seq2seq.train()


if __name__ == '__main__':
    train()
