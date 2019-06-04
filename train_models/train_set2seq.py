from utils.conf import *
from models.Set2Seq import *
from preprocesses.DataIterator import FruitSeqDataset
from preprocesses.Voc import Voc


def train_epoch(model, data_batch, param_optimizer, decoder_optimizer, clip=CLIP):
    # Zero gradients
    param_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Forward pass through model
    loss, print_losses, n_corrects, n_totals = model(data_batch)
    # Perform backpropatation
    loss.backward()
    # Calculate accuracy
    acc = round(float(n_corrects) / float(n_totals), 6)

    # Clip gradients: gradients are modified in place
    nn.utils.clip_grad_norm_(model.parameters(), clip)

    # Adjust model weights
    param_optimizer.step()
    decoder_optimizer.step()

    return acc, sum(print_losses) / n_totals


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
    set2seq = Set2Seq(voc.num_words).to(DEVICE)
    param_optimizer = OPTIMISER(set2seq.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = OPTIMISER(set2seq.decoder.parameters(), 
                                    lr=LEARNING_RATE * DECODER_LEARING_RATIO)
    if PARAM_FILE is not None:
        print('\tloading saved parameters from ' + PARAM_FILE + '...')
        checkpoint = torch.load(PARAM_FILE)
        set2seq.load_state_dict(checkpoint['model'])
        param_optimizer.load_state_dict(checkpoint['opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])
        voc = checkpoint['voc']
        print('\tdone')
    print('done')
    
    print('initialising...')
    start_iteration = 1
    print_loss = 0.
    print_acc = 0.
    max_dev_acc = 0.
    print('done')

    print('training...')
    for iter in range(start_iteration, NUM_ITERS+1):
        for idx, data_batch in enumerate(train_set):
            acc, loss = train_epoch(set2seq,
                data_batch,
                param_optimizer,
                decoder_optimizer
            )
            print_loss += loss
            print_acc += acc

        if iter % PRINT_EVERY == 0:
            print_loss_avg = print_loss / (PRINT_EVERY * len(train_set))
            print_acc_avg = print_acc / (PRINT_EVERY * len(train_set))
            print("Iteration: {}; Percent complete: {:.1f}%; Avg loss: {:.4f}; Avg acc: {:.4f}".format(
                iter, iter / NUM_ITERS * 100, print_loss_avg, print_acc_avg
                ))
            print_loss = 0.
            print_acc = 0.

        if iter % EVAL_EVERY == 0:
            set2seq.eval()
            
            dev_loss = 0.
            dev_acc = 0.
            for idx, data_batch in enumerate(dev_set):
                _, print_losses, n_corrects, n_totals = set2seq(data_batch)
                dev_loss += sum(print_losses) / n_totals
                dev_acc += float(n_corrects) / float(n_totals)

            dev_loss /= len(dev_set)
            dev_acc /= len(dev_set)
            if dev_acc > max_dev_acc:
                max_dev_acc = dev_acc

            print("[EVAL]Iteration: {}; Loss: {:.4f}; Avg Acc: {:.4f}; Best Acc: {:.4f}".format(
                iter, dev_loss, dev_acc, max_dev_acc))

            set2seq.train()

        if iter % SAVE_EVERY == 0:
            directory = os.path.join(SAVE_DIR, 'set2seq_' + str(HIDDEN_SIZE))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iter,
                'model': set2seq.state_dict(),
                'opt': param_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc': voc
            }, os.path.join(directory, '{}_{:.4f}_{}.tar'.format(iter, dev_acc, 'checkpoint')))


if __name__ == '__main__':
    train()
