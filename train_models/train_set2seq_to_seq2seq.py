from utils.conf import *
from models.Set2Seq_to_Seq2Seq import Set2Seq_Seq2Seq
from preprocesses.DataIterator import FruitSeqDataset
from preprocesses.Voc import Voc


def msg_tau_schedule(best_dev_acc):
    if best_dev_acc >= 0.6:
        MSG_TAU = 1.
    elif best_dev_acc >= 0.8:
        MSG_TAU = 0.5
    elif best_dev_acc >= 0.9:
        MSG_TAU = 0.1
    else:
        MSG_TAU = 2.


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


def eval_model(model, dataset):
    model.eval()
            
    loss = 0.
    acc = 0.
    for idx, data_batch in enumerate(dataset):
        _, print_losses, n_corrects, n_totals = model(data_batch)
        loss += sum(print_losses) / n_totals
        acc += float(n_corrects) / float(n_totals)

    loss /= len(dataset)
    acc /= len(dataset)

    model.train()

    return acc, loss


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
    model = Set2Seq_Seq2Seq(voc.num_words).to(DEVICE)
    param_optimizer = OPTIMISER(model.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = OPTIMISER(model.speaker.decoder.parameters(), 
                                    lr=LEARNING_RATE * DECODER_LEARING_RATIO)
    if PARAM_FILE is not None:
        print('\tloading saved parameters from ' + PARAM_FILE + '...')
        checkpoint = torch.load(PARAM_FILE)
        model.load_state_dict(checkpoint['model'])
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
        msg_tau_schedule(max_dev_acc)

        for idx, data_batch in enumerate(train_set):
            acc, loss = train_epoch(
                model,
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
            dev_acc, dev_loss = eval_model(model, dev_set)
            if dev_acc > max_dev_acc:
                max_dev_acc = dev_acc

            print("[EVAL]Iteration: {}; Loss: {:.4f}; Avg Acc: {:.4f}; Best Acc: {:.4f}".format(
                iter, dev_loss, dev_acc, max_dev_acc))

        if iter % SAVE_EVERY == 0:
            path_join = 'set2seq_to_seq2seq_' + MSG_MODE
            if MSG_MODE == 'GUMBEL':
                model_str = '_hard' if MSG_HARD else '_soft'
                path_join += model_str
            directory = os.path.join(SAVE_DIR, path_join)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iter,
                'model': model.state_dict(),
                'opt': param_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc': voc
            }, os.path.join(directory, '{}_{:.4f}_{}.tar'.format(iter, dev_acc, 'checkpoint')))


def test():
    print('building model...')
    voc = Voc()
    model = Set2Seq_Seq2Seq(voc.num_words).to(DEVICE)
    param_optimizer = OPTIMISER(model.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = OPTIMISER(model.speaker.decoder.parameters(), 
                                    lr=LEARNING_RATE * DECODER_LEARING_RATIO)
    print('done')

    if PARAM_FILE is None:
        print('please specify the saved param file.')
        exit(-1)
    else:
        print('loading saved parameters from ' + PARAM_FILE + '...')
        checkpoint = torch.load(PARAM_FILE)
        model.load_state_dict(checkpoint['model'])
        param_optimizer.load_state_dict(checkpoint['opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])
        voc = checkpoint['voc']
        print('done')

    print('loading test data...')
    test_set = FruitSeqDataset(voc, dataset_file_path=TEST_FILE_PATH)
    print('done')
    
    test_acc, test_loss = eval_model(model, test_set)
    print("[TEST]Loss: {:.4f}; Accuracy: {:.4f}".format(
                test_loss, test_acc * 100))


if __name__ == '__main__':
    if TEST_MODE:
        test()
    else:
        train()
