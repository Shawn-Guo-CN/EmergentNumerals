from utils.conf import *
from models.Set2Seq2Seq import Set2Seq2Seq
from preprocesses.DataIterator import FruitSeqDataset
from preprocesses.Voc import Voc


def msg_tau_schedule(best_acc):
    if best_acc >= 0.6:
        MSG_TAU = 1.
    elif best_acc >= 0.8:
        MSG_TAU = 0.5
    elif best_acc >= 0.9:
        MSG_TAU = 0.1
    else:
        MSG_TAU = 2.


def train_epoch(model, data_batch, param_optimizer, decoder_optimizer, clip=CLIP):
    # Zero gradients
    param_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Forward pass through model
    loss, log_msg_prob, print_losses, \
        n_correct_seq, n_correct_token, n_total_token = model(data_batch)
    # Perform backpropatation
    if MSG_MODE == 'REINFORCE':
        log_msg_prob = (loss.detach() * log_msg_prob).mean()
        log_msg_prob.backward()
    loss.mean().backward()
    # Calculate accuracy
    tok_acc = round(float(n_correct_token) / float(n_total_token), 6)
    seq_acc = round(float(n_correct_seq) / float(data_batch['input'].shape[1]), 6)

    # Clip gradients: gradients are modified in place
    nn.utils.clip_grad_norm_(model.parameters(), clip)

    # Adjust model weights
    param_optimizer.step()
    decoder_optimizer.step()

    return seq_acc, tok_acc, sum(print_losses) / len(print_losses)


def eval_model(model, dataset):
    model.eval()

    loss = 0.
    seq_acc = 0.
    tok_acc = 0.
    for _, data_batch in enumerate(dataset):
        __, ___, print_losses, n_correct_seq, n_correct_token, n_total_token = model(data_batch)
        loss += sum(print_losses) / n_total_token
        seq_acc += round(float(n_correct_seq) / float(data_batch['input'].shape[1]), 6)
        tok_acc += float(n_correct_token) / float(n_total_token)

    loss /= len(dataset)
    seq_acc /= len(dataset)
    tok_acc /= len(dataset)

    model.train()

    return seq_acc, tok_acc, loss


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
    model = Set2Seq2Seq(voc.num_words).to(DEVICE)
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
    print_seq_acc = 0.
    print_tok_acc = 0.
    max_dev_seq_acc = 0.
    max_dev_tok_acc = 0.
    print('done')

    print('training...')
    for iter in range(start_iteration, NUM_ITERS+1):
        if MSG_MODE == 'GUMBEL':
            msg_tau_schedule(max_dev_tok_acc)

        for idx, data_batch in enumerate(train_set):
            seq_acc, tok_acc, loss = train_epoch(model,
                data_batch,
                param_optimizer,
                decoder_optimizer
            )
            print_loss += loss
            print_seq_acc += seq_acc
            print_tok_acc += tok_acc

        if iter % PRINT_EVERY == 0:
            print_loss_avg = print_loss / (PRINT_EVERY * len(train_set))
            print_seq_acc_avg = print_seq_acc / (PRINT_EVERY * len(train_set))
            print_tok_acc_avg = print_tok_acc / (PRINT_EVERY * len(train_set))
            print("Iteration: {}; Percent complete: {:.1f}%; Avg loss: {:.4f}; Avg seq acc: {:.4f}; Avg tok acc: {:.4f}".format(
                iter, iter / NUM_ITERS * 100, print_loss_avg, print_seq_acc_avg, print_tok_acc_avg
                ))
            print_seq_acc = 0.
            print_tok_acc = 0.
            print_loss = 0.

        if iter % EVAL_EVERY == 0:
            dev_seq_acc, dev_tok_acc, dev_loss = eval_model(model, dev_set)
            if dev_seq_acc > max_dev_seq_acc:
                max_dev_seq_acc = dev_seq_acc
            if dev_tok_acc > max_dev_tok_acc:
                max_dev_tok_acc = dev_tok_acc

            print("[EVAL]Iteration: {}; Loss: {:.4f}; Avg Seq Acc: {:.4f}; Avg Tok Acc: {:.4f}; Best Seq Acc: {:.4f}".format(
                iter, dev_loss, dev_seq_acc, dev_tok_acc, max_dev_seq_acc))
        
        if iter % SAVE_EVERY == 0:
            path_join = 'set2seq2seq_' + MSG_MODE
            path_join += '_hard' if MSG_HARD else '_soft'
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
            }, os.path.join(directory, '{}_{:.4f}_{}.tar'.format(iter, dev_seq_acc, 'checkpoint')))


def test():
    print('building model...')
    voc = Voc()
    model = Set2Seq2Seq(voc.num_words).to(DEVICE)
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
    
    test_seq_acc, test_tok_acc, test_loss = eval_model(model, test_set)
    print("[TEST]Loss: {:.4f}; Seq-level Accuracy: {:.4f}; Tok-level Accuracy: {:.4f}".format(
                test_loss, test_seq_acc * 100, test_tok_acc * 100)
         )


if __name__ == '__main__':
    random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    with autograd.detect_anomaly():
        print('with detect_anomaly')
        if TEST_MODE:
            test()
        else:
            train()
