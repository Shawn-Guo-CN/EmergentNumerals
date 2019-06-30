import torch
import random
import torch.autograd as autograd
import torch.nn as nn
import os

from utils.conf import args
from models.Set2Seq2Seq import Set2Seq2Seq
from preprocesses.DataIterator import FruitSeqDataset
from preprocesses.Voc import Voc
from analysis.cal_topological_similarity import cal_topological_sim
from analysis.get_input_message_pairs import reproduce_msg_output


def msg_tau_schedule(best_acc):
    if best_acc >= 0.95:
        args.tau = 0.05
    elif best_acc >= 0.9:
        args.tau = 0.1
    elif best_acc >= 0.8:
        args.tau = 0.5
    elif best_acc >= 0.6:
        args.tau = 1.
    else:
        args.tau = 2.


def lr_schedule(best_acc, model, m_optim, s_optim, l_optim):
    if best_acc <= 0.8:
        return m_optim, s_optim, l_optim

    if best_acc >= 0.90:
        lr = 1e-6
    elif best_acc >= 0.80:
        lr = 1e-5
    
    args.learning_rate = lr
    m_optim = args.optimiser(model.parameters(), lr=args.learning_rate)
    s_optim = args.optimiser(model.speaker.parameters(), 
                                        lr=args.learning_rate * args.speaker_ratio)
    l_optim = args.optimiser(model.listener.parameters(),
                                        lr=args.learning_rate * args.speaker_ratio)
    return m_optim, s_optim, l_optim


def train_epoch(model, data_batch, m_optimizer, s_optimizer, l_optimizer, clip=args.clip):
    # Zero gradients
    m_optimizer.zero_grad()
    s_optimizer.zero_grad()
    l_optimizer.zero_grad()

    # Forward pass through model
    loss, log_msg_prob, baseline, print_losses, \
        n_correct_seq, n_correct_token, n_total_token, _ = model(data_batch)
    # Perform backpropatation
    if args.msg_mode == 'REINFORCE':
        log_msg_prob = (loss.detach() * log_msg_prob).mean()
        log_msg_prob.backward()
    elif args.msg_mode == 'SCST':
        log_msg_prob = ((loss.detach() - baseline.detach()) * log_msg_prob).mean()
        log_msg_prob.backward()
    loss.mean().backward()
    # Calculate accuracy
    tok_acc = round(float(n_correct_token) / float(n_total_token), 6)
    seq_acc = round(float(n_correct_seq) / float(data_batch['input'].shape[1]), 6)

    # Clip gradients: gradients are modified in place
    nn.utils.clip_grad_norm_(model.parameters(), clip)

    # Adjust model weights
    m_optimizer.step()
    s_optimizer.step()
    l_optimizer.step()

    return seq_acc, tok_acc, sum(print_losses) / len(print_losses)


def eval_model(model, dataset):
    model.eval()

    loss = 0.
    seq_acc = 0.
    tok_acc = 0.
    for _, data_batch in enumerate(dataset):
        print_losses, n_correct_seq, n_correct_token, n_total_token = model(data_batch)[-5:-1]
        loss += sum(print_losses) / len(print_losses)
        seq_acc += round(float(n_correct_seq) / float(data_batch['input'].shape[1]), 6)
        tok_acc += float(n_correct_token) / float(n_total_token)

    loss /= len(dataset)
    seq_acc /= len(dataset)
    tok_acc /= len(dataset)

    model.train()

    return seq_acc, tok_acc, loss


def sim_check_model(model, voc, in_set, batch_set):
    model.eval()
    tmp_file_path = './tmp/' + str(random.random()) + '.txt'

    tmp_file = open(tmp_file_path, 'a')
    for idx, data_batch in enumerate(batch_set):
        message, output = reproduce_msg_output(model, voc, data_batch, args)
        print(in_set[idx] + '\t' + message + '\t' + output, file=tmp_file)
    tmp_file.close()

    corr = cal_topological_sim(
                msg_file_path=tmp_file_path, 
                in_dis_measure='hamming',
                msg_dis_measure='edit',
                corr_method='pearson'
            )

    os.remove(tmp_file_path)
    model.train()

    return corr


def prepare_data4sim_check(voc, dataset_file_path=args.dev_file):
    in_set = random.choices(FruitSeqDataset.load_stringset(dataset_file_path), k=args.sim_chk_k)
    
    tmp_file_path = './tmp/' + str(random.random()) + '.txt'
    tmp_file = open(tmp_file_path, 'a')
    for in_str in in_set:
        print(in_str, file=tmp_file)
    tmp_file.close()

    batch_set = FruitSeqDataset(voc, batch_size=1, dataset_file_path=tmp_file_path).batches

    os.remove(tmp_file_path)

    return in_set, batch_set



def train():
    print('building vocabulary...')
    voc = Voc()
    print('done')

    print('loading data and building batches...')
    train_set = FruitSeqDataset(voc, dataset_file_path=args.train_file)
    dev_set = FruitSeqDataset(voc, dataset_file_path=args.dev_file)
    # test_set = FruitSeqDataset(voc, dataset_file_path=TEST_FILE_PATH)
    print('done')
        
    if args.param_file is not None:
        print('loading saved parameters from ' + args.param_file + '...')
        checkpoint = torch.load(args.param_file, map_location=args.device)
        train_args = checkpoint['args']
        voc = checkpoint['voc']
        print('done')

        print('arguments for training:')
        print(train_args)

        print('rebuilding model...')

        model = Set2Seq2Seq(voc.num_words).to(args.device)
        model.load_state_dict(checkpoint['model'])
        model_optimiser = train_args.optimiser(model.parameters(), lr=train_args.learning_rate)
        speaker_optimiser = train_args.optimiser(model.speaker.parameters(), 
                                        lr=train_args.learning_rate * train_args.speaker_ratio)
        listner_optimiser = train_args.optimiser(model.listener.parameters(), 
                                        lr=train_args.learning_rate * train_args.speaker_ratio)
        print('\tdone')
    else:
        print('building model...')
        model = Set2Seq2Seq(voc.num_words).to(args.device)
        model_optimiser = args.optimiser(model.parameters(), lr=args.learning_rate)
        speaker_optimiser = args.optimiser(model.speaker.parameters(), 
                                        lr=args.learning_rate * args.speaker_ratio)
        listner_optimiser = args.optimiser(model.listener.parameters(),
                                        lr=args.learning_rate * args.speaker_ratio)
        print('done')

    print('preparing data for testing topological similarity...')
    sim_chk_inset, sim_chk_batchset = prepare_data4sim_check(voc, args.dev_file)
    print('done')
    
    print('initialising...')
    start_iteration = 1
    print_loss = 0.
    print_seq_acc = 0.
    print_tok_acc = 0.
    max_dev_seq_acc = 0.
    max_dev_tok_acc = 0.
    training_losses = []
    training_tok_acc = []
    training_seq_acc = []
    training_sim = []
    print('done')

    sim = sim_check_model(model, voc, sim_chk_inset, sim_chk_batchset)
    print('[SIM]Iteration: {}; Sim: {:.4f}'.format(0, sim))

    print('training...')
    for iter in range(start_iteration, args.iter_num+1):
        if args.msg_mode == 'GUMBEL':
            msg_tau_schedule(max_dev_tok_acc)

        for idx, data_batch in enumerate(train_set):
            seq_acc, tok_acc, loss = train_epoch(model,
                data_batch,
                model_optimiser,
                speaker_optimiser,
                listner_optimiser
            )
            print_loss += loss
            print_seq_acc += seq_acc
            print_tok_acc += tok_acc

        if iter % args.print_freq == 0:
            print_loss_avg = print_loss / (args.print_freq * len(train_set))
            print_seq_acc_avg = print_seq_acc / (args.print_freq * len(train_set))
            print_tok_acc_avg = print_tok_acc / (args.print_freq * len(train_set))

            model_optimiser, speaker_optimiser, listner_optimiser = lr_schedule(
                print_tok_acc_avg,
                model,
                model_optimiser, 
                speaker_optimiser, 
                listner_optimiser
            )

            print("Iteration: {}; Percent complete: {:.1f}%; Avg loss: {:.4f}; Avg seq acc: {:.4f}; Avg tok acc: {:.4f}".format(
                iter, iter / args.iter_num * 100, print_loss_avg, print_seq_acc_avg, print_tok_acc_avg
                ))
            training_seq_acc.append(print_seq_acc_avg)
            training_tok_acc.append(print_tok_acc_avg)
            training_losses.append(print_loss_avg)
            print_seq_acc = 0.
            print_tok_acc = 0.
            print_loss = 0.

        if iter % args.eval_freq == 0:
            dev_seq_acc, dev_tok_acc, dev_loss = eval_model(model, dev_set)
            if dev_seq_acc > max_dev_seq_acc:
                max_dev_seq_acc = dev_seq_acc
            if dev_tok_acc > max_dev_tok_acc:
                max_dev_tok_acc = dev_tok_acc

            print("[EVAL]Iteration: {}; Loss: {:.4f}; Avg Seq Acc: {:.4f}; Avg Tok Acc: {:.4f}; Best Seq Acc: {:.4f}".format(
                iter, dev_loss, dev_seq_acc, dev_tok_acc, max_dev_seq_acc))

        if iter % args.sim_chk_freq == 0:
            sim = sim_check_model(model, voc, sim_chk_inset, sim_chk_batchset)
            training_sim.append(sim)
            print('[SIM]Iteration: {}; Sim: {:.4f}'.format(0, sim))

        if iter % args.l_reset_freq == 0 and not args.l_reset_freq == -1:
            model.listener.reset_params()
            print('[RESET] reset listener')
        
        if iter % args.save_freq == 0:
            path_join = 'set2seq2seq_' + str(args.num_words) + '_' + args.msg_mode
            path_join += '_hard' if not args.soft else '_soft'
            directory = os.path.join(args.save_dir, path_join)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iter,
                'model': model.state_dict(),
                'opt': [
                    model_optimiser.state_dict(),
                    speaker_optimiser.state_dict(),
                    listner_optimiser.state_dict()
                ],
                'loss': loss,
                'voc': voc,
                'args': args,
                'records': [training_seq_acc, training_tok_acc, training_losses, training_sim]
            }, os.path.join(directory, '{}_{:.4f}_{}.tar'.format(iter, dev_seq_acc, 'checkpoint')))


def test():
    if args.param_file is None:
        print('please specify the saved param file.')
        exit(-1)
    else:
        print('loading saved parameters from ' + args.param_file + '...')
        checkpoint = torch.load(args.param_file, map_location=args.device)
        train_args = checkpoint['args']
        voc = checkpoint['voc']
        print('done')

        print('arguments for train:')
        print(train_args)
        
        print('rebuilding model...')
        model = Set2Seq2Seq(voc.num_words).to(args.device)
        model.load_state_dict(checkpoint['model'])
        param_optimizer = train_args.optimiser(model.parameters(), lr=args.learning_rate)
        decoder_optimizer = train_args.optimiser(model.speaker.decoder.parameters(), 
                                        lr=args.learning_rate * args.decoder_ratio)
        param_optimizer.load_state_dict(checkpoint['opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])
        print('done')

    print('loading test data...')
    test_set = FruitSeqDataset(voc, dataset_file_path=args.test_file)
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
        if args.test:
            test()
        else:
            train()
