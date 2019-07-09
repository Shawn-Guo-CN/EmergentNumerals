import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import pandas as pd
import os
import random

from utils.conf import args, set_random_seed
from models.Set2Seq import Set2Seq
from preprocesses.DataIterator import FruitSeqDataset
from preprocesses.Voc import Voc


def load_stringset(file_path):
    str_set = []
    for line in open(file_path, 'r'):
        line = line.strip()
        str_set.append(line)
    return str_set


def reproduce_input_hidden_pairs(model, voc, dataset_file_path):
    repro_dataset = FruitSeqDataset(voc, dataset_file_path=dataset_file_path, batch_size=1)
    repro_strset = load_stringset(dataset_file_path)

    def _instr2coordinate_(in_str, voc):
        coordinate = []
        for i in range(voc.num_words - 3):
            coordinate.append(in_str.count(chr(65+i)))
        return np.asarray(coordinate)

    pair_set = []
    for idx, data_batch in enumerate(repro_dataset):
        input_var = data_batch['input']
        input_mask = data_batch['input_mask']
        hidden, _ = model.encoder(model.embedding(input_var.t()), input_mask)
        hidden = hidden.squeeze().detach().cpu().numpy()
        in_vec = _instr2coordinate_(repro_strset[idx], voc)
        pair_set.append([in_vec, hidden])
    
    return pair_set


def input_hidden_sim_analysis(model, voc, data_file_path=args.data_file):
    in_hidden_pairs = reproduce_input_hidden_pairs(model, voc, data_file_path)

    mean_distances = []
    symbol_distances = []

    num_pairs = len(in_hidden_pairs)
    for i in range(num_pairs - 1):
        for j in range(i+1, num_pairs):
            mean_distances.append(np.linalg.norm(in_hidden_pairs[i][0] - in_hidden_pairs[j][0]))
            symbol_distances.append(np.linalg.norm(in_hidden_pairs[i][1] - in_hidden_pairs[j][1]))

    mean_distances = np.asarray(mean_distances)
    symbol_distances = np.asarray(symbol_distances)

    if symbol_distances.sum() == 0:
        symbol_distances = symbol_distances + 0.1
        symbol_distances[-1] -= 0.01

    dis_table = pd.DataFrame({'MD': mean_distances, 'SD': symbol_distances})
    corr = dis_table.corr('pearson')['SD']['MD']

    return corr


def train_epoch(model, data_batch, param_optimizer, decoder_optimizer, clip=args.clip):
    # Zero gradients
    param_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss, print_losses, _, _, tok_acc, seq_acc = model(data_batch)
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), clip)

    param_optimizer.step()
    decoder_optimizer.step()

    return seq_acc, tok_acc, sum(print_losses) / len(print_losses)


def eval_model(model, dataset):
    model.eval()

    loss = 0.
    seq_acc = 0.
    tok_acc = 0.
    for _, data_batch in enumerate(dataset):
        loss, print_losses, _, _, t_acc, s_acc = model(data_batch)
        loss += sum(print_losses) / len(print_losses)
        seq_acc += t_acc
        tok_acc += s_acc

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
    train_set = FruitSeqDataset(voc, dataset_file_path=args.train_file)
    dev_set = FruitSeqDataset(voc, dataset_file_path=args.dev_file)
    # test_set = FruitSeqDataset(voc, dataset_file_path=TEST_FILE_PATH)
    print('done')
    
    print('building model...')
    set2seq = Set2Seq(voc.num_words).to(args.device)
    param_optimizer = args.optimiser(set2seq.parameters(), lr=args.learning_rate)
    decoder_optimizer = args.optimiser(set2seq.decoder.parameters(), 
                                    lr=args.learning_rate * args.speaker_ratio)
    if args.param_file is not None:
        print('\tloading saved parameters from ' + args.param_file + '...')
        checkpoint = torch.load(args.param_file)
        set2seq.load_state_dict(checkpoint['model'])
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
    training_losses = []
    training_tok_acc = []
    training_seq_acc = []
    training_sim = []
    eval_tok_acc = []
    eval_seq_acc = []
    print('done')

    print('training...')
    for iter in range(start_iteration, args.iter_num+1):
        for idx, data_batch in enumerate(train_set):
            seq_acc, tok_acc, loss = train_epoch(set2seq,
                data_batch,
                param_optimizer,
                decoder_optimizer
            )
            print_loss += loss
            print_seq_acc += seq_acc
            print_tok_acc += tok_acc

        if iter % args.print_freq == 0:
            print_loss_avg = print_loss / (args.print_freq * len(train_set))
            print_seq_acc_avg = print_seq_acc / (args.print_freq * len(train_set))
            print_tok_acc_avg = print_tok_acc / (args.print_freq * len(train_set))
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
            dev_seq_acc, dev_tok_acc, dev_loss = eval_model(set2seq, dev_set)
            eval_tok_acc.append(dev_tok_acc)
            eval_seq_acc.append(dev_seq_acc)
            if dev_seq_acc > max_dev_seq_acc:
                max_dev_seq_acc = dev_seq_acc
            print("[EVAL]Iteration: {}; Loss: {:.4f}; Avg Seq Acc: {:.4f}; Avg Tok Acc: {:.4f}; Best Seq Acc: {:.4f}".format(
                iter, dev_loss, dev_seq_acc, dev_tok_acc, max_dev_seq_acc))

        if iter % args.sim_chk_freq == 0:
            corr = input_hidden_sim_analysis(set2seq, voc, args.data_file)
            training_sim.append(corr)
            print("[SIM]Iteration: {}; Sim: {:.4f}".format(iter, corr))

        if iter % args.save_freq == 0:
            directory = os.path.join(args.save_dir, 'set2seq_' + str(args.hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iter,
                'model': set2seq.state_dict(),
                'opt': param_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc': voc,
                'args': args,
                'records': {
                    'training_loss': training_losses,
                    'training_tok_acc': training_tok_acc,
                    'training_seq_acc': training_seq_acc,
                    'training_sim': training_sim,
                    'eval_tok_acc': eval_tok_acc,
                    'eval_seq_acc': eval_seq_acc
                }
            }, os.path.join(directory, '{}_{:.4f}_{}.tar'.format(iter, dev_seq_acc, 'checkpoint')))


def test():
    print('building model...')
    voc = Voc()
    set2seq = Set2Seq(voc.num_words).to(args.device)
    param_optimizer = args.optimiser(set2seq.parameters(), lr=args.learning_rate)
    decoder_optimizer = args.optimiser(set2seq.decoder.parameters(), 
                                    lr=args.learning_rate * args.decoder_ratio)
    print('done')

    if args.param_file is None:
        print('please specify the saved param file.')
        exit(-1)
    else:
        print('loading saved parameters from ' + args.param_file + '...')
        checkpoint = torch.load(args.param_file)
        set2seq.load_state_dict(checkpoint['model'])
        param_optimizer.load_state_dict(checkpoint['opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])
        voc = checkpoint['voc']
        print('done')

    print('loading test data...')
    test_set = FruitSeqDataset(voc, dataset_file_path=args.test_file)
    print('done')
    
    test_seq_acc, test_tok_acc, test_loss = eval_model(set2seq, test_set)
    print("[TEST]Loss: {:.4f}; Seq-level Accuracy: {:.4f}; Tok-level Accuracy: {:.4f}".format(
                test_loss, test_seq_acc * 100, test_tok_acc * 100)
         )


if __name__ == '__main__':
    set_random_seed(1234)
    if args.test:
        test()
    else:
        train()
