import torch
import random
import torch.autograd as autograd
import torch.nn as nn
import os
import numpy as np
import pandas as pd

from utils.conf import args, set_random_seed
from models.Set2Seq2Choice import Set2Seq2Choice
from preprocesses.DataIterator import ChooseDataset, FruitSeqDataset
from preprocesses.Voc import Voc
from analysis.training_sim_check import sim_check


def get_batches4sim_check(voc, dataset_file_path=args.data_file):
    in_set = FruitSeqDataset.load_stringset(dataset_file_path)
    batch_set = ChooseDataset(voc, batch_size=1, dataset_file_path=dataset_file_path)
    return in_set, batch_set


def train_epoch(model, data_batch, s_optimizer, l_optimizer, clip=args.clip):
    s_optimizer.zero_grad()
    l_optimizer.zero_grad()

    loss, print_loss, acc, c_correct, log_msg_prob, log_choose_prob,\
         baseline, spk_entropy = model(data_batch)
    
    if args.msg_mode == 'REINFORCE':
        (c_correct.detach() * log_msg_prob + 0.05 * spk_entropy).mean().backward()
        (c_correct.detach() * log_choose_prob).mean().backward()
    elif args.msg_mode == 'SCST':
        ((c_correct.detach()-baseline.detach()) * log_msg_prob).mean().backward()
        ((c_correct.detach()-baseline.detach()) * log_choose_prob).mean().backward()
    elif args.msg_mode == 'GUMBEL':
        loss.mean().backward()
    else:
        raise NotImplementedError
    
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    s_optimizer.step()
    l_optimizer.step()

    return acc, print_loss


def eval_model(model, dataset):
    model.eval()

    loss = 0.
    avg_acc = 0.
    for _, data_batch in enumerate(dataset):
        print_loss, acc = model(data_batch)[1:3]
        loss += print_loss
        avg_acc += acc

    loss /= len(dataset)
    avg_acc /=len(dataset)

    model.train()

    return avg_acc, loss


def train():
    print('building vocabulary...')
    voc = Voc()
    print('done')

    print('loading data and building batches...')
    train_set = ChooseDataset(voc, dataset_file_path=args.train_file)
    dev_set = ChooseDataset(voc, dataset_file_path=args.dev_file)
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

        model = Set2Seq2Choice(voc.num_words).to(args.device)
        model.load_state_dict(checkpoint['model'])
        speaker_optimiser = train_args.optimiser(model.speaker.parameters(), 
                                        lr=train_args.learning_rate)
        listner_optimiser = train_args.optimiser(model.listener.parameters(), 
                                        lr=train_args.learning_rate)
        print('\tdone')
    else:
        print('building model...')
        model = Set2Seq2Choice(voc.num_words).to(args.device)
        speaker_optimiser = args.optimiser(model.speaker.parameters(), lr=args.learning_rate)
        listner_optimiser = args.optimiser(model.listener.parameters(), lr=args.learning_rate)
        print('done')

    print('preparing data for testing topological similarity...')
    sim_chk_inset, sim_chk_batchset = get_batches4sim_check(voc, args.data_file)
    print('done')
    
    print('initialising...')
    start_iteration = 1
    print_loss = 0.
    print_acc = 0.
    max_dev_acc = 0.
    training_losses = []
    training_acc = []
    training_in_spkh_sim = []
    training_in_msg_sim = []
    training_in_lish_sim = []
    eval_acc = []
    print('done')

    in_spk_sim, in_msg_sim, in_lis_sim = sim_check(
        model, sim_chk_inset, sim_chk_batchset
    )
    print('[SIM]Iteration: {}; In-SpkHidden Sim: {:.4f}; In-Msg Sim: {:.4f}; In-LisHidden Sim: {:.4f}'.format(
                0, in_spk_sim, in_msg_sim, in_lis_sim))

    print('training...')
    for iter in range(start_iteration, args.iter_num+1):
        for idx, data_batch in enumerate(train_set):
            acc, loss = train_epoch(
                model,
                data_batch,
                speaker_optimiser,
                listner_optimiser
            )
            print_loss += loss
            print_acc += acc

        if iter % args.print_freq == 0:
            print_loss_avg = print_loss / (args.print_freq * len(train_set))
            print_acc_avg = print_acc / (args.print_freq * len(train_set))
            print("Iteration: {}; Percent complete: {:.1f}%; Avg loss: {:.4f}; Avg acc: {:.4f};".format(
                iter, iter / args.iter_num * 100, print_loss_avg, print_acc_avg))
            training_acc.append(print_acc_avg)
            training_losses.append(print_loss_avg)
            print_acc = 0.
            print_loss = 0.

        if iter % args.eval_freq == 0:
            dev_acc, dev_loss = eval_model(model, dev_set)
            if dev_acc > max_dev_acc:
                max_dev_acc = dev_acc
            eval_acc.append(dev_acc)
            print("[EVAL]Iteration: {}; Loss: {:.4f}; Avg Acc: {:.4f}; Best Acc: {:.4f}".format(
                iter, dev_loss, dev_acc, max_dev_acc))

        if iter % args.sim_chk_freq == 0:
            in_spk_sim, in_msg_sim, in_lis_sim = sim_check(
                model, sim_chk_inset, sim_chk_batchset
            )
            training_in_spkh_sim.append(in_spk_sim)
            training_in_msg_sim.append(in_msg_sim)
            training_in_lish_sim.append(in_lis_sim)
            print('[SIM]Iteration: {}; In-SpkHidden Sim: {:.4f}; In-Msg Sim: {:.4f}; In-LisHidden Sim: {:.4f}'.format(
                0, in_spk_sim, in_msg_sim, in_lis_sim))
        
        if iter % args.save_freq == 0:
            path_join = 'set2seq2choice_' + str(args.num_words) + '_' + args.msg_mode
            path_join += '_hard' if not args.soft else '_soft'
            directory = os.path.join(args.save_dir, path_join)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iter,
                'model': model.state_dict(),
                'opt': [
                    speaker_optimiser.state_dict(),
                    listner_optimiser.state_dict()
                ],
                'loss': loss,
                'voc': voc,
                'args': args,
                'records': {
                    'training_loss': training_losses,
                    'training_acc': training_acc,
                    'training_in_spkh_sim': training_in_spkh_sim,
                    'training_in_msg_sim': training_in_msg_sim,
                    'training_in_lish_sim': training_in_lish_sim,
                    'eval_acc': eval_acc,
                }
            }, os.path.join(directory, '{}_{:.4f}_{}.tar'.format(iter, dev_acc, 'checkpoint')))


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
        model = Set2Seq2Choice(voc.num_words).to(args.device)
        model.load_state_dict(checkpoint['model'])
        print('done')

    print('loading test data...')
    test_set = ChooseDataset(voc, dataset_file_path=args.test_file)
    print('done')
    
    test_acc, test_loss = eval_model(model, test_set)
    print("[TEST]Loss: {:.4f}; Accuracy: {:.4f};".format(
                test_loss, test_acc * 100)
         )


if __name__ == '__main__':
    set_random_seed(1234)
    with autograd.detect_anomaly():
        print('with detect_anomaly')
        if args.test:
            test()
        else:
            train()
