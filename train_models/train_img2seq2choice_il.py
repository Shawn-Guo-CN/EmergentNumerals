import torch
import random
import torch.autograd as autograd
import torch.nn as nn
import os
import numpy as np
import pandas as pd

from utils.conf import args, set_random_seed
from models.Img2Seq2Choice import Img2Seq2Choice
from preprocesses.DataIterator import ImgChooseDataset
from analysis.training_sim_check import sim_check
from models.Losses import seq_cross_entropy_loss
from train_models.train_img2seq2choice import get_batches4sim_check
from train_models.train_set2seq2seq_3phases import knowledge_generation_phase
from train_models.train_set2seq2choice import train_epoch, eval_model


def game_play_phase(
    model, train_set, dev_set, sim_chk_inset, sim_chk_batchset,
    s_optimiser, l_optimiser,
    clip=args.clip, generation_idx=0, alpha=0.1
):

    max_dev_acc = 0.
    print_loss = 0.
    print_acc = 0.
    training_losses = []
    training_acc = []
    training_in_spkh_sim = []
    training_in_msg_sim = []
    training_in_lish_sim = []
    eval_acc = []

    num_play_iter = args.num_play_iter+1 # if not generation_idx == args.num_generation else args.num_play_iter*3+1
    accumulated_acc = 0.

    for iter in range(1, num_play_iter):
        for data_batch in train_set:
            acc, loss = train_epoch(
                model,
                data_batch,
                args.tau,
                s_optimiser,
                l_optimiser
            )
            print_loss += loss
            print_acc += acc

        break_flag = False
        if iter % args.print_freq == 0:
            print_loss_avg = print_loss / (args.print_freq * len(train_set))
            print_acc_avg = print_acc / (args.print_freq * len(train_set))
            print("Generation: {}; Iteration: {}; Percent complete: {:.1f}%; Avg loss: {:.4f}; Avg acc: {:.4f};".format(
                generation_idx, iter, iter / args.num_play_iter * 100, print_loss_avg, print_acc_avg))
            training_acc.append(print_acc_avg)
            training_losses.append(print_loss_avg)
            
            accumulated_acc = accumulated_acc * (1 - alpha) + print_acc_avg * alpha
            if accumulated_acc > args.early_stop:
                break_flag = True
            
            print_acc = 0.
            print_loss = 0.
        
        if iter % args.eval_freq == 0 or break_flag:
            dev_acc, dev_loss = eval_model(model, dev_set)
            if dev_acc > max_dev_acc:
                max_dev_acc = dev_acc
            eval_acc.append(dev_acc)
            print("Generation: {}; [EVAL]Iteration: {}; Loss: {:.4f}; Avg Acc: {:.4f}; Best Acc: {:.4f}".format(
                generation_idx, iter, dev_loss, dev_acc, max_dev_acc))

        if iter % args.sim_chk_freq == 0 or break_flag:
            in_spk_sim, in_msg_sim, in_lis_sim = sim_check(
                model, sim_chk_inset, sim_chk_batchset, label_mode=True
            )
            training_in_spkh_sim.append(in_spk_sim)
            training_in_msg_sim.append(in_msg_sim)
            training_in_lish_sim.append(in_lis_sim)
            print('Generation: {}; [SIM]Iteration: {}; In-SpkHidden Sim: {:.4f}; In-Msg Sim: {:.4f}; In-LisHidden Sim: {:.4f}'.format(
                generation_idx, iter, in_spk_sim, in_msg_sim, in_lis_sim))

        if break_flag:
            break

    return training_losses, training_acc, training_in_spkh_sim, training_in_msg_sim, training_in_lish_sim, eval_acc


def listener_warming_up_phase(
    model, train_set, dev_set, s_optimiser, l_optimiser,
    clip=args.clip, generation_idx=0
):
    print_loss = 0.
    print_acc = 0.

    model.speaker.eval()

    for iter in range(1, args.num_lwarmup_iter+1):
        for data_batch in train_set:
            acc, loss = train_epoch(
                model,
                data_batch,
                s_optimiser,
                l_optimiser
            )
            print_loss += loss
            print_acc += acc

        if iter % args.print_freq == 0:
            print_loss_avg = print_loss / (args.print_freq * len(train_set))
            print_acc_avg = print_acc / (args.print_freq * len(train_set))
            print("Generation: {}; Warming Up Iteration: {}; Percent complete: {:.1f}%; Avg loss: {:.4f}; Avg acc: {:.4f};".format(
                generation_idx, iter, iter / args.num_lwarmup_iter * 100, print_loss_avg, print_acc_avg
                ))
            print_loss = 0.
            print_acc = 0.

    model.speaker.train()


def _speaker_learn_(model, data_batch, target, tgt_mask):
    input_var = data_batch['correct']['imgs']

    message, msg_logits, _ = model.speaker(input_var)

    loss_max_len = min(message.shape[0], target.shape[0])
    loss, _, _, _, tok_acc, seq_acc\
        = seq_cross_entropy_loss(msg_logits, target, tgt_mask, loss_max_len)
    
    return loss, tok_acc, seq_acc


def speaker_learning_phase(model, s_optimizer, input_set, target_set, tgt_mask_set, 
    generation_idx=0, clip=args.clip):
    assert len(input_set) == len(target_set)
    assert len(target_set) == len(tgt_mask_set)

    print_loss = 0.
    print_seq_acc = 0.
    print_tok_acc = 0.

    for iter in range(1, args.num_spklearn_iter+1):
        for idx, data_batch in enumerate(input_set):
            s_optimizer.zero_grad()
            loss, tok_acc, seq_acc = \
                _speaker_learn_(model, data_batch, target_set[idx], tgt_mask_set[idx])
            loss.mean().backward()
            nn.utils.clip_grad_norm_(model.speaker.parameters(), clip)
            s_optimizer.step()
            print_loss += loss.mean()
            print_seq_acc += seq_acc
            print_tok_acc += tok_acc
            
        if iter % args.print_freq == 0:
            print_loss_avg = print_loss / (args.print_freq * len(input_set))
            print_seq_acc_avg = print_seq_acc / (args.print_freq * len(input_set))
            print_tok_acc_avg = print_tok_acc / (args.print_freq * len(input_set))
            print("Generation: {}; Speaker Learning Phase; Iteration: {}; Percent complete: {:.1f}%; Avg loss: {:.4f}; Avg seq acc: {:.4f}; Avg tok acc: {:.4f}".format(
                generation_idx, iter, iter / args.num_spklearn_iter * 100, print_loss_avg, print_seq_acc_avg, print_tok_acc_avg
                ))
            print_loss = 0.
            print_seq_acc = 0.
            print_tok_acc = 0.


def train_generation(
    model, train_set, dev_set, learn_set, sim_chk_inset, sim_chk_batchset,
    clip=args.clip, generation_idx=0
):
    s_optimiser = args.optimiser(model.speaker.parameters(), lr=args.learning_rate)
    l_optimiser = args.optimiser(model.listener.parameters(), lr=args.learning_rate)

    training_losses, training_acc, training_in_spkh_sim, training_in_msg_sim, \
        training_in_lish_sim, eval_acc = \
            game_play_phase(model, train_set, dev_set, sim_chk_inset, sim_chk_batchset, s_optimiser, l_optimiser, clip, generation_idx)

    if not generation_idx == args.num_generation:
        random.shuffle(learn_set.databatch_set)
        reproduced_msg_set, reproduced_msg_masks = \
            knowledge_generation_phase(model, learn_set)
        print('Generation: {}; Message Reproduction Phase Done.'.format(generation_idx))

        model.reset_speaker()
        print('Generation: {}; Speaker Reset Done.'.format(generation_idx))
        model.reset_listener()
        print('Generation: {}; Listener Reset Done.'.format(generation_idx))

        s_optimiser = args.optimiser(model.speaker.parameters(), lr=args.learning_rate)
        l_optimiser = args.optimiser(model.listener.parameters(), lr=args.learning_rate)

        speaker_learning_phase(model, s_optimiser, \
            learn_set, reproduced_msg_set, reproduced_msg_masks, generation_idx, clip)
        print('Generation: {}; Speaker Learning Phase Done.'.format(generation_idx))

        listener_warming_up_phase(model, train_set, dev_set, s_optimiser, l_optimiser, clip, generation_idx)
        print('Generation: {}; Listener Warming Up Phase Done.'.format(generation_idx))

        del reproduced_msg_set
        del reproduced_msg_masks

    return training_losses, training_acc, training_in_spkh_sim, training_in_msg_sim, training_in_lish_sim, eval_acc


def train():
    print('loading data and building batches...')
    train_set = ImgChooseDataset(dataset_dir_path=args.train_file)
    dev_set = ImgChooseDataset(dataset_dir_path=args.dev_file)
    learn_set = ImgChooseDataset(dataset_dir_path=args.train_file, batch_size=1)
    print('done')
        
    if args.param_file is not None:
        print('loading saved parameters from ' + args.param_file + '...')
        checkpoint = torch.load(args.param_file, map_location=args.device)
        train_args = checkpoint['args']
        print('done')

        print('arguments for training:')
        print(train_args)

        print('rebuilding model...')

        model = Img2Seq2Choice(
            msg_length=train_args.max_msg_len, msg_vocsize=train_args.msg_vocsize, 
            hidden_size=train_args.hidden_size, dropout=train_args.dropout_ratio, msg_mode=train_args.msg_mode
        ).to(args.device)
        model.load_state_dict(checkpoint['model'])
        print('\tdone')
    else:
        print('building model...')
        model = Img2Seq2Choice().to(args.device)
        print('done')

    print('preparing data for testing topological similarity...')
    sim_chk_inset, sim_chk_batchset = get_batches4sim_check(args.data_file)
    print('done')
    
    print('initialising...')
    start_iteration = 1
    training_losses = []
    training_acc = []
    training_in_spkh_sim = []
    training_in_msg_sim = []
    training_in_lish_sim = []
    eval_acc = []
    print('done')

    in_spk_sim, in_msg_sim, in_lis_sim = sim_check(
        model, sim_chk_inset, sim_chk_batchset, label_mode=True
    )
    print('[SIM]Iteration: {}; In-SpkHidden Sim: {:.4f}; In-Msg Sim: {:.4f}; In-LisHidden Sim: {:.4f}'.format(
                0, in_spk_sim, in_msg_sim, in_lis_sim))

    print('training...')
    for iter in range(start_iteration, args.num_generation+1):
        training_records = train_generation(
            model, train_set, dev_set, learn_set, sim_chk_inset, sim_chk_batchset,
            generation_idx=iter
        )

        training_losses += training_records[0]
        training_acc += training_records[1]
        training_in_spkh_sim += training_records[2]
        training_in_msg_sim+= training_records[3]
        training_in_lish_sim += training_records[4]
        eval_acc += training_records[5]
        
        if iter % args.save_freq == 0:
            path_join = 'img2seq2choice_il_' + str(args.num_words) + '_' + args.msg_mode
            path_join += '_hard' if not args.soft else '_soft'
            directory = os.path.join(args.save_dir, path_join)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iter,
                'model': model.state_dict(),
                'args': args,
                'records': {
                    'training_loss': training_losses,
                    'training_acc': training_acc,
                    'training_in_spkh_sim': training_in_spkh_sim,
                    'training_in_msg_sim': training_in_msg_sim,
                    'training_in_lish_sim': training_in_lish_sim,
                    'eval_acc': eval_acc,
                }
            }, os.path.join(directory, '{}_{:.4f}_{}.tar'.format(iter, eval_acc[-1], 'checkpoint')))


if __name__ == '__main__':
    set_random_seed(1234)
    with autograd.detect_anomaly():
        print('with detect_anomaly')
        if args.test:
            # test()
            raise NotImplementedError
        else:
            train()
