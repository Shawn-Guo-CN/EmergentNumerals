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
from models.Losses import seq_cross_entropy_loss
from train_models.train_set2seq2choice import get_batches4sim_check, eval_model
from train_models.train_set2seq2seq_3phases import knowledge_generation_phase


def game_play_phase(
    model, train_set, dev_set, sim_chk_inset, sim_chk_batchset,
    m_optimiser, s_optimiser, l_optimiser,
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
                m_optimiser,
                s_optimiser,
                l_optimiser
            )
            print_loss += loss
            print_acc += acc

        accumulated_acc = accumulated_acc * (1 - alpha) + print_acc * alpha
        if accumulated_acc > args.early_stop:
            break_flag = True
        else: 
            break_flag = False
        
        if iter % args.print_freq == 0 or break_flag:
            print_loss_avg = print_loss / (args.print_freq * len(train_set))
            print_acc_avg = print_acc / (args.print_freq * len(train_set))
            print("Generation: {}; Iteration: {}; Percent complete: {:.1f}%; Avg loss: {:.4f}; Avg acc: {:.4f};".format(
                generation_idx, iter, iter / args.iter_num * 100, print_loss_avg, print_acc_avg))
            training_acc.append(print_acc_avg)
            training_losses.append(print_loss_avg)
            print_acc = 0.
            print_loss = 0.
        
        if iter % args.eval_freq == 0 or break_flag:
            dev_acc, dev_loss = eval_model(model, dev_set)
            if dev_acc > max_dev_acc:
                max_dev_acc = dev_acc
            eval_acc.append(dev_acc)
            print("Generation: {}; [EVAL]Iteration: {}; Loss: {:.4f}; Avg Acc: {:.4f}; Best Acc: {:.4f}".format(
                generation_idx, iter, dev_loss, dev_acc, max_dev_acc))

        if iter % args.sim_chk_freq == 0:
            in_spk_sim, in_msg_sim, in_lis_sim = sim_check(
                model, sim_chk_inset, sim_chk_batchset
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
    model, train_set, dev_set, m_optimiser, s_optimiser, l_optimiser,
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
                m_optimiser,
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
    input_var = data_batch['correct']['input']
    input_mask = data_batch['correct']['input_mask']

    message, msg_logits, _ = model.speaker(input_var, input_mask)

    loss_max_len = min(message.shape[0], target.shape[0])
    loss, _, _, _, tok_acc, seq_acc\
        = seq_cross_entropy_loss(msg_logits, target, tgt_mask, loss_max_len)
    
    return loss, tok_acc, seq_acc


def speaker_learning_phase(model, m_optimizer, s_optimizer, input_set, target_set, tgt_mask_set, 
    generation_idx=0, clip=args.clip):
    assert len(input_set) == len(target_set)
    assert len(target_set) == len(tgt_mask_set)

    print_loss = 0.
    print_seq_acc = 0.
    print_tok_acc = 0.

    for iter in range(1, args.num_spklearn_iter+1):
        for idx, data_batch in enumerate(input_set):
            m_optimizer.zero_grad()
            s_optimizer.zero_grad()
            loss, tok_acc, seq_acc = \
                _speaker_learn_(model, data_batch, target_set[idx], tgt_mask_set[idx])
            loss.mean().backward()
            nn.utils.clip_grad_norm_(model.speaker.parameters(), clip)
            m_optimizer.step()
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
    m_optimiser = args.optimiser(model.parameters(), lr=args.learning_rate)
    s_optimiser = args.optimiser(model.speaker.decoder.parameters(), 
                                        lr=args.learning_rate * args.speaker_ratio)
    l_optimiser = args.optimiser(model.listener.parameters(),
                                        lr=args.learning_rate * args.listener_ratio)

    training_losses, training_acc, training_in_spkh_sim, training_in_msg_sim, \
        training_in_lish_sim, eval_acc = \
            game_play_phase(model, train_set, dev_set, sim_chk_inset, sim_chk_batchset, m_optimiser, s_optimiser, l_optimiser, clip, generation_idx)

    if not generation_idx == args.num_generation:
        random.shuffle(learn_set.databatch_set)
        reproduced_msg_set, reproduced_msg_masks = \
            knowledge_generation_phase(model, learn_set)
        print('Generation: {}; Message Reproduction Phase Done.'.format(generation_idx))

        model.reset_speaker()
        print('Generation: {}; Speaker Reset Done.'.format(generation_idx))
        # model.reset_listener()
        # print('Generation: {}; Listener Reset Done.'.format(generation_idx))

        m_optimiser = args.optimiser(model.parameters(), lr=args.learning_rate)
        s_optimiser = args.optimiser(model.speaker.decoder.parameters(), 
                                            lr=args.learning_rate * args.speaker_ratio)
        l_optimiser = args.optimiser(model.listener.parameters(),
                                            lr=args.learning_rate * args.listener_ratio)

        speaker_learning_phase(model, m_optimiser, s_optimiser, \
            learn_set, reproduced_msg_set, reproduced_msg_masks, generation_idx, clip)
        print('Generation: {}; Speaker Learning Phase Done.'.format(generation_idx))

        listener_warming_up_phase(model, train_set, dev_set, m_optimiser, s_optimiser, l_optimiser, clip, generation_idx)
        print('Generation: {}; Listener Warming Up Phase Done.'.format(generation_idx))

        del reproduced_msg_set
        del reproduced_msg_masks

    return training_losses, training_acc, training_in_spkh_sim, training_in_msg_sim, training_in_lish_sim, eval_acc

def train_epoch(model, data_batch, m_optimiser, s_optimiser, l_optimiser, clip=args.clip):
    m_optimiser.zero_grad()
    s_optimiser.zero_grad()
    l_optimiser.zero_grad()

    # model.speaker.eval()
    
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
    m_optimiser.step()
    s_optimiser.step()
    l_optimiser.step()

    return acc, print_loss


def train():
    print('building vocabulary...')
    voc = Voc()
    print('done')

    print('loading data and building batches...')
    train_set = ChooseDataset(voc, dataset_file_path=args.train_file)
    dev_set = ChooseDataset(voc, dataset_file_path=args.dev_file)
    learn_set = ChooseDataset(voc, dataset_file_path=args.train_file, batch_size=1)
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
        print('\tdone')
    else:
        print('building model...')
        model = Set2Seq2Choice(voc.num_words).to(args.device)
        print('done')

    print('preparing data for testing topological similarity...')
    sim_chk_inset, sim_chk_batchset = get_batches4sim_check(voc, args.data_file)
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
        model, sim_chk_inset, sim_chk_batchset
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
            path_join = 'set2seq2choice_' + str(args.num_words) + '_' + args.msg_mode
            path_join += '_hard' if not args.soft else '_soft'
            directory = os.path.join(args.save_dir, path_join)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iter,
                'model': model.state_dict(),
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
