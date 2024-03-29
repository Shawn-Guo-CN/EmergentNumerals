import torch
import random
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import os

from utils.conf import args, set_random_seed
from models.Set2Seq2Seq import ListeningAgent
from models.Losses import seq_cross_entropy_loss
from preprocesses.DataIterator import PairDataset
from preprocesses.Voc import Voc
from analysis.cal_topological_similarity import cal_topological_sim
from analysis.get_input_message_pairs import reproduce_msg_output


class Set2Seq2Seq(nn.Module):
    def __init__(self, voc_size, msg_length=args.max_msg_len, msg_vocsize=args.msg_vocsize, 
                    hidden_size=args.hidden_size, dropout=args.dropout_ratio):
        super().__init__()
        self.voc_size = voc_size
        self.msg_length = msg_length
        self.msg_vocsize = msg_vocsize
        self.hidden_size = hidden_size
        self.dropout = dropout

        # For embedding inputs
        self.embedding = nn.Embedding(self.voc_size, self.hidden_size)
        self.msg_embedding = nn.Embedding(self.msg_vocsize, self.hidden_size).weight

        # Listening agent
        self.listener = ListeningAgent(
            self.msg_vocsize, self.hidden_size, self.voc_size,
            self.dropout, self.embedding.weight, self.msg_embedding
        )

    def forward(self, data_batch):
        input_var = data_batch['input']
        msg_mask = data_batch['input_mask']
        target_var = data_batch['target']
        target_mask = data_batch['target_mask']
        target_max_len = data_batch['target_max_len']

        msg = F.one_hot(input_var, num_classes=args.msg_vocsize).to(torch.float32)
        msg_mask = msg_mask.to(torch.float32).unsqueeze(1)

        listener_outputs = self.listener(msg, msg_mask, target_max_len)

        loss_max_len = min(listener_outputs.shape[0], target_var.shape[0])
        loss, print_losses, tok_correct, seq_correct, tok_acc, seq_acc\
            = seq_cross_entropy_loss(listener_outputs, target_var, target_mask, loss_max_len)
        
        return loss, 0., print_losses, tok_correct, seq_correct, tok_acc, seq_acc


def train_epoch(model, data_batch, m_optimizer, clip=args.clip):
    # Zero gradients
    m_optimizer.zero_grad()

    # Forward pass through model
    loss, baseline, print_losses, tok_correct, seq_correct, tok_acc, seq_acc = model(data_batch)
    # Perform backpropatation
    loss.mean().backward()

    # Clip gradients: gradients are modified in place
    nn.utils.clip_grad_norm_(model.parameters(), clip)

    # Adjust model weights
    m_optimizer.step()

    return seq_acc, tok_acc, sum(print_losses) / len(print_losses)


def eval_model(model, dataset):
    model.eval()

    loss = 0.
    seq_acc = 0.
    tok_acc = 0.
    for _, data_batch in enumerate(dataset):
        _, baseline, print_losses, tok_correct, seq_correct, t_acc, s_acc = model(data_batch)
        loss += sum(print_losses) / len(print_losses)
        seq_acc += s_acc
        tok_acc += t_acc

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
    train_set = PairDataset(voc, dataset_file_path=args.train_file)
    dev_set = PairDataset(voc, dataset_file_path=args.dev_file)
    # test_set = PairDataset(voc, dataset_file_path=TEST_FILE_PATH)
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
        print('\tdone')
    else:
        print('building model...')
        model = Set2Seq2Seq(voc.num_words).to(args.device)
        model_optimiser = args.optimiser(model.parameters(), lr=args.learning_rate)
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
    eval_tok_acc = []
    eval_seq_acc = []
    print('done')

    print('training...')
    for iter in range(start_iteration, args.iter_num+1):

        for idx, data_batch in enumerate(train_set):
            seq_acc, tok_acc, loss = train_epoch(model,
                data_batch,
                model_optimiser
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
            dev_seq_acc, dev_tok_acc, dev_loss = eval_model(model, dev_set)
            if dev_seq_acc > max_dev_seq_acc:
                max_dev_seq_acc = dev_seq_acc
            if dev_tok_acc > max_dev_tok_acc:
                max_dev_tok_acc = dev_tok_acc
            eval_tok_acc.append(dev_tok_acc)
            eval_seq_acc.append(dev_seq_acc)
            print("[EVAL]Iteration: {}; Loss: {:.4f}; Avg Seq Acc: {:.4f}; Avg Tok Acc: {:.4f}; Best Seq Acc: {:.4f}".format(
                iter, dev_loss, dev_seq_acc, dev_tok_acc, max_dev_seq_acc))
 
        if iter % args.save_freq == 0:
            path_join = 'listener_' + str(args.num_words) + '_' + args.msg_mode
            path_join += '_hard' if not args.soft else '_soft'
            directory = os.path.join(args.save_dir, path_join)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iter,
                'model': model.state_dict(),
                'opt': [
                    model_optimiser.state_dict()
                ],
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
            }, os.path.join(directory, '{}_{}_{}.tar'.format(args.seed, iter, 'checkpoint')))


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
        model_optimizer = train_args.optimiser(model.parameters(), lr=args.learning_rate)
        model_optimizer.load_state_dict(checkpoint['opt'])
        print('done')

    print('loading test data...')
    test_set = PairDataset(voc, dataset_file_path=args.test_file)
    print('done')
    
    test_seq_acc, test_tok_acc, test_loss = eval_model(model, test_set)
    print("[TEST]Loss: {:.4f}; Seq-level Accuracy: {:.4f}; Tok-level Accuracy: {:.4f}".format(
                test_loss, test_seq_acc * 100, test_tok_acc * 100)
         )


if __name__ == '__main__':
    set_random_seed(args.seed)
    with autograd.detect_anomaly():
        print('with detect_anomaly')
        if args.test:
            test()
        else:
            train()
