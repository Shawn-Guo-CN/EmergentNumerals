import torch
import random
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import os

from utils.conf import args
from models.Set2Seq2Seq import SpeakingAgent
from models.Set2Seq2Seq import mask_NLL_loss
from preprocesses.DataIterator import PairDataset
from preprocesses.Voc import Voc


class Set2Seq(nn.Module):
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
        self.msg_embedding = nn.Parameter(
                torch.randn(self.msg_vocsize, self.hidden_size, device=args.device)
            )

        # Speaking agent
        self.speaker = SpeakingAgent(
                self.embedding, self.voc_size, 
                hidden_size=args.hidden_size, 
                dropout=args.dropout_ratio,
                msg_embedding=self.msg_embedding
            )

    def forward(self, data_batch):
        input_var = data_batch['input']
        input_mask = data_batch['input_mask']
        target_var = data_batch['target']
        target_mask = data_batch['target_mask']
        target_max_len = data_batch['target_max_len']

        batch_size = input_var.shape[1]

        speaker_input = self.embedding(input_var.t())
        message, msg_mask, msg_digits, log_msg_prob = self.speaker(speaker_input, input_mask)

        loss = 0
        print_losses = []
        n_correct_tokens = 0
        n_total_tokens = 0
        n_correct_seqs = 0

        seq_correct = torch.ones((1, batch_size), device=args.device)
        eq_vec = torch.ones((1, batch_size), device=args.device)
        for t in range(target_max_len):
            mask_loss, eq_vec, n_correct, n_total = mask_NLL_loss(
                msg_digits[t],
                target_var[t],
                target_mask[t],
                eq_vec
            )
            loss += mask_loss
            print_losses.append(mask_loss.mean().item())
            n_total_tokens += n_total
            n_correct_tokens += n_correct
            seq_correct = seq_correct * eq_vec

        n_correct_seqs = seq_correct.sum().item()

        return loss, print_losses, \
            n_correct_seqs, n_correct_tokens, n_total_tokens, message, log_msg_prob


def train_epoch(model, data_batch, m_optimizer, clip=args.clip):
    # Zero gradients
    m_optimizer.zero_grad()

    # Forward pass through model
    loss, print_losses, \
        n_correct_seq, n_correct_token, n_total_token,\
             _, log_msg_prob = model(data_batch)
    
    # Perform backpropatation
    if args.msg_mode == 'REINFORCE':
        log_msg_prob = (loss.detach() * log_msg_prob).mean()
        log_msg_prob.backward()
    # elif args.msg_mode == 'SCST':
    #     log_msg_prob = ((loss.detach() - baseline.detach()) * log_msg_prob).mean()
    #     log_msg_prob.backward()
    else:
        loss.mean().backward()

    # Calculate accuracy
    tok_acc = round(float(n_correct_token) / float(n_total_token), 6)
    seq_acc = round(float(n_correct_seq) / float(data_batch['input'].shape[1]), 6)

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
        print_losses, n_correct_seq, n_correct_token, n_total_token = model(data_batch)[1:-2]
        loss += sum(print_losses) / len(print_losses)
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
    train_set = PairDataset(voc, dataset_file_path=args.train_file, reverse=True)
    dev_set = PairDataset(voc, dataset_file_path=args.dev_file, reverse=True)
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

        model = Set2Seq(voc.num_words).to(args.device)
        model.load_state_dict(checkpoint['model'])
        model_optimiser = train_args.optimiser(model.parameters(), lr=train_args.learning_rate)
        print('\tdone')
    else:
        print('building model...')
        model = Set2Seq(voc.num_words).to(args.device)
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

            print("[EVAL]Iteration: {}; Loss: {:.4f}; Avg Seq Acc: {:.4f}; Avg Tok Acc: {:.4f}; Best Seq Acc: {:.4f}".format(
                iter, dev_loss, dev_seq_acc, dev_tok_acc, max_dev_seq_acc))

        
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
                    model_optimiser.state_dict()
                ],
                'loss': loss,
                'voc': voc,
                'args': args,
                'records': [training_seq_acc, training_tok_acc, training_losses, training_sim]
            }, os.path.join(directory, '{}_{}.tar'.format(iter, 'checkpoint')))


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
        model = Set2Seq(voc.num_words).to(args.device)
        model.load_state_dict(checkpoint['model'])
        model_optimizer = train_args.optimiser(model.parameters(), lr=args.learning_rate)
        model_optimizer.load_state_dict(checkpoint['opt'])
        print('done')

    print('loading test data...')
    test_set = PairDataset(voc, dataset_file_path=args.test_file, reverse=True)
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
