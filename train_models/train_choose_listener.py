import torch
import random
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import os

from utils.conf import args, set_random_seed
from models.Losses import choice_cross_entropy_loss
from preprocesses.DataIterator import ChoosePairDataset
from preprocesses.Voc import Voc
from analysis.cal_topological_similarity import cal_topological_sim
from analysis.get_input_message_pairs import reproduce_msg_output
from train_models.train_set2seq2choice import eval_model
from models.Encoders import SetEncoder, SeqEncoder
from models.Decoders import SeqDecoder


class ListeningAgent(nn.Module):
    def __init__(
            self, voc_size, msg_vocsize, hidden_size, 
            dropout=args.dropout_ratio,
            embedding=None,
            msg_embedding=None
        ):
        super().__init__()
        self.voc_size = voc_size
        self.msg_vocsize = msg_vocsize
        self.hidden_size = hidden_size

        if embedding is None:
            self.embedding = nn.Embedding(self.voc_size, self.hidden_size)
        else:
            self.embedding = embedding

        if msg_embedding is None:
            self.msg_embedding = nn.Embedding(self.msg_vocsize, self.hidden_size).weight
        else:
            self.msg_embedding = msg_embedding
        
        self.msg_encoder = SeqEncoder(self.hidden_size, self.hidden_size)
        self.can_encoder = SetEncoder(self.hidden_size, self.hidden_size)

    def forward(self, message, msg_mask, candidates):
        batch_size = message.shape[1]

        msg_len = msg_mask.squeeze(1).sum(dim=0)
        message = message.transpose(0, 1)

        if self.msg_embedding is not None:
            message = F.relu(
                torch.bmm(message, self.msg_embedding.expand(batch_size, -1, -1))
            )

        _, msg_encoder_hidden, _ = self.msg_encoder(message, msg_len)
        msg_encoder_hidden = msg_encoder_hidden.transpose(0, 1).transpose(1, 2)

        can_encoder_hiddens = []
        for candidate in candidates:
            input_var = candidate['sequence']
            input_mask = candidate['seq_mask']
            embedded_input = self.embedding(input_var.t())
            encoder_hidden, _ = self.can_encoder(embedded_input, input_mask)
            can_encoder_hiddens.append(encoder_hidden)

        can_encoder_hiddens = torch.stack(can_encoder_hiddens).transpose(0, 1)

        choose_logits = torch.bmm(can_encoder_hiddens, msg_encoder_hidden).squeeze(2)
        return choose_logits


class Seq2Choose(nn.Module):
    def __init__(self, voc_size, msg_length=args.max_msg_len, msg_vocsize=args.msg_vocsize, 
                    hidden_size=args.hidden_size, dropout=args.dropout_ratio):
        super().__init__()
        self.voc_size = voc_size
        self.msg_length = msg_length
        self.msg_vocsize = msg_vocsize
        self.hidden_size = hidden_size
        self.dropout = dropout

        # For embedding inputs
        self.embedding = None
        self.msg_embedding = None

        # Listening agent
        self.listener = ListeningAgent(
            self.voc_size, self.msg_vocsize, self.hidden_size, 
            self.dropout, self.embedding, self.msg_embedding
        )

    def forward(self, data_batch):
        correct_data = data_batch['correct']
        candidates = data_batch['candidates']
        golden_label = data_batch['label']

        msg = correct_data['message']
        msg_mask = correct_data['msg_mask']

        msg = F.one_hot(msg, num_classes=self.msg_vocsize).to(torch.float32)
        msg_mask = msg_mask.to(torch.float32).unsqueeze(1)

        choose_logits = self.listener(msg, msg_mask, candidates)

        loss, print_loss, acc, c_correct = choice_cross_entropy_loss(choose_logits, golden_label)
        
        return loss, print_loss, acc, c_correct


def train_epoch(model, data_batch, m_optimizer, clip=args.clip):
    # Zero gradients
    m_optimizer.zero_grad()

    # Forward pass through model
    loss, print_loss, acc, _  = model(data_batch)
    # Perform backpropatation
    loss.mean().backward()

    # Clip gradients: gradients are modified in place
    nn.utils.clip_grad_norm_(model.parameters(), clip)

    # Adjust model weights
    m_optimizer.step()

    return acc, print_loss


def train():
    print('building vocabulary...')
    voc = Voc()
    print('done')

    print('loading data and building batches...')
    train_set = ChoosePairDataset(voc, dataset_file_path=args.train_file)
    dev_set = ChoosePairDataset(voc, dataset_file_path=args.dev_file)
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

        model = Seq2Choose(voc.num_words).to(args.device)
        model.load_state_dict(checkpoint['model'])
        model_optimiser = train_args.optimiser(model.parameters(), lr=train_args.learning_rate)
        print('\tdone')
    else:
        print('building model...')
        model = Seq2Choose(voc.num_words).to(args.device)
        model_optimiser = args.optimiser(model.parameters(), lr=args.learning_rate)
        print('done')
    
    print('initialising...')
    start_iteration = 1
    print_loss = 0.
    print_acc = 0.
    max_dev_acc = 0.
    training_losses = []
    training_acc = []
    eval_acc = []
    print('done')

    print('training...')
    for iter in range(start_iteration, args.iter_num+1):

        for idx, data_batch in enumerate(train_set):
            acc, loss = train_epoch(model,
                data_batch,
                model_optimiser
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
 
        if iter % args.save_freq == 0:
            path_join = 'choose_listener_' + str(args.num_words) + '_' + args.msg_mode
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
                    'training_acc': training_acc,
                    'eval_acc': eval_acc,
                }
            }, os.path.join(directory, '{}_{}.tar'.format(iter, 'checkpoint')))


if __name__ == '__main__':
    set_random_seed(1234)
    with autograd.detect_anomaly():
        print('with detect_anomaly')
        if args.test:
            test()
        else:
            train()
