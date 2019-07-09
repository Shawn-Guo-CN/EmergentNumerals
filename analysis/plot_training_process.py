import torch
import matplotlib.pyplot as plt
from math import inf

from utils.conf import args


def load_training_status(file_path:str) -> tuple:
    checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
    records = checkpoint['records']
    print(checkpoint['args'])
    return records


def EMA_smooth(l:list, alpha=0.1) -> list:
    if len(l) == 0:
        return l
    new_l = [l[0]]
    for i in l:
        new_l.append(alpha * i + (1-alpha) * new_l[-1])
    return new_l


def plot_training_into_1_figure(chk_points_paths:list, label_list:list) -> None:
    # seq_accs, tok_accs, losses, sims = [], [], [], []
    min_len = inf

    fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(nrows=2, ncols=3)
    for idx, file_path in enumerate(chk_points_paths):
        records = load_training_status(file_path)
        t_s_acc = records['training_seq_acc']
        t_t_acc = records['training_tok_acc']
        t_loss = records['traing_loss']
        t_sim = records['training_sim']
        e_t_acc = records['eval_tok_acc']
        e_s_acc = records['eval_seq_acc']

        t_s_acc = EMA_smooth(t_s_acc)
        t_t_acc = EMA_smooth(t_t_acc)
        t_loss  = EMA_smooth(t_loss)
        t_sim   = EMA_smooth(t_sim)
        e_t_acc = EMA_smooth(e_t_acc)
        e_s_acc = EMA_smooth(e_s_acc)

        linewidth = 0.6
        ax0.plot(e_s_acc, label=label_list[idx], linewidth=linewidth)
        ax1.plot(t_sim, label=label_list[idx], linewidth=linewidth)
        ax2.plot(t_s_acc, label=label_list[idx], linewidth=linewidth)
        ax3.plot(e_t_acc, label=label_list[idx], linewidth=linewidth)
        ax4.plot(t_loss, label=label_list[idx], linewidth=linewidth)
        ax5.plot(t_t_acc, label=label_list[idx], linewidth=linewidth)

    def _set_legend_(ax):
        leg = ax.legend()
        leg_lines = leg.get_lines()
        plt.setp(leg_lines, linewidth=2)
    
    for ax in [ax0, ax1, ax2, ax3, ax4, ax5]:
        _set_legend_(ax)
    
    ax0.title.set_text('Eval Sequence Accuracy')
    ax1.title.set_text('Topological Similarity')
    ax2.title.set_text('Training Sequence Accuracy')
    ax3.title.set_text('Eval Token Accuracy')
    ax4.title.set_text('Training Loss')
    ax5.title.set_text('Training Token Accuracy')
    plt.show()

def main():
    chk_point_path_list = [
        './params/speaker/set2seq_hidden/4/500_h256_seed1234.tar',
        './params/speaker/set2seq_hidden/4/500_h128_seed1234.tar',
        './params/speaker/set2seq_hidden/4/500_h64_seed1234.tar',
        './params/speaker/set2seq_hidden/4/500_h32_seed1234.tar',
    ]

    label_list = [
        '256',
        '128',
        '64',
        '32'
    ]

    plot_training_into_1_figure(chk_point_path_list, label_list)


if __name__ == '__main__':
    main()
