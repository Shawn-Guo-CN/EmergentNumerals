import torch
import matplotlib.pyplot as plt
from math import inf

from utils.conf import args


def load_training_status(file_path:str) -> tuple:
    checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
    seq_acc, tok_acc, losses, sim = checkpoint['records']
    print(checkpoint['args'])
    return seq_acc, tok_acc, losses, sim


def EMA_smooth(l:list, alpha=0.005) -> list:
    if len(l) == 0:
        return l
    new_l = [l[0]]
    for i in l:
        new_l.append(alpha * i + (1-alpha) * new_l[-1])
    return new_l


def plot_training_into_1_figure(chk_points_paths:list, label_list:list) -> None:
    # seq_accs, tok_accs, losses, sims = [], [], [], []
    min_len = inf

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
    for idx, file_path in enumerate(chk_points_paths):
        s_acc, t_ac, loss, sim = load_training_status(file_path)
        
        sim = EMA_smooth(sim)
        s_acc = EMA_smooth(s_acc)
        t_ac = EMA_smooth(t_ac)
        loss = EMA_smooth(loss)

        linewidth = 0.6
        ax0.plot(sim, label=label_list[idx], linewidth=linewidth)
        ax1.plot(t_ac, label=label_list[idx], linewidth=linewidth)
        ax2.plot(loss, label=label_list[idx], linewidth=linewidth)
        ax3.plot(s_acc, label=label_list[idx], linewidth=linewidth)

    def _set_legend_(ax):
        leg = ax.legend()
        leg_lines = leg.get_lines()
        plt.setp(leg_lines, linewidth=2)
    
    for ax in [ax0, ax1, ax2, ax3]:
        _set_legend_(ax)
    
    ax0.title.set_text('Topological Similarity')
    ax1.title.set_text('Token Accuracy')
    ax2.title.set_text('Loss')
    ax3.title.set_text('Sequence Accuracy')
    plt.show()

def main():
    chk_point_path_list = [
        './params/listener/comp/4_60000_hidden128.tar',
        './params/listener/holistic/4_60000_hidden128.tar',
        './params/listener/comp/0705_12000_4.tar',
        './params/listener/holistic/0705_12000_4.tar'
    ]

    label_list = [
        'comp-hidden128',
        'holistic-hidden128',
        'comp-hidden256',
        'holistic-hidden256'
    ]

    plot_training_into_1_figure(chk_point_path_list, label_list)


if __name__ == '__main__':
    main()
