import torch
import matplotlib.pyplot as plt
from math import inf

from utils.conf import args


def load_training_status(file_path:str) -> tuple:
    checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
    seq_acc, tok_acc, losses, sim = checkpoint['records']
    return seq_acc, tok_acc, losses, sim


def plot_training_into_1_figure(chk_points_paths:list) -> None:
    # seq_accs, tok_accs, losses, sims = [], [], [], []
    min_len = inf

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
    for file_path in chk_points_paths:
        s_acc, t_ac, loss, sim = load_training_status(file_path)
        label = file_path.split('.')[1].split('/')[-2]
        if min_len > len(sim):
            min_len = len(sim)
        ax0.plot(sim, label=label, linewidth=0.2)
        ax1.plot(t_ac, label=label, linewidth=0.2)
        ax2.plot(loss, label=label, linewidth=0.2)
        ax3.plot(s_acc, label=label, linewidth=0.2)

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
        './params/listener/comp/4_11000.tar',
        './params/listener/holistic/4_11000.tar',
    ]

    plot_training_into_1_figure(chk_point_path_list)


if __name__ == '__main__':
    main()
