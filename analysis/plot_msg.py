import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


MSG_FILE = './data/input_msg_pairs.txt'
PAD_VALUE = -1


def load_msgs(file=open(MSG_FILE, 'r')):
    idx = 0
    msgs = []
    max_len = 0
    for line in file.readlines():
        idx += 1
        line = line.strip()
        in_str, msg_str, out_str = line.split('\t')
        msg_str = msg_str[:-1]
        if len(msg_str) > max_len:
            max_len = len(msg_str)
        msg = []
        for c in msg_str:
            msg.append(int(c))
        A = in_str.count('A')
        B = in_str.count('B')
        C = in_str.count('C')
        D = in_str.count('D')
        msg = {
            'array': msg,
            'str': msg_str,
            'in': in_str,
            'A': A,
            'B': B,
            'C': C,
            'D': D
        }
        msgs.append(msg)

    return msgs, max_len

def build_x_y_c_from_msgs(msgs, max_len):
    x = []
    y = []
    c = []

    for mid, msg in enumerate(msgs):
        for cid in range(len(msg)):
            x.append(mid)
            y.append(cid)
            c.append(msg[cid])
        for cid in range(len(msg), max_len):
            x.append(mid)
            y.append(cid)
            c.append(PAD_VALUE)
    
    return x, y, c

def build_heatmap_data_from_msgs(msgs, max_len):
    hm_data = []

    for mid, msg in enumerate(msgs):
        line_data = []
        for cid in range(len(msg['array'])):
            line_data.append(msg['array'][cid])
        for cid in range(len(msg['array']), max_len):
            line_data.append(PAD_VALUE)
        hm_data.append(line_data)
    
    return np.asarray(hm_data)

msgs, max_len = load_msgs()
# msgs = sorted(msgs, key=lambda i: (i['A'], i['D'], i['C'], i['B']))
# msgs = sorted(msgs, key=lambda i: (i['B'], i['D'], i['C'], i['A']))
# msgs = sorted(msgs, key=lambda i: (i['C'], i['B'], i['A'], i['D']))
msgs = sorted(msgs, key=lambda i: (i['D'], i['B'], i['A'], i['C']))
data = build_heatmap_data_from_msgs(msgs, max_len)


cmap = colors.ListedColormap(['white','green','blue','red','yellow'])
bounds=[-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
norm = colors.BoundaryNorm(bounds, cmap.N)
heatmap = plt.pcolor(data, cmap=cmap, norm=norm)
plt.colorbar(heatmap, ticks=[-1, 0, 1, 2, 3])

plt.show()
