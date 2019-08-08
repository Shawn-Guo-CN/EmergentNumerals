import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


class Voc(object):
    def __init__(self):
        self.mapping = {}
    
    def get_value(self, key):
        if key in self.mapping.keys():
            return self.mapping[key]
        else:
            self.mapping[key] = len(self.mapping)
            return self.mapping[key]


MSG_FILE = './data/rebuilt_language_2.txt'
PAD_VALUE = -1
VOC = Voc()


def load_msgs(file=open(MSG_FILE, 'r')):
    idx = 0
    msgs = []
    max_len = 0
    for line in file.readlines():
        idx += 1
        line = line.strip()
        msg_str, in_str = line.split('\t')
        if len(msg_str) > max_len:
            max_len = len(msg_str)
        msg = []
        for c in msg_str:
            msg.append(VOC.get_value(c))
        A = in_str.count('A')
        B = in_str.count('B')
        msg = {
            'array': msg,
            'str': msg_str,
            'in': in_str,
            'A': A,
            'B': B,
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

    hm_data.append(PAD_VALUE * np.ones(4))

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
msgs = sorted(msgs, key=lambda i: (i['B'], i['A']))
data = build_heatmap_data_from_msgs(msgs, max_len)


cmap = colors.ListedColormap(['white','green','blue','red','yellow', 'cyan', 'black', 'orange', 'deeppink'])
bounds=[-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
norm = colors.BoundaryNorm(bounds, cmap.N)
heatmap = plt.pcolor(data, cmap=cmap, norm=norm)

plt.xticks(np.arange(0, 5, step=1))
plt.yticks(np.arange(0, 36, step=6))
plt.colorbar(heatmap, ticks=[-1, 0, 1, 2, 3, 4, 5, 6, 7])
plt.grid(color='black', linewidth=5)
plt.show()
