from utils.conf import *

def indices_from_sequence(voc, sequence):
    return [voc.word2index[word] for word in sequence] + [EOS_token]
