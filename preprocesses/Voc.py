from utils.conf import args


class Voc:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}

        self.word2index[args.pad_token] = args.pad_index
        self.index2word[args.pad_index] = args.pad_token
        self.word2index[args.sos_token] = args.sos_index
        self.index2word[args.sos_index] = args.sos_token
        self.word2index[args.eos_token] = args.eos_index
        self.index2word[args.eos_index] = args.eos_token
        for i in range(1, args.num_words+1):
            self.word2index[chr(64+i)] = args.eos_index + i
            self.index2word[args.eos_index + i] = chr(64+i)
        
        self.num_words = args.num_words + 3  # include SOS, EOS and PAD