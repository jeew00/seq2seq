from collections import Counter, defaultdict
import json
import pickle
from tqdm import tqdm
import pathlib


class Seq2SeqVocab():
    def __init__(self):
        self.unk_token = '<UNK>'
        self.unk_idx = 3
        self.src_itos = {
            0: '<PAD>',
            1: '<SOS>',
            2: '<EOS>',
            self.unk_idx: self.unk_token
        }
        self.src_stoi = {w: i for i, w in self.src_itos.items()}
        
        self.trg_itos = {
            0: '<PAD>',
            1: '<SOS>',
            2: '<EOS>',
            self.unk_idx: self.unk_token
        }
        self.trg_stoi = {w: i for i, w in self.trg_itos.items()}
        
        self.unk_idx = 3

    def build_vocab(self, data, is_src, top=100000, min_count=2):
        '''Build vocabulary for given data with additional special tokens
        <PAD>, <SOS>, <EOS>, <UNK>.

        Args:
            data (list of list of tokenized data): Data to build vocabulary from.
            is_src (bool): Choose between source or target language.
            top (int, optional): Maximum vocab size sorted by frequency.
            min_count (int, optional):  Minimum count for words to be included in the vocab

        Returns:
            obj: Counter object that count the frequency of tokens.
            obj: Defaultdict object that convert string to index with default values of 3 (<UNK> token).
            dict: Dictionary that convert index to string.
        '''
        if is_src:
            
            self.src_counter = Counter(ti for ei in data for ti in ei)
            self.src_counter = Counter(
                {word: count for word, count in self.src_counter.items() if count >= min_count}
            )
            self.src_counter = self.src_counter.most_common(top)
            
            self.src_itos = {**self.src_itos, **{i: w for i, (w, _) in enumerate(self.src_counter, 4)}}
            self.src_stoi = defaultdict(
                lambda: self.unk_idx,
                {**self.src_stoi, **{w: i for i, (w, _) in enumerate(self.src_counter, 4)}}
            )
        
        elif not is_src:
            
            self.trg_counter = Counter(ti for ei in data for ti in ei)
            self.trg_counter = Counter(
                {word: count for word, count in self.trg_counter.items() if count >= min_count}
            )
            self.trg_counter = self.trg_counter.most_common(top)
            
            self.trg_itos = {**self.trg_itos, **{i: w for i, (w, _) in enumerate(self.trg_counter, 4)}}
            self.trg_stoi = defaultdict(
                lambda: self.unk_idx,
                {**self.trg_stoi, **{w: i for i, (w, _) in enumerate(self.trg_counter, 4)}}
            )

    def save_vocab(self, save_path):
        
        self.src_stoi = defaultdict(
            lambda: self.unk_idx,
            {w: i for w, i in self.src_stoi.items() if not (i == self.unk_idx and w != self.unk_token)}
        )
        
        self.trg_stoi = defaultdict(
            lambda: self.unk_idx,
            {w: i for w, i in self.trg_stoi.items() if not (i == self.unk_idx and w != self.unk_token)}
        )
        
        save_path = pathlib.Path(save_path)
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(
                {
                    'src': {
                        'stoi': dict(self.src_stoi),
                        'itos': self.src_itos
                    },
                    'trg': {
                        'stoi': dict(self.trg_stoi),
                        'itos': self.trg_itos
                    },
                }, f)
    
    def load_vocab(self, vocab_path):
        with open(vocab_path, 'r') as f:
            vocab_tmp = json.load(f)
        self.src_stoi = defaultdict(lambda: self.unk_idx, vocab_tmp['src']['stoi'])
        self.src_itos = vocab_tmp['src']['itos']
        
        self.trg_stoi = defaultdict(lambda: self.unk_idx, vocab_tmp['trg']['stoi'])
        self.trg_itos = vocab_tmp['trg']['itos']
        del vocab_tmp
        
def tokenize(s):
    '''Tokenize string to list of token.

    Args:
        s (str): String to be tokenized.

    Returns:
        list: List of token of the given string.
    '''
    return list(s)

def load_data(data_path):
    '''Load data from file.
    
    Args:
        data_path (string): Path to data file.
    
    Returns:
        obj: Data object.
    '''
    with open(data_path, 'rb') as file:
        data = pickle.load(file)
    return data

def numel(seq, stoi):
    '''Map list of token to list index in the given string to index vocab
    and add End of sequence token at the end.

    Args:
        seq (list): List of token to be numelicalize.
        stoi (dict-like): String to index vocab.

    Returns:
        list: List of numelicalized token
    '''
    return [stoi[token] for token in seq]

def pad_seq(seq, max_len, pad_idx=0):
    num_pad = max_len - len(seq)
    return seq + [pad_idx] * num_pad

def remove_dupe(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]    

def cal_elapsed_time(start_time, end_time):
    '''Calculate time delta.
    
    Args:
        start_time (float): Start time.
        end_time (float): Finish time.

    Returns:
        int: Elapsed time in minute.
        int: Remain elapsed time from minute in second.
    '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_stat(epoch, train_loss, valid_loss):
    if epoch == 0:
        print(f"epoch  trn_loss  val_loss")
    if epoch < 10:
        tqdm.write(f"    {epoch}  {train_loss:.4f}    {valid_loss:.4f}")
    else:
        tqdm.write(f"   {epoch}  {train_loss:.4f}    {valid_loss:.4f}")
