import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Seq2SeqDataset(Dataset):
    def __init__(self, data, device, max_sl_src=None, max_sl_tgt=None, sos_idx=1, eos_idx=2):
        self.src = data[0]
        self.tgt = data[1]

        if not max_sl_src:
            self.max_sl_src = max(len(seq) for seq in self.src)
        else:
            self.max_sl_src = max_sl_src

        if not max_sl_tgt:
            self.max_sl_tgt = max(len(seq) for seq in self.tgt)
        else:
            self.max_sl_tgt = max_sl_tgt

        self.sos = np.array([sos_idx])
        self.eos = np.array([eos_idx])
        self.device = device
    
    def __getitem__(self, idx):
        if len(self.src[idx]) + 2 <= self.max_sl_src:
            src_tmp = np.concatenate([self.sos, self.src[idx], self.eos])
            src_tmp = np.pad(src_tmp, (0, self.max_sl_src - len(self.src[idx]) - 2), mode='constant')

        elif len(self.src[idx]) + 2 > self.max_sl_src:
            src_tmp = self.src[idx][:self.max_sl_src - 2]
            src_tmp = np.concatenate([self.sos, src_tmp, self.eos])

        if len(self.tgt[idx]) + 2 <= self.max_sl_tgt:
            tgt_tmp = np.concatenate([self.sos, self.tgt[idx], self.eos])
            tgt_tmp = np.pad(tgt_tmp, (0, self.max_sl_tgt - len(self.tgt[idx]) - 2), mode='constant')

        elif len(self.tgt[idx]) + 2 > self.max_sl_tgt:
            tgt_tmp = self.tgt[idx][:self.max_sl_tgt - 2]
            tgt_tmp = np.concatenate([self.sos, tgt_tmp, self.eos])

        src_tmp = torch.from_numpy(src_tmp).to(self.device)
        tgt_tmp = torch.from_numpy(tgt_tmp).to(self.device)

        return src_tmp, tgt_tmp

    def __len__(self):
        return len(self.src)
