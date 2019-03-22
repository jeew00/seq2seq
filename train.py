"""Train model
Usage:
python train.py \
    --train_file data/dataset/train.npy \
    --valid_file data/dataset/valid.npy \
    --vocab_file data/dataset/vocab.json \
    --output_dir model \
    --batch_size 64 \
    --max_sl_src 30 \
    --max_sl_tgt 30 \
    --n_epoch 10
"""


from architecture.transformer.transformer import create_transformer
from architecture.transformer import hparams
from architecture.transformer.optimizer import NoamOpt
from helper.dataset import Seq2SeqDataset
from helper import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm, tqdm_notebook
import numpy as np

import argparse
import pathlib
import json
import time


def load_data(trn_path, val_path):
    trn = np.load(trn_path)
    val = np.load(val_path)
    return trn, val

def get_vocab_size(vocab_path):
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    return len(vocab['src']['itos']), len(vocab['trg']['itos'])

def create_dataloader(trn, val, bs, device):
    trn_set = Seq2SeqDataset(
        data=trn,
        device=device,
        max_sl_src=args.max_sl_src,
        max_sl_tgt=args.max_sl_tgt
    )

    val_set = Seq2SeqDataset(
        data=val,
        device=device
    )

    trn_loader = DataLoader(trn_set, batch_size=bs)
    val_loader = DataLoader(val_set, batch_size=bs)

    return trn_loader, val_loader

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    with tqdm(total=len(iterator), leave=False, unit='b') as pbar:    
        for i, (src, trg) in enumerate(iterator):
            optimizer.optimizer.zero_grad()
            output = model(src, trg[:,:-1])
            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:,1:].contiguous().view(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), refresh=False)
            pbar.update()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            output = model(src, trg[:,:-1])
            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:,1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trn, val = load_data(args.train_file, args.valid_file)
    
    print(f'*' * 100, end='\n\n\n')
    print(f'Train on {trn.shape[1]} examples\nValidate on {val.shape[1]} examples')
    input_dim, output_dim = get_vocab_size(args.vocab_file)
    print(f'Source language vocab size: {input_dim}\nTarget language vocab size: {output_dim}')

    #data
    trn_loader, val_loader = create_dataloader(trn, val, args.batch_size, device)

    print(f'Loading model...')
    #model
    model = create_transformer(
        input_dim=input_dim,
        output_dim=output_dim,
        d_model=hparams.d_model,
        N=hparams.N,
        h=hparams.h,
        d_ff=hparams.d_ff,
        dropout=hparams.dropout,
        device=device
    )

    #optimizer
    optimizer = NoamOpt(
        model_size=hparams.d_model,
        factor=1,
        warmup=4000,
        optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    )

    # loss func
    criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore pad index = 0

    save_dir = pathlib.Path('model')
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    best_valid_loss = float('inf')

    print(f'Training on {device}...')
    for epoch in tqdm(range(args.n_epoch)):
        
        start_time = time.time()
        
        train_loss = train(model, trn_loader, optimizer, criterion, clip=1)
        valid_loss = evaluate(model, val_loader, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = utils.cal_elapsed_time(start_time, end_time)
        
        utils.print_stat(epoch, train_loss, valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), save_dir/f'{hparams.d_model}-{hparams.N}-{hparams.h}-{hparams.d_ff}.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--valid_file', type=str, required=True)
    parser.add_argument('--vocab_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--max_sl_src', type=int, required=True)
    parser.add_argument('--max_sl_tgt', type=int, required=True)
    parser.add_argument('--n_epoch', type=int, required=True)
    args = parser.parse_args()
    main()
