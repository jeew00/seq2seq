"""Preprocess data with given vocab
Usage:
python preprocess.py \
    --input_file path/to/input_file \
    --output_file path/to/output/file \
    --vocab_file path/to/vocab_file
"""


import pandas as pd
import numpy as np
from pythainlp.tokenize import word_tokenize
from helper import utils
import argparse
import pathlib

def load_raw_file(path):
    path = pathlib.Path(path)
    if path.suffix == '.xlsx':
        return pd.read_excel(path)
    elif path.suffix == '.csv':
        return pd.read_csv(path)

def tokenize(df):
    src_tok = [word_tokenize(text) for text in df.iloc[:, 0]]
    tgt_tok = [word_tokenize(text) for text in df.iloc[:, 1]]
    return src_tok, tgt_tok

def load_vocab(vocab_path):
    vocab = utils.Seq2SeqVocab()
    vocab.load_vocab(vocab_path)
    return vocab

def numelicalize(src_tok, tgt_tok, vocab):
    src_numel = np.array([np.array(utils.numel(seq, vocab.src_stoi)) for seq in src_tok])
    tgt_numel = np.array([np.array(utils.numel(seq, vocab.trg_stoi)) for seq in tgt_tok])
    dataset_numel = np.array([src_numel, tgt_numel])
    return dataset_numel

def save_data(dataset_numel, out_file):
    save_path = pathlib.Path(out_file)
    np.save(save_path, dataset_numel)

def main():

    # check outputs file type
    assert args.vocab_file[-5:] == '.json'
    assert args.output_file[-4:] == '.npy'

    # load data
    df = load_raw_file(args.input_file)

    # tokenize
    src_tok, tgt_tok = tokenize(df)

    # load vocab
    vocab = load_vocab(args.vocab_file)

    # numelicalize data
    dataset_numel = numelicalize(src_tok, tgt_tok, vocab)

    # save data to file
    save_data(dataset_numel, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--vocab_file', type=str, required=True)
    args = parser.parse_args()
    main()
    print('Done!')
