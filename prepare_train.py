"""Prepare training data and vocab
Usage:
python prepare_train.py \
    --input_file path/to/input_file \
    --output_file path/to/output/file \
    --output_vocab_file path/to/output_vocab_file \
    --max_vocab_size 50000 50000 (optional: default 50000 50000) \
    --vocab_min_count 1 1(optional: default 1 1)
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

def build_vocab(src_tok, tgt_tok, max_vocab_size, min_count):
    max_vocab_size_src, max_vocab_size_tgt = max_vocab_size
    min_count_src, min_count_tgt = min_count

    vocab = utils.Seq2SeqVocab()
    vocab.build_vocab(src_tok, is_src=True, top=max_vocab_size_src, min_count=min_count_src)
    vocab.build_vocab(tgt_tok, is_src=False, top=max_vocab_size_tgt, min_count=min_count_tgt)
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
    assert args.output_vocab_file[-5:] == '.json'
    assert args.output_file[-4:] == '.npy'

    # load data
    df = load_raw_file(args.input_file)

    # tokenize
    src_tok, tgt_tok = tokenize(df)

    # build vocab and save
    vocab = build_vocab(src_tok, tgt_tok, args.max_vocab_size, args.vocab_min_count)
    vocab.save_vocab(args.output_vocab_file)

    # numelicalize data
    dataset_numel = numelicalize(src_tok, tgt_tok, vocab)

    # save data to file
    save_data(dataset_numel, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--output_vocab_file', type=str, required=True)
    parser.add_argument('--max_vocab_size', nargs=2, type=int, default=[50000, 50000], metavar=('src', 'tgt'))
    parser.add_argument('--vocab_min_count', nargs=2, type=int, default=[1, 1], metavar=('src', 'tgt'))
    args = parser.parse_args()
    main()
    print(f'Done!\nOutput file path: {args.output_file}\nOutput vocab path: {args.output_vocab_file}')
