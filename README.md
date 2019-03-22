# **Seq2Seq**
## **Installation**
Support `Python 3.6` or newer.

```
git clone https://github.com/jeewling/seq2seq.git && cd seq2seq
pip install -r requirements.txt
```

## **Getting started**
1. Split data into training set and validation set
2. Preprocess training set and build vocabulary
3. Preprocess validation set with vocabulary created in step 1
4. Start training
5. Generate some text

## 1. Split data
Only support excel file (.xlsx) and csv file (.csv) for now

Please split your data into something like `train.csv` or `train.xlsx` and `val.csv` or `val.xlsx` 

example of `train.csv`/`val.xlsx`

<source_lang_column> | <target_lang_column>
--- | ---
source text 01 | target text 01
source text 02 | target text 02
source text 03 | target text 03


## 2. Preprocess training set and build vocabulary
run
```
python prepare_train.py \
    --input_file path/to/input_file.xlsx or .csv \
    --output_file path/to/output/file.npy \
    --output_vocab_file path/to/output_vocab_file.json \
    --max_vocab_size 50000 50000 \
    --vocab_min_count 1 1
```

Arguments | Description
--- | ---
--input_file | Path to training set **.xlsx** or **.csv**
--output_file | Path to output file **.npy**
--output_vocab_file | Path to vocab file **.json** (recommended name vocab.json)
--max_vocab_size (optional) | Maximum unique token in source language vocab and target language vocab respectively. (default: 50000 50000)
--vocab_min_count (optional) | Minimum frequency of token allow in source language vocab and target language vocab respectively. (default: 1 1)

Preprocessed training data and vocabulary will be created at the given path.

## 3. Preprocess validation set
run
```
python preprocess.py \
    --input_file path/to/input_file.xlsx or .csv \
    --output_file path/to/output/file.npy \
    --vocab_file path/to/vocab_file.json
```

Arguments | Description
--- | ---
--input_file | Path to validation set **.xlsx** or **.csv**
--output_file | Path to output file **.npy**
--vocab_file | Path to vocab file **vocab.json** which created in step 1.

Preprocessed validation data will be created at the given path.

## 4. Start training
run
```
python train.py \
    --train_file path/to/training_data.npy \
    --valid_file path/to/validation_data.npy \
    --vocab_file path/to/vocab.json \
    --output_dir path/to/model/output/dir \
    --batch_size 64 \
    --max_sl_src 30 \
    --max_sl_tgt 30 \
    --n_epoch 10
```

Arguments | Description
--- | ---
--train_file | Path to training data **.npy** which created in step 1.
--valid_file | Path to validation data **.npy** which created in step 2.
--vocab_file | Path to vocab file **vocab.json** which created in step 1.
--output_dir | Model will be saved in the given directory.
--batch_size | Batch size.
--max_sl_src | Max sequence length on source language of training data.
--max_sl_tgt | Max sequence length on target language of training data.
--n_epoch | Number of epoch to train for.

## 5. Generate some text
See `inference-demo.ipynb` for demo

# **Note**
- Default tokenizer from pythainlp (Can be modify in `prepare_train.py`  and `preprocess.py` tokenize function)
- Default model parameter is the same as the orginal paper. You can adjust in `architecture/transformer/hparams.py`
- Saved model is the model on the epoch that has the lowest validation loss.
