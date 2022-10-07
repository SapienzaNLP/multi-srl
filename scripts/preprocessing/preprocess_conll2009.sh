#!/bin/bash

# This script preprocesses the CoNLL-2009 dataset.

# English
python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/en/CoNLL2009_train.txt \
    --output data/preprocessed/conll2009/en/CoNLL2009_train.json \
    --add_predicate_pos \
    --keep_lemmas

python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/en/CoNLL2009_dev.txt \
    --output data/preprocessed/conll2009/en/CoNLL2009_dev.json \
    --add_predicate_pos \
    --keep_lemmas

python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/en/CoNLL2009_test.txt \
    --output data/preprocessed/conll2009/en/CoNLL2009_test.json \
    --add_predicate_pos \
    --keep_lemmas

python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/en/CoNLL2009_test-ood.txt \
    --output data/preprocessed/conll2009/en/CoNLL2009_test_ood.json \
    --add_predicate_pos \
    --keep_lemmas

# German
python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/de/CoNLL2009_train.txt \
    --output data/preprocessed/conll2009/de/CoNLL2009_train.json \
    --keep_lemmas

python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/de/CoNLL2009_dev.txt \
    --output data/preprocessed/conll2009/de/CoNLL2009_dev.json \
    --keep_lemmas

python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/de/CoNLL2009_test.txt \
    --output data/preprocessed/conll2009/de/CoNLL2009_test.json \
    --keep_lemmas

python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/de/CoNLL2009_test-ood.txt \
    --output data/preprocessed/conll2009/de/CoNLL2009_test_ood.json \
    --keep_lemmas

# Czech
python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/cs/CoNLL2009_train.txt \
    --output data/preprocessed/conll2009/cs/CoNLL2009_train.json \
    --keep_lemmas \
    --czech

python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/cs/CoNLL2009_dev.txt \
    --output data/preprocessed/conll2009/cs/CoNLL2009_dev.json \
    --keep_lemmas \
    --czech

python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/cs/CoNLL2009_test.txt \
    --output data/preprocessed/conll2009/cs/CoNLL2009_test.json \
    --keep_lemmas \
    --czech

python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/cs/CoNLL2009_test-ood.txt \
    --output data/preprocessed/conll2009/cs/CoNLL2009_test_ood.json \
    --keep_lemmas \
    --czech

# Spanish
python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/es/CoNLL2009_train.txt \
    --output data/preprocessed/conll2009/es/CoNLL2009_train.json \
    --keep_lemmas

python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/es/CoNLL2009_dev.txt \
    --output data/preprocessed/conll2009/es/CoNLL2009_dev.json \
    --keep_lemmas

python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/es/CoNLL2009_test.txt \
    --output data/preprocessed/conll2009/es/CoNLL2009_test.json \
    --keep_lemmas

# Catalan
python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/ca/CoNLL2009_train.txt \
    --output data/preprocessed/conll2009/ca/CoNLL2009_train.json \
    --keep_lemmas

python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/ca/CoNLL2009_dev.txt \
    --output data/preprocessed/conll2009/ca/CoNLL2009_dev.json \
    --keep_lemmas

python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/ca/CoNLL2009_test.txt \
    --output data/preprocessed/conll2009/ca/CoNLL2009_test.json \
    --keep_lemmas

# Chinese
python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/zh/CoNLL2009_train.txt \
    --output data/preprocessed/conll2009/zh/CoNLL2009_train.json \
    --keep_lemmas

python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/zh/CoNLL2009_dev.txt \
    --output data/preprocessed/conll2009/zh/CoNLL2009_dev.json \
    --keep_lemmas

python3 scripts/preprocessing/preprocess_conll2009.py \
    --input data/original/conll2009/zh/CoNLL2009_test.txt \
    --output data/preprocessed/conll2009/zh/CoNLL2009_test.json \
    --keep_lemmas