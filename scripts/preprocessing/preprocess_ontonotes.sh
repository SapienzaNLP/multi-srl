#!/bin/bash

python3 scripts/preprocessing/preprocess_ontonotes.py \
    --input data/original/ontonotes/ontonotes_train.txt \
    --output data/preprocessed/ontonotes/ontonotes_train.json \
    --keep_lemmas

python3 scripts/preprocessing/preprocess_ontonotes.py \
    --input data/original/ontonotes/ontonotes_dev.txt \
    --output data/preprocessed/ontonotes/ontonotes_dev.json \
    --keep_lemmas

python3 scripts/preprocessing/preprocess_ontonotes.py \
    --input data/original/ontonotes/ontonotes_test.txt \
    --output data/preprocessed/ontonotes/ontonotes_test.json \
    --keep_lemmas

python3 scripts/preprocessing/preprocess_ontonotes.py \
    --input data/original/ontonotes/ontonotes_test-conll2012.txt \
    --output data/preprocessed/ontonotes/ontonotes_test-conll2012.json \
    --keep_lemmas