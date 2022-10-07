#!/bin/bash

python3 scripts/preprocessing/preprocess_conll2012.py \
    --input data/original/conll2012/en/CoNLL2012_train.txt \
    --output data/preprocessed/conll2012/en/CoNLL2012_train.json \
    --add_predicate_pos \
    --keep_lemmas

python3 scripts/preprocessing/preprocess_conll2012.py \
    --input data/original/conll2012/en/CoNLL2012_dev.txt \
    --output data/preprocessed/conll2012/en/CoNLL2012_dev.json \
    --add_predicate_pos \
    --keep_lemmas

python3 scripts/preprocessing/preprocess_conll2012.py \
    --input data/original/conll2012/en/CoNLL2012_test.txt \
    --output data/preprocessed/conll2012/en/CoNLL2012_test.json \
    --add_predicate_pos \
    --keep_lemmas