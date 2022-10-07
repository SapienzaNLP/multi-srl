#!/bin/bash

# CoNLL-2009
python scripts/preprocessing/compute_vocabulary.py \
    --input data/preprocessed/conll2009/en/CoNLL2009_train.json \
    --output data/preprocessed/conll2009/en/vocabulary.json

python scripts/preprocessing/compute_vocabulary.py \
    --input data/preprocessed/conll2009/ca/CoNLL2009_train.json \
    --output data/preprocessed/conll2009/ca/vocabulary.json

python scripts/preprocessing/compute_vocabulary.py \
    --input data/preprocessed/conll2009/cs/CoNLL2009_train.json \
    --output data/preprocessed/conll2009/cs/vocabulary.json

python scripts/preprocessing/compute_vocabulary.py \
    --input data/preprocessed/conll2009/de/CoNLL2009_train.json \
    --output data/preprocessed/conll2009/de/vocabulary.json

python scripts/preprocessing/compute_vocabulary.py \
    --input data/preprocessed/conll2009/es/CoNLL2009_train.json \
    --output data/preprocessed/conll2009/es/vocabulary.json

python scripts/preprocessing/compute_vocabulary.py \
    --input data/preprocessed/conll2009/zh/CoNLL2009_train.json \
    --output data/preprocessed/conll2009/zh/vocabulary.json

python scripts/preprocessing/compute_vocabulary.py \
    --input data/preprocessed/conll2009/en/CoNLL2009_train.va.json \
    --output data/preprocessed/conll2009/en/vocabulary.va.json


# CoNLL-2012
python scripts/preprocessing/compute_vocabulary.py \
    --input data/preprocessed/conll2012/en/CoNLL2012_train.json \
    --output data/preprocessed/conll2012/en/vocabulary.json

python scripts/preprocessing/compute_vocabulary.py \
    --input data/preprocessed/conll2012/en/CoNLL2012_train.va.json \
    --output data/preprocessed/conll2012/en/vocabulary.va.json


# OntoNotes
python scripts/preprocessing/compute_vocabulary.py \
    --input data/preprocessed/ontonotes/ontonotes_train.json \
    --output data/preprocessed/ontonotes/vocabulary.json

python scripts/preprocessing/compute_vocabulary.py \
    --input data/preprocessed/ontonotes/ontonotes_train.va.json \
    --output data/preprocessed/ontonotes/vocabulary.va.json


# UniteD-SRL
python scripts/preprocessing/compute_vocabulary.py \
    --input data/preprocessed/united/en/dependency/united_train.json \
    --output data/preprocessed/united/en/dependency/vocabulary.json

python scripts/preprocessing/compute_vocabulary.py \
    --input data/preprocessed/united/en/span/united_train.json \
    --output data/preprocessed/united/en/span/vocabulary.json
