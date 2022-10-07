#!/bin/bash

# CoNLL-2009
python scripts/preprocessing/remap_pb2va.py \
    --input data/preprocessed/conll2009/en/CoNLL2009_train.json \
    --output data/preprocessed/conll2009/en/CoNLL2009_train.va.json

python scripts/preprocessing/remap_pb2va.py \
    --input data/preprocessed/conll2009/en/CoNLL2009_dev.json \
    --output data/preprocessed/conll2009/en/CoNLL2009_dev.va.json

python scripts/preprocessing/remap_pb2va.py \
    --input data/preprocessed/conll2009/en/CoNLL2009_test.json \
    --output data/preprocessed/conll2009/en/CoNLL2009_test.va.json


# CoNLL-2012
python scripts/preprocessing/remap_pb2va.py \
    --input data/preprocessed/conll2012/en/CoNLL2012_train.json \
    --output data/preprocessed/conll2012/en/CoNLL2012_train.va.json

python scripts/preprocessing/remap_pb2va.py \
    --input data/preprocessed/conll2012/en/CoNLL2012_dev.json \
    --output data/preprocessed/conll2012/en/CoNLL2012_dev.va.json

python scripts/preprocessing/remap_pb2va.py \
    --input data/preprocessed/conll2012/en/CoNLL2012_test.json \
    --output data/preprocessed/conll2012/en/CoNLL2012_test.va.json

# OntoNotes
python scripts/preprocessing/remap_pb2va.py \
    --input data/preprocessed/ontonotes/ontonotes_train.json \
    --output data/preprocessed/ontonotes/ontonotes_train.va.json \
    --pb2va data/resources/verbatlas-1.1/pb2va.new.tsv \
    --propbank-3

python scripts/preprocessing/remap_pb2va.py \
    --input data/preprocessed/ontonotes/ontonotes_dev.json \
    --output data/preprocessed/ontonotes/ontonotes_dev.va.json \
    --pb2va data/resources/verbatlas-1.1/pb2va.new.tsv \
    --propbank-3

python scripts/preprocessing/remap_pb2va.py \
    --input data/preprocessed/ontonotes/ontonotes_test.json \
    --output data/preprocessed/ontonotes/ontonotes_test.va.json \
    --pb2va data/resources/verbatlas-1.1/pb2va.new.tsv \
    --propbank-3