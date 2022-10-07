#!/bin/bash

# English
python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/EN/span/train.conllu \
    --output data/preprocessed/united/en/span/united_train.json

python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/EN/span/train.extra.conllu \
    --output data/preprocessed/united/en/span/united_train.extra.json

python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/EN/span/validation.conllu \
    --output data/preprocessed/united/en/span/united_dev.json

python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/EN/span/test.conllu \
    --output data/preprocessed/united/en/span/united_test.json

# Chinese
python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/ZH/span/train.conllu \
    --output data/preprocessed/united/zh/span/united_train.json

python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/ZH/span/train.extra.conllu \
    --output data/preprocessed/united/zh/span/united_train.extra.json

python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/ZH/span/validation.conllu \
    --output data/preprocessed/united/zh/span/united_dev.json

python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/ZH/span/test.conllu \
    --output data/preprocessed/united/zh/span/united_test.json

# Spanish
python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/ES/span/train.extra.conllu \
    --output data/preprocessed/united/es/span/united_train.extra.json

python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/ES/span/validation.conllu \
    --output data/preprocessed/united/es/span/united_dev.json

python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/ES/span/test.conllu \
    --output data/preprocessed/united/es/span/united_test.json

# French
python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/FR/span/train.extra.conllu \
    --output data/preprocessed/united/fr/span/united_train.extra.json

python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/FR/span/validation.conllu \
    --output data/preprocessed/united/fr/span/united_dev.json

python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/FR/span/test.conllu \
    --output data/preprocessed/united/fr/span/united_test.json

