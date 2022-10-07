#!/bin/bash

# English
python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/EN/dependency/train.conllu \
    --output data/preprocessed/united/en/dependency/united_train.json

python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/EN/dependency/train.extra.conllu \
    --output data/preprocessed/united/en/dependency/united_train.extra.json

python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/EN/dependency/validation.conllu \
    --output data/preprocessed/united/en/dependency/united_dev.json

python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/EN/dependency/test.conllu \
    --output data/preprocessed/united/en/dependency/united_test.json

# Chinese
python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/ZH/dependency/train.conllu \
    --output data/preprocessed/united/zh/dependency/united_train.json

python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/ZH/dependency/train.extra.conllu \
    --output data/preprocessed/united/zh/dependency/united_train.extra.json

python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/ZH/dependency/validation.conllu \
    --output data/preprocessed/united/zh/dependency/united_dev.json

python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/ZH/dependency/test.conllu \
    --output data/preprocessed/united/zh/dependency/united_test.json

# Spanish
python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/ES/dependency/train.extra.conllu \
    --output data/preprocessed/united/es/dependency/united_train.extra.json

python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/ES/dependency/validation.conllu \
    --output data/preprocessed/united/es/dependency/united_dev.json

python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/ES/dependency/test.conllu \
    --output data/preprocessed/united/es/dependency/united_test.json

# French
python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/FR/dependency/train.extra.conllu \
    --output data/preprocessed/united/fr/dependency/united_train.extra.json

python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/FR/dependency/validation.conllu \
    --output data/preprocessed/united/fr/dependency/united_dev.json

python3 scripts/preprocessing/preprocess_united.py \
    --input data/original/united/FR/dependency/test.conllu \
    --output data/preprocessed/united/fr/dependency/united_test.json

