#!/bin/bash

python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/en/CoNLL2009_train.txt --output data/json/en/CoNLL2009_train.json
python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/en/CoNLL2009_dev.txt --output data/json/en/CoNLL2009_dev.json
python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/en/CoNLL2009_test.txt --output data/json/en/CoNLL2009_test.json
python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/en/CoNLL2009_test_ood.txt --output data/json/en/CoNLL2009_test_ood.json

python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/de/CoNLL2009_train.txt --output data/json/de/CoNLL2009_train.json
python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/de/CoNLL2009_dev.txt --output data/json/de/CoNLL2009_dev.json
python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/de/CoNLL2009_test.txt --output data/json/de/CoNLL2009_test.json
python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/de/CoNLL2009_test_ood.txt --output data/json/de/CoNLL2009_test_ood.json

python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/zh/CoNLL2009_train.txt --output data/json/zh/CoNLL2009_train.json
python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/zh/CoNLL2009_dev.txt --output data/json/zh/CoNLL2009_dev.json
python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/zh/CoNLL2009_test.txt --output data/json/zh/CoNLL2009_test.json

python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/ca/CoNLL2009_train.txt --output data/json/ca/CoNLL2009_train.json
python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/ca/CoNLL2009_dev.txt --output data/json/ca/CoNLL2009_dev.json
python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/ca/CoNLL2009_test.txt --output data/json/ca/CoNLL2009_test.json

python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/es/CoNLL2009_train.txt --output data/json/es/CoNLL2009_train.json
python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/es/CoNLL2009_dev.txt --output data/json/es/CoNLL2009_dev.json
python3 scripts/preprocess/preprocess_conll_2009.py --input data/txt/es/CoNLL2009_test.txt --output data/json/es/CoNLL2009_test.json

python3 scripts/preprocess/preprocess_conll_2009.py --czech --input data/txt/cz/CoNLL2009_train.txt --output data/json/cz/CoNLL2009_train.json
python3 scripts/preprocess/preprocess_conll_2009.py --czech --input data/txt/cz/CoNLL2009_dev.txt --output data/json/cz/CoNLL2009_dev.json
python3 scripts/preprocess/preprocess_conll_2009.py --czech --input data/txt/cz/CoNLL2009_test.txt --output data/json/cz/CoNLL2009_test.json
python3 scripts/preprocess/preprocess_conll_2009.py --czech --input data/txt/cz/CoNLL2009_test_ood.txt --output data/json/cz/CoNLL2009_test_ood.json
