#!/bin/bash

python3 scripts/preprocess/collect_data_info.py --input data/json/en/CoNLL2009_train.json
python3 scripts/preprocess/collect_data_info.py --input data/json/es/CoNLL2009_train.json
python3 scripts/preprocess/collect_data_info.py --input data/json/ca/CoNLL2009_train.json
python3 scripts/preprocess/collect_data_info.py --input data/json/de/CoNLL2009_train.json
python3 scripts/preprocess/collect_data_info.py --input data/json/zh/CoNLL2009_train.json
python3 scripts/preprocess/collect_data_info.py --input data/json/cz/CoNLL2009_train.json --czech
python3 scripts/preprocess/collect_data_info.py --input data/json/en/CoNLL2012_train.json --conll_2012