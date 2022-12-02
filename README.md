<div align="center">    
 
# Bridging the Gap in Multilingual Semantic Role Labeling: a Language-Agnostic Approach     

[![Paper](http://img.shields.io/badge/paper-ACL--anthology-B31B1B.svg)](https://www.aclweb.org/anthology/2020.coling-main.120/)
[![Conference](http://img.shields.io/badge/COLING-2020-4b44ce.svg)](https://coling2020.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

</div>

<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> Table of contents</h2>

<details open="open">
  <ol>
    <li><a href="#about-the-project"> ➤ About the project</a></li>
    <li><a href="#abstract"> ➤ Abstract</a></li>
    <li><a href="#cite-this-work"> ➤ Cite this work</a></li>
    <li><a href="#results"> ➤ Results</a></li>
      <ul>
        <li><a href="#conll-2009">CoNLL-2009</a></li>
        <li><a href="#conll-2012">CoNLL-2012</a></li>
      </ul>
    </li>
    <li><a href="#how-to-use"> ➤ How to use</a>
      <ul>
        <li><a href="#getting-the-data">Getting the data</a></li>
        <li><a href="#data-preprocessing">Data preprocessing</a></li>
        <li><a href="#training-a-model">Training a model</a></li>
        <li><a href="#data-preprocessing">Evaluating a model</a></li>
        <li><a href="#data-preprocessing">Using a model for inference/prediction</a></li>
      </ul>
    </li>
    <li><a href="#acknowledgements"> ➤ Acknowledgements</a></li>
    <li><a href="#license"> ➤ License</a></li>
  </ol>
</details>

## About the project
This is the repository for the paper [*Bridging the Gap in Multilingual Semantic Role Labeling: a Language-Agnostic Approach*](https://www.aclweb.org/anthology/2020.coling-main.120/), presented at COLING 2020 by [Simone Conia](https://c-simone.github.io/) and [Roberto Navigli](https://www.diag.uniroma1.it/navigli/).


## Abstract
> Recent research indicates that taking advantage of complex syntactic features leads to favorable results in Semantic Role Labeling. Nonetheless, an analysis of the latest state-of-the-art multilingual systems reveals the difficulty of bridging the wide gap in performance between high-resource (e.g., English) and low-resource (e.g., German) settings. To overcome this issue, we propose a fully language-agnostic model that does away with morphological and syntactic features to achieve robustness across languages. Our approach outperforms the state of the art in all the languages of the CoNLL-2009 benchmark dataset, especially whenever a scarce amount of training data is available. Our objective is not to reject approaches that rely on syntax, rather to set a strong and consistent language-independent baseline for future innovations in Semantic Role Labeling.


## Cite this work
If you use any part of this work, please consider citing the paper as follows:

```
@inproceedings{conia-navigli-2020-multilingual-srl,
    title     = "Bridging the Gap in Multilingual {S}emantic {R}ole {L}abeling: {A} Language-Agnostic Approach",
    author    = "Conia, Simone and Navigli, Roberto",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics, COLING 2020",
    month     = dec,
    year      = "2020",
    address   = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url       = "https://aclanthology.org/2020.coling-main.120",
    doi       = "10.18653/v1/2020.coling-main.120",
    pages     = "1396--1410",
}
```

## Experiment results
Here we provide an overview of the results obtained by our model on CoNLL-2009 (English, Czech, German, Catalan, Spanish, Chinese) and CoNLL-2012.

### CoNLL-2009
Results on the CoNLL-2009 Shared Task by [Hajic et al., 2009. The CoNLL-2009 Shared Task: Syntactic and Semantic Dependencies in Multiple Languages](https://aclanthology.org/W09-1201/).

#### English
![Results on the validation and test sets of CoNLL-2009](/github/CoNLL-2009_english_results.png "English results on CoNLL-2009.")

#### Czech
![Results on the validation and test sets of CoNLL-2009](/github/CoNLL-2009_czech_results.png "Czech results on CoNLL-2009.")

#### German
![Results on the validation and test sets of CoNLL-2009](/github/CoNLL-2009_german_results.png "German results on CoNLL-2009.")

#### Catalan
![Results on the validation and test sets of CoNLL-2009](/github/CoNLL-2009_catalan_results.png "Catalan results on CoNLL-2009.")

#### Spanish
![Results on the validation and test sets of CoNLL-2009](/github/CoNLL-2009_spanish_results.png "Spanish results on CoNLL-2009.")

#### Chinese
![Results on the validation and test sets of CoNLL-2009](/github/CoNLL-2009_chinese_results.png "Chinese results on CoNLL-2009.")

### CoNLL-2012
Results on the CoNLL-2012 Shared Task by [Pradhan et al., 2012. CoNLL-2012 Shared Task: Modeling Multilingual Unrestricted Coreference in OntoNotes](https://aclanthology.org/W12-4501/).

![Results on the validation and test sets of CoNLL-2012](/github/CoNLL-2012_english_results.png "English results on CoNLL-2012.")



## How to use
You'll need a working Python environment to run the code. The recommended way to set up your environment is through the [Anaconda Python distribution](https://www.anaconda.com/download/) which provides the `conda` package manager. Anaconda can be installed in your user directory and does not interfere with the system Python installation. 

We use `conda` virtual environments to manage the project dependencies in
isolation. Thus, you can install our dependencies without causing conflicts with your
setup (even with different Python versions).

Run the following command and follow the steps to create a separate environment:
```bash
> bash setup.sh
> Enter environment name (recommended: multilingual-srl): multilingual-srl
> Enter python version (recommended: 3.8): 3.8
> Enter cuda version (e.g. '11.3' or 'none' to avoid installing cuda support):
```
All the code in this repository was tested using Python 3.8 and CUDA 11.3.


### Getting the data
> NOTE: If you don't want to retrain or evaluate a model, you can skip this step.

Depending on the task you want to perform (e.g., dependency-based SRL or span-based SRL),
you need to obtain some datasets (unfortunately, some of these datasets require a license fee).
> NOTE: Not all of the following datasets are required. E.g., if you are only interested
  in dependency-based SRL with PropBank labels, you just need CoNLL-2009. 

* [Hajic et al., 2009. The CoNLL-2009 Shared Task: Syntactic and Semantic Dependencies in Multiple Languages](https://aclanthology.org/W09-1201/).
The dataset is available on LDC ([LDC2012T04](https://catalog.ldc.upenn.edu/LDC2012T04)).
* [Pradhan et al., 2012. CoNLL-2012 Shared Task: Modeling Multilingual Unrestricted Coreference in OntoNotes](https://aclanthology.org/W12-4501/).
The dataset is available on LDC ([LDC2013T19](https://catalog.ldc.upenn.edu/LDC2013T19)).
* [OntoNotes 5](https://catalog.ldc.upenn.edu/LDC2013T19) tagged with [PropBank 3.x](https://github.com/propbank/propbank-release).
* [UniteD-SRL v2]() (Coming soon!).

Once you have downloaded and unzipped the data, place it in `data/original` as follows:
```
data/original/
├── conll2009
│   ├── ca
│   │   ├── CoNLL2009_dev.txt
│   │   ├── CoNLL2009_test.txt
│   │   └── CoNLL2009_train.txt
│   ├── cs
│   │   ├── CoNLL2009_dev.txt
│   │   ├── CoNLL2009_test-ood.txt
│   │   ├── CoNLL2009_test.txt
│   │   └── CoNLL2009_train.txt
│   ├── de
│   │   ├── CoNLL2009_dev.txt
...
├── conll2012
│   └── en
│       ├── CoNLL2012_dev.txt
│       ├── CoNLL2012_test.txt
│       └── CoNLL2012_train.txt
├── ontonotes
│   ├── ontonotes_dev.txt
│   ├── ontonotes_test-conll2012.txt
│   ├── ontonotes_test.txt
│   └── ontonotes_train.txt
...
└── united
    ├── EN
    │   ├── dependency
    │   │   ├── test.conllu
    │   │   ├── train.conllu
    │   │   ├── train.extra.conllu
    │   │   └── validation.conllu
    │   └── span
    │       ├── test.conllu
    │       ├── train.conllu
    │       ├── train.extra.conllu
    │       └── validation.conllu
    ├── ES
    │   ├── dependency
    │   │   ├── test.conllu
    │   │   ├── train.extra.conllu
    │   │   └── validation.conllu
...
```

**Note:** Make sure that the datasets are renamed as specified in the example above.
If you have your own datasets or the same datasets using a different name, check
the corresponding script (e.g., `scripts/preprocessing/preprocess_conll_2009.sh` for CoNLL-2009) and modify it accordingly.


### Data preprocessing
To preprocess the datasets, run the script `preprocess_<dataset_name>.sh` from the root directory of the project. For example, for CoNLL-2009:
```bash
bash scripts/preprocessing/preprocess_conll2009.sh
```
Then, compute the vocabularies:
```bash
bash scripts/preprocessing/compute_vocabulary.sh
```

After running the script, you will find the preprocessed datasets in `data/preprocessed/` as follows:
```
data/preprocessed/
├── conll2009
│   ├── ca
│   │   ├── CoNLL2009_dev.json
│   │   ├── CoNLL2009_test.json
│   │   ├── CoNLL2009_train.json
│   │   └── vocabulary.json
│   ├── cs
│   │   ├── CoNLL2009_dev.json
│   │   ├── CoNLL2009_test.json
│   │   ├── CoNLL2009_test_ood.json
│   │   ├── CoNLL2009_train.json
│   │   └── vocabulary.json
│   ├── de
│   │   ├── CoNLL2009_dev.json
...
```

#### VerbAtlas
If you have CoNLL-2009, CoNLL-2012 or OntoNotes, you can map the PropBank labels to the VerbAtlas labels (and then train a system to predict VerbAtlas labels directly).

1. Preprocess the dataset(s) as described above.
2. [Download VerbAtlas from the official website](https://verbatlas.org/download).
3. Extract the compressed folder and place it in `data/resources/`, i.e., `data/resources/verbatlas-1.1`.
4. Run the command `bash scripts/preprocessing/remap_pb2va.sh`.
5. Rerun the command `bash scripts/preprocessing/compute_vocabulary.sh`.
6. Run the Python script `python scripts/preprocessing/compute_candidates_from_verbatlas.py`

You should now see the preprocessed VerbAtlas dataset and the label vocabulary:
```
data/preprocessed/
├── conll2009
...
│   ├── en
│   │   ├── CoNLL2009_dev.json
│   │   ├── CoNLL2009_dev.va.json
│   │   ├── CoNLL2009_test.json
│   │   ├── CoNLL2009_test_ood.json
│   │   ├── CoNLL2009_test.va.json
│   │   ├── CoNLL2009_train.json
│   │   ├── CoNLL2009_train.va.json
│   │   ├── vocabulary.json
│   │   └── vocabulary.va.json
...
```

### Training a model
Once you have everything ready, training a model is quite simple. Just run the command:
```bash
export PYTHONPATH="$PWD" && \
python scripts/training/trainer.py [fit|validate|test] \
  --config path/to/base/configuration.yaml \
  --config path/to/dataset-specific/values.yaml
```

For example, training on CoNLL-2009 with PropBank labels using RoBERTa-base as the underlying language model:
```bash
export PYTHONPATH="$PWD" && \
python scripts/training/trainer.py fit \
  --config configurations/conll2009/base.yaml \
  --config configurations/conll2009/roberta/roberta-base-ft.yaml
```

Where `--config <configuration-file>` specifies the paths to the datasets to use for training, validation, and testing, the hyperparameters to use, etc. Take a look at the available configurations in `configurations/`. Feel free to contribute new configurations!

### Evaluating a model
If you have a checkpoint, you can evaluate the performance of the model on the default validation/development set by using the following command:
```bash
export PYTHONPATH="$PWD" && \
python scripts/training/trainer.py validate \
  --config configurations/conll2009/base.yaml \
  --config configurations/conll2009/roberta/roberta-base-ft.yaml \
  --ckpt_path path/to/checkpoint.ckpt
```

**Note:** The results are just an estimate (+/- 0.1) of the official scorers. This estimates are useful to simplify the code during training, validation and debugging without having to resort to the official scorers for CoNLL-2009 and CoNLL-2005 (used for CoNLL-2012), which are written in Perl. If you want to obtain the official scores and compare them with the literature, please refer to the section below.

Or you can evaluate the model on the default test set by changing `validate` with `test`:
```bash
export PYTHONPATH="$PWD" && \
python scripts/training/trainer.py test \
  --config configurations/conll2009/base.yaml \
  --config configurations/conll2009/roberta/roberta-base-ft.yaml \
  --ckpt_path path/to/checkpoint.ckpt
```

#### Specifying a custom evaluation dataset
You can evaluate a model on a custom dataset by specifying the path of the dataset with the `--data.test_path` argument:
```bash
export PYTHONPATH="$PWD" && \
python scripts/training/trainer.py test \
  --config configurations/conll2009/base.yaml \
  --config configurations/conll2009/roberta/roberta-base-ft.yaml \
  --ckpt_path path/to/checkpoint.ckpt \
  --data.test_path path/to/custom/dataset.json
```
The custom dataset must have the same format as the preprocessed datasets.

#### Comparing the results with the State of the Art
Each time you run the `trainer.py` script using `validate` or `test`, the script will also generate a `val_predictions.json` or `test_predictions.json` file. This file can be used to measure the performance of the model according to the official scorer of a task (e.g., CoNLL-2009).

##### CoNLL-2009
To obtain the official scores for the CoNLL-2009 task, run:
```bash
python scripts/evaluation/evaluate_on_conll2009.py \
  --gold data/original/conll2009/en/CoNLL2009_test.txt \
  --predictions path/to/test_predictions.json

# Output
  SYNTACTIC SCORES:
  Labeled   attachment score: 48711 / 48711 * 100 = 100.00 %
  Unlabeled attachment score: 48711 / 48711 * 100 = 100.00 %
  Label accuracy score:       48711 / 48711 * 100 = 100.00 %
  Exact syntactic match:      2000 / 2000 * 100 = 100.00 %

  SEMANTIC SCORES: 
  Labeled precision:          (18170 + 8802) / (20029 + 8987) * 100 = 92.96 %
  Labeled recall:             (18170 + 8802) / (19949 + 8987) * 100 = 93.21 %
  Labeled F1:                 93.08 
  Unlabeled precision:        (18926 + 8987) / (20029 + 8987) * 100 = 96.20 %
  Unlabeled recall:           (18926 + 8987) / (19949 + 8987) * 100 = 96.46 %
  Unlabeled F1:               96.33 
  Proposition precision:      6741 / 8987 * 100 = 75.01 %
  Proposition recall:         6741 / 8987 * 100 = 75.01 %
  Proposition F1:             75.01 
  Exact semantic match:       766 / 2000 * 100 = 38.30 %

  OVERALL MACRO SCORES (Wsem = 0.50):
  Labeled macro precision:    96.48 %
  Labeled macro recall:       96.61 %
  Labeled macro F1:           96.54 %
  Unlabeled macro precision:  98.10 %
  Unlabeled macro recall:     98.23 %
  Unlabeled macro F1:         98.17 %
  Exact overall match:        766 / 2000 * 100 = 38.30 %

  OVERALL MICRO SCORES:
  Labeled micro precision:    (48711 + 18170 + 8802) / (48711 + 20029 + 8987) * 100 = 97.37 %
  Labeled micro recall:       (48711 + 18170 + 8802) / (48711 + 19949 + 8987) * 100 = 97.47 %
  Labeled micro F1:           97.42 
  Unlabeled micro precision:  (48711 + 18926 + 8987) / (48711 + 20029 + 8987) * 100 = 98.58 %
  Unlabeled micro recall:     (48711 + 18926 + 8987) / (48711 + 19949 + 8987) * 100 = 98.68 %
  Unlabeled micro F1:         98.63 
```
Which means that model achieved about 93.1% in F1 score (`Labeled F1` in `SEMANTIC SCORES`). The script will also create a `test_predictions.conll` file in the same CoNLL format as the gold file.

##### CoNLL-2012
To obtain the official score for CoNLL-2012, run:
```bash
python scripts/evaluation/evaluate_on_conll2012.py \
  --gold data/original/conll2012/en/CoNLL2012_test.txt \
  --predictions predictions/conll2012/roberta-large-ft/test_predictions.json

# Output
Number of Sentences    :        9263
Number of Propositions :       26715
Percentage of perfect props :  75.72

              corr.  excess  missed    prec.    rec.      F1
------------------------------------------------------------
   Overall    54635    7467    6802    87.98   88.93   88.45
----------
      ARG0    12322     816     710    93.79   94.55   94.17
      ARG1    18953    1862    1638    91.05   92.05   91.55
      ARG2     6405     982     888    86.71   87.82   87.26
      ARG3      314     107     102    74.58   75.48   75.03
      ARG4      332      77      64    81.17   83.84   82.48
      ARG5        7       6       2    53.85   77.78   63.64
      ARGA        0       0       2     0.00    0.00    0.00
  ARGM-ADJ      164     102      92    61.65   64.06   62.84
  ARGM-ADV     1700     718     796    70.31   68.11   69.19
  ARGM-CAU      310     118      79    72.43   79.69   75.89
  ARGM-COM       24      31       6    43.64   80.00   56.47
  ARGM-DIR      278     111     163    71.47   63.04   66.99
  ARGM-DIS     2308     439     356    84.02   86.64   85.31
  ARGM-DSP        0       0       1     0.00    0.00    0.00
  ARGM-EXT      108      73      70    59.67   60.67   60.17
  ARGM-GOL       39      38      41    50.65   48.75   49.68
  ARGM-LOC     1192     335     359    78.06   76.85   77.45
  ARGM-LVB       66       8       6    89.19   91.67   90.41
  ARGM-MNR     1183     430     367    73.34   76.32   74.80
  ARGM-MOD     2210      52      29    97.70   98.70   98.20
  ARGM-NEG     1194      91      29    92.92   97.63   95.22
  ARGM-PNC       16      17      63    48.48   20.25   28.57
  ARGM-PRD       92     163     196    36.08   31.94   33.89
  ARGM-PRP      333     160     131    67.55   71.77   69.59
  ARGM-REC       15       4      20    78.95   42.86   55.56
  ARGM-TMP     3811     547     441    87.45   89.63   88.52
    R-ARG0      594      43      45    93.25   92.96   93.10
    R-ARG1      521      66      44    88.76   92.21   90.45
    R-ARG2       28      10      19    73.68   59.57   65.88
    R-ARG3        1       3       0    25.00  100.00   40.00
    R-ARG4        1       0       2   100.00   33.33   50.00
R-ARGM-ADV        0       0       2     0.00    0.00    0.00
R-ARGM-CAU        4       2       0    66.67  100.00   80.00
R-ARGM-DIR        0       1       1     0.00    0.00    0.00
R-ARGM-EXT        1       0       0   100.00  100.00  100.00
R-ARGM-LOC       54      29      15    65.06   78.26   71.05
R-ARGM-MNR        5       5       6    50.00   45.45   47.62
R-ARGM-PRD        0       0       1     0.00    0.00    0.00
R-ARGM-PRP        0       0       2     0.00    0.00    0.00
R-ARGM-TMP       50      21      14    70.42   78.12   74.07
------------------------------------------------------------
         V    26713       7       2    99.97   99.99   99.98
------------------------------------------------------------
```
Which means that model achieved about 88.5% in F1 score (`F1` in `Overall`).



### Using a model for prediction/inference
If you have a trained model checkpoint, you can use it to generate new SRL annotations.

We provide a simple script to annotate new text, which you can use as follows:
```bash
# Command
export PYTHONPATH="$PWD" && \
  python scripts/inference/predict.py \
  --checkpoint logs/ontonotes/roberta-base-ft_extended-vocab/checkpoints/epoch\=06-f1\=0.8938.ckpt \
  --vocabulary data/preprocessed/ontonotes/vocabulary.extended.json \
  --text "The quick brown fox jumps over the lazy dog."

# Output
{'sentence_id': 0, 'words': ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.'], 'predictions': {4: {'predicate': 'jump.03', 'roles': {(0, 4): 'ARG0', (5, 9): 'ARGM-DIR'}}}}
```

The script also takes care of sentence splitting.
```bash
export PYTHONPATH="$PWD" && \
  python scripts/inference/predict.py \
  --checkpoint logs/ontonotes/roberta-base-ft_extended-vocab/checkpoints/epoch\=06-f1\=0.8938.ckpt \
  --vocabulary data/preprocessed/ontonotes/vocabulary.extended.json \
  --text "The quick brown fox jumps over the lazy dog. Today I ate pizza. Semantic Role Labeling is the task of extracting the predicate-argument structures within a given sentence."

# Output
{'sentence_id': 0, 'words': ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.'], 'predictions': {4: {'predicate': 'jump.03', 'roles': {(0, 4): 'ARG0', (5, 9): 'ARGM-DIR'}}}}
{'sentence_id': 1, 'words': ['Today', 'I', 'ate', 'pizza', '.'], 'predictions': {2: {'predicate': 'eat.01', 'roles': {(0, 1): 'ARGM-TMP', (1, 2): 'ARG0', (3, 4): 'ARG1'}}}}
{'sentence_id': 2, 'words': ['Semantic', 'Role', 'Labeling', 'is', 'the', 'task', 'of', 'extracting', 'the', 'predicate', '-', 'argument', 'structures', 'within', 'a', 'given', 'sentence', '.'], 'predictions': {3: {'predicate': 'be.03', 'roles': {}}, 7: {'predicate': 'extract.01', 'roles': {(8, 17): 'ARG1'}}, 15: {'predicate': 'give.01', 'roles': {(16, 17): 'ARG1'}}}}
```

And lets you specify the language of the input text (default: `en`):
```bash
# Command
export PYTHONPATH="$PWD" && \
  python scripts/inference/predict.py \
  --checkpoint logs/ontonotes/xlm-roberta-base-ft_va/checkpoints/epoch\=11-f1\=0.8875.ckpt \
  --vocabulary data/preprocessed/ontonotes/vocabulary.va.extended.it.json \
  --language it --text "Mi chiamo Simone. Oggi ho mangiato un gelato."

# Output
{'sentence_id': 0, 'words': ['Mi', 'chiamo', 'Simone', '.'], 'predictions': {1: {'predicate': 'NAME', 'roles': {(0, 1): 'Theme', (2, 3): 'Attribute'}}}}
{'sentence_id': 1, 'words': ['Oggi', 'ho', 'mangiato', 'un', 'gelato', '.'], 'predictions': {2: {'predicate': 'EAT_BITE', 'roles': {(0, 1): 'Time', (3, 5): 'Patient'}}}}
```

You can also write the predictions directly to file:
```bash
export PYTHONPATH="$PWD" && \
  python scripts/inference/predict.py \
  --checkpoint logs/ontonotes/roberta-base-ft_extended-vocab/checkpoints/epoch\=06-f1\=0.8938.ckpt \
  --vocabulary data/preprocessed/ontonotes/vocabulary.extended.json \
  --text "The quick brown fox jumps over the lazy dog."
  --output_file out.jsonl

# The following predictions will be stored in 'out.jsonl'.
{'sentence_id': 0, 'words': ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.'], 'predictions': {4: {'predicate': 'jump.03', 'roles': {(0, 4): 'ARG0', (5, 9): 'ARGM-DIR'}}}}
```

Under the hood, we use [Stanza](https://stanfordnlp.github.io/stanza/) to preprocess the text. Thanks Stanza!


## Acknowledgements

The authors gratefully acknowledge the support of the [ERC Consolidator Grant MOUSSE No. 726487](http://mousse-project.org/) under the European Union’s Horizon 2020 research and innovation programme.

This work was supported in part by the MIUR under grant “Dipartimenti di eccellenza 2018-2022” of the Department of Computer Science of Sapienza University of Rome.


## License
This work (paper, models, checkpoints and all the contents of this repository) are licensed under Creative Commons Attribution-NonCommercial 4.0 International.

See LICENSE.txt for more details.

