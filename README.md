# Multilingual-SRL (COLING 2020)
This is the repository for the paper *Bridging the Gap in Multilingual Semantic Role Labeling: a Language-Agnostic Approach*,
to be presented at COLING 2020 by Simone Conia and Roberto Navigli.

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/) 


## Abstract
> Recent research indicates that taking advantage of complex syntactic features leads to favorable results in Semantic Role Labeling. 
  Nonetheless, an analysis of the latest state-of-the-art mul- tilingual systems reveals the difficulty of bridging the wide gap in
  performance between high- resource (e.g., English) and low-resource (e.g., German) settings.
  To overcome this issue, we pro- pose a fully language-agnostic model that does away with morphological and syntactic features
  to achieve robustness across languages.
  Our approach outperforms the state of the art in all the lan- guages of the CoNLL-2009 benchmark dataset,
  especially whenever a scarce amount of training data is available. Our objective is not to reject approaches that rely on syntax,
  rather to set a strong and consistent language-independent baseline for future innovations in Semantic Role Labeling.


## Download
You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/SapienzaNLP/multi-srl.git

or [download a zip archive](https://github.com/SapienzaNLP/multi-srl/archive/master.zip).


## Dependencies
You'll need a working Python environment to run the code.
The recommended way to set up your environment is through the
[Anaconda Python distribution](https://www.anaconda.com/download/) which
provides the `conda` package manager.
Anaconda can be installed in your user directory and does not interfere with
the system Python installation.
The required dependencies are specified in the file `environment.yml`.

We use `conda` virtual environments to manage the project dependencies in
isolation.
Thus, you can install our dependencies without causing conflicts with your
setup (even with different Python versions).

Run the following command in the repository folder (where `environment.yml`
is located) to create a separate environment and install all required
dependencies in it:

    conda env create -f environment.yml

This project currently depends on:
* PyTorch 1.5
* PyTorch Lightning 0.8.5

We are in the process of updating the code to PyTorch 1.7 and PyTorch Lightning 1.0.

## Cite this work
    @inproceedings{conia-and-navigli-2020-multilingual-srl,
      title     = {Bridging the Gap in Multilingual {S}emantic {R}ole {L}abeling: {A} Language-Agnostic Approach},
      author    = {Conia, Simone and Navigli, Roberto},
      booktitle = {Proceedings of the 28th International Conference on Computational Linguistics, COLING 2020},
      year      = {2020}
    }
