# CLCD: Cross-Lingual Contradiction Detection

[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/felipessalvatore/CLCD/blob/master/LICENSE)


## Introduction

In this is the repository you will find the data and the code for the experiments reported in the paper  [A logical-based corpus for cross-lingual evaluation](https://arxiv.org/abs/1905.05704).


## Install

To install all the libraries run:

```
$ bash install.sh
```

## Text Generation

The code to generate the synthetic dataset is in the folder `clcd/text_generation`. In the same folder you can find a detailed description of all the templates used to generate sentence pairs. 

To create all datasets both in English and Portuguese just run:

```
$ bash gen.sh
```

the result datasets will be stored in the folder `text_gen_output`.

The list of all used templates is displayed in the file `templates.pdf`.


## Dataset

All data used in the paper can be found in the folder `clcd_datasets/`.


## Reproducing the experiments

To train the different models you will need to use the file `basic_RNN_BERT_train.py`. It requires the path for training and testing datasets, it also requires a name to store all the results. For example: 

```
$ python basic_RNN_BERT_train.py clcd_datasets/Portuguese/boolean_coordination_pt_train.csv clcd_datasets/Portuguese/boolean_coordination_pt_test.csv pt_experiments 
```

## Citation
```
  @misc{clcd2019,
    author = {Felipe Salvatore},
    title = {Cross-Lingual Contradiction Detection},
    year = {2019},
    howpublished = {\url{https://github.com/felipessalvatore/CLCD}},
    note = {commit xxxxxxx}
  }
```
