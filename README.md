# GNNSDP

A higher-order semantic dependency parser using graph neural networks, implemented with graph convolutional networks and graph attention networks.
HOSDP is a model extended biaffine parser. 

Code for the paper **Improving semantic dependency parsing with higher-order information encoded by graph neural networks (Applied Sciences).

## Dataset
We conduct experiment on SemEval 2015 Task 18 English dataset [`SemEval2015 Task 18`](https://alt.qcri.org/semeval2015/task18/). 
Trial data has been provided in this repository, full dataset [`LDC2016T10`](https://catalog.ldc.upenn.edu/LDC2016T10) are available in LDC official site.

## Installation

```sh
$ pip install -r requirements.txt
```


As a prerequisite, the following requirements should be satisfied:
* `python`: >= 3.7
* [`pytorch`](https://github.com/pytorch/pytorch): >= 1.7
* [`transformers`](https://github.com/huggingface/transformers): >= 4.0


## Training

To train a higher-order semantic dependency parser, you need to provide initial parsing result file (conll-u format),
because the initial adjacency matrix of semantic dependency graph is needed to build graph neural networks. 
Put the initial result predicted by biaffine parser (or any other semantic dependency parser) 
and specify the file path in `hosdp.py`.
Some hyperparameters is configured in the file `hosdp.ini`
```sh
$ python hosdp.py train -b -d 0 -p /HOSDP/output/trial/model -c /HOSDP/hosdp.ini
```

## Evaluation

To evaluate trained model, you need to run the following command.
```sh
$ python hosdp.py evaluate -d 0 -p /HOSDP/output/trial/model -c /HOSDP/hosdp.ini
```

## Prediction
To write predicted result in a output file, you need to run the following command. The output file path can be specified in `hosdp.py`. 
```sh
$ python hosdp.py test -b -d 0 -p /HOSDP/output/trial/model -c /HOSDP/hosdp.ini
```

## Citation
If you use the code of this repository, please cite our paper:
```
@Article{app12084089,
AUTHOR = {Li, Bin and Fan, Yunlong and Sataer, Yikemaiti and Gao, Zhiqiang and Gui, Yaocheng},
TITLE = {Improving Semantic Dependency Parsing with Higher-Order Information Encoded by Graph Neural Networks},
JOURNAL = {Applied Sciences},
VOLUME = {12},
YEAR = {2022},
NUMBER = {8},
ARTICLE-NUMBER = {4089},
URL = {https://www.mdpi.com/2076-3417/12/8/4089},
ISSN = {2076-3417},
DOI = {10.3390/app12084089}
}
```
