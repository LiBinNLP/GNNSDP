#!D:/Code/python
# -*- coding: utf-8 -*-
# @Time : 2021/11/13 11:45
# @Author : libin
# @File : np_chunking.py
# @Software: PyCharm
import nltk

sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"), ("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]
grammar = "NP: {<DT>?<JJ>*<NN>}"
cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence)
print(result)