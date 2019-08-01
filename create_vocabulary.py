#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 18:27:45 2019

@author: lorenzo
"""
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
lista = []

with open('vocabulary.txt','r') as v:
    for line in v:
        lista.append(line)
        
vectorizer.fit_transform(lista)