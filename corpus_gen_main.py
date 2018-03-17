#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 14:07:15 2018

@author: jonathan
"""
import generator as gen

seed=0
rule_d,weight_d=gen.read_grammar('grammar2')
corpus_filename='corp1.txt'
output_filename='corp1_ug.txt'
corpus_size=1000


gen.gen_corpus(corpus_filename,rule_d,weight_d,corpus_size)
gen.gen_ungram_corpus(corpus_filename,output_filename,rule_d,weight_d)

