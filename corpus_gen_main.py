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
ungram_filename='corp1_ug_cand.txt'
filt_ungram_filename='corp1_ug.txt'
corpus_size=1000

### there is a manual step:
# parse corp1_ug_cand.txt to produce out_ug.txt
# this is done on athena by calling: 
#./parse -g grammar2 -i corp1_ug.txt -o out_ugr.txt

gen.gen_corpus(corpus_filename,rule_d,weight_d,corpus_size)
gen.gen_ungram_corpus(corpus_filename,ungram_filename,rule_d,weight_d)

gen.filter_ungram('out_ugr1.txt',ungram_filename,filt_ungram_filename)