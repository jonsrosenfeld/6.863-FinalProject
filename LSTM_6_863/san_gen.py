#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 14:14:03 2018

@author: jonathan
"""
import numpy as np


def dec2ab(num_dec,N):
    s=format(num_dec, '0'+str(N)+'b')#str.format('%'+str(N)+'s', int.toBinaryString(num_dec)).replace(" ", "0")
    s=s.replace("0","a ")
    s=s.replace("1","b ")
    return s
    

def get_number_sets(sent_len):
    
    N=sent_len
    ungram_num_list=[i for i in range(2**N)] #init as all numbers 0..2^N-1
    if N%2 !=0:
        gram_num_list=[]
    else:
        print('N='+str(N)+',=>'+str(2**(int(N/2))-1))
        gram_num_list=[2**(int(N/2))-1] # this is the only grammatical sequence corresponding number for sequence length of length N
        ungram_num_list.remove(gram_num_list[-1]) # remove the grammatical corresponding number to get the ungrammatical number set
    return gram_num_list,ungram_num_list


def gen_data(max_sent_len,output_gr_filename,output_ungr_filename):
    with open(output_gr_filename,'w') as fgr, open(output_ungr_filename,'w') as fug:
        for N in np.arange(1,max_sent_len+1):
            gram_num_list,ungram_num_list=get_number_sets(N)
            if gram_num_list:
                line_gr=dec2ab(gram_num_list[-1],N)+'GR\n'
                fgr.write(line_gr)
            for entry in range(len(ungram_num_list)):
                line_ug=dec2ab(ungram_num_list[entry],N)+'UGR\n'
                fug.write(line_ug)

output_gr_filename='gr_data.txt'
output_ungr_filename='ug_data.txt'

gen_data(10,output_gr_filename,output_ungr_filename)
