#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 14:14:03 2018

@author: jonathan
"""
import numpy as np
import random

def dec2binstr(num_dec,N):
    s=format(num_dec, '0'+str(N)+'b')
    return s
def dec2ab(num_dec,N):
    s= dec2binstr(num_dec,N)#str.format('%'+str(N)+'s', int.toBinaryString(num_dec)).replace(" ", "0")
    return alterbin2ab(s)

def alterbin2ab(num_bin_str):
    s=num_bin_str
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

def get_number_gr(sent_len):
    N=sent_len
    if N%2 !=0:
        gram_num_list=[]
    else:
#        print('N='+str(N)+',=>'+str(2**(int(N/2))-1))
        gram_num_list=[2**(int(N/2))-1] # this is the only grammatical sequence corresponding number for sequence length of length N
    return gram_num_list

def gen_full_data(max_sent_len,output_gr_filename,output_ungr_filename):
    with open(output_gr_filename,'w') as fgr, open(output_ungr_filename,'w') as fug:
        for N in np.arange(1,max_sent_len+1):
            gram_num_list,ungram_num_list=get_number_sets(N)
            if gram_num_list:
                line_gr=dec2ab(gram_num_list[-1],N)+'GR\n'
                fgr.write(line_gr)
            for entry in range(len(ungram_num_list)):
                line_ug=dec2ab(ungram_num_list[entry],N)+'UGR\n'
                fug.write(line_ug)

def gen_gr_only(max_sent_len,output_gr_filename):
    with open(output_gr_filename,'w') as fgr:
        for N in np.arange(1,max_sent_len+1):
            gram_num_list=get_number_gr(N)
            if gram_num_list:
                line_gr=dec2ab(gram_num_list[-1],N)+'GR\n'
                fgr.write(line_gr)                
            
def gen_ug_template(max_sent_len,output_ug_filename,ungram_func):
    with open(output_ug_filename,'w') as fug:
        for N in np.arange(1,max_sent_len+1):
            gram_num_list=get_number_gr(N)
            if gram_num_list:
                gr_binstr=dec2binstr(gram_num_list[-1],N)
                ug_binstr_list=ungram_func(gr_binstr)
                for ug_binstr in ug_binstr_list:
                    line_ug=alterbin2ab(ug_binstr)+'UGR\n'
                    fug.write(line_ug)

def gen_ug_hammdist(max_sent_len,output_ug_filename,dist):
    def hammdist(s):
        return hamming(s, dist)
    gen_ug_template(max_sent_len,output_ug_filename,hammdist)
    
def gen_ug_subset(max_sent_len,output_ug_filename):
    with open(output_ug_filename,'w') as fug:
        gr_binstr=dec2binstr(get_number_gr(max_sent_len)[-1],max_sent_len)
        ug_binstr_list=substring(gr_binstr)
        for N in np.arange(1,max_sent_len+1):
            gram_num_list=get_number_gr(N)
            if gram_num_list:
                gr_binstr=dec2binstr(gram_num_list[-1],N)
                if gr_binstr in ug_binstr_list:
                    ug_binstr_list.remove(gr_binstr)
                       
        for ug_binstr in ug_binstr_list:
            line_ug=alterbin2ab(ug_binstr)+'UGR\n'
            fug.write(line_ug)

def flip(c):
    return str(1-int(c))

def flip_s(s, i):
    t =  s[:i]+flip(s[i])+s[i+1:]
    return t

def hamming(s, k):
    if k>1:
        c = s[-1]
        s1 = [y+c for y in hamming(s[:-1], k)] if len(s) > k else []
        s2 = [y+flip(c) for y in hamming(s[:-1], k-1)]
        r = []
        r.extend(s1)
        r.extend(s2)
        return r
    else:
        return [flip_s(s,i) for i in range(len(s))]

def substring(s,inclusive=False):
    substr_list=[]
    for startind in range(len(s)-1):
        for endind in np.arange(startind,len(s)+1):
            cand_str=s[startind:endind]
            if cand_str not in substr_list:
                substr_list.append(cand_str)
    if not inclusive:
        substr_list.remove('')
        substr_list.remove(s)
    return substr_list


gen_gr_only(50,'gr_data.txt')
gen_ug_hammdist(50,'ug_data_hamm1.txt',1) ### 'easy'. of the form ababbb this gives changes of the form ababbb
gen_ug_subset(50,'ug_data_subset.txt')


output_gr_filename = 'gr_full_data.txt'
output_ungr_filename = 'ug_full_data.txt'
gen_full_data(10,output_gr_filename,output_ungr_filename)