#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 14:07:01 2018

test script for ungrammatical generator
loosly - should scan a file which has 1 sentence per line
alter 1 word in every sentence to a disjoint pre-terminal set

@author: jonathan
"""


def alter_line(line,line_endings):
    slist=line.split()
    alter_index=wchoices([i for i in range(len(slist)-2)])
    word=slist[alter_index][0]
    term_list=word2preterms(word,vocab_to_preterm_dict) #returns a list of all pre-terminals
    altered_word=choose_disjoint_word(term_lis)
        # last element - verify belongs to line-endings and discard
        #if not (slist.pop() in line_endings):
    slist[alter_index]=altered_word
    return "".join(slist)

def word2preterms(word,vocab_to_preterm_dict):
    return vocab_to_preterm_dict.get(word)

def choose_disjoint_word(term_list,pre_term_dict)
    key_list=[key for key in pre_term_dict.keys()]
    cand=wchoices()
    while cand in term_list:
        cand=wchoices([key for key in pre_term_dict.keys()])
    word=rule_select(cand,rule_dictionary,weight_dictionary)
    retrun word
 
def get_terminal_structures(rule_dictionary):
    '''
    for a given rule_dictionary
    return:
        pre_terms_list: list containing all LHS which are pre-terminal
        vocab_to_preterm_dict: dictionary {termianl: [pre_term1,pre_term2,...]} - containing 
        all the potential preterms leading to a (terminal) word in the vocabulary 
    '''
    key_list=[key for key in rule_dictionary.keys()]
    pre_terms_list=[]
    vocab_to_preterm_dict={}
    for LHS in key_list:
        RHS=rule_dictionary[LHS]
        if len(RHS)==1: #this LHS might be is a preterm
                if isterm(RHS[0]): #Indeed the LHS is a pre-term        
                    pre_terms_list.append(LHS)
                    pre_term_list=vocab_to_preterm_dict.get(RHS[0],False)
                    if pre_term_list==False:
                        vocab_to_preterm_dict[RHS[0]]=[LHS]
                    else:
                        vocab_to_preterm_dict[RHS[0]].append(LHS) #add the preterminal that led to this terminal
    return pre_terms_list,vocab_to_preterm_dict


    
def gen_ungram_corpus(corpus_filename,output_filename):
    with open(filename,'r') as corp:
        for line in corp:
            alternate=alter_line(line)
            # write line to tile
            
