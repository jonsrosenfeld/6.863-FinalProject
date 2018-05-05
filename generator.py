#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 19:53:03 2018

@author: jonathan
"""

import random
import numpy as np
#from itertools import izip

seed = 0
if seed!=0:
    rng=random.Random(seed)
else:
    rng=random.Random()


def wchoices(population,weights=None):
    if weights==None:
        weights=np.ones(len(population)).tolist()
    cumsum=[0]
    for w in weights:
        cumsum.append(w+cumsum[len(cumsum)-1])
    
    cumsum_vec= np.array(cumsum)
    point=rng.uniform(0,cumsum[len(cumsum)-1])
    ind_leq_vec=(cumsum_vec<=point)
    c=int(np.sum(np.ones(len(population)+1)[ind_leq_vec])-1)
    return [population[c]]
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def filterline(line):
    slist=line.split()
    if len(slist)>0:
        if is_number(slist[0]):    #grammer rules iff start with a wieght.
            s_comments_removed=line.split('#')[0] # remove comments
            slist=s_comments_removed.split() #note that name re-use is adecuate in this case as slist is identical apart from removed comment section
            L=len(slist) # note that L > 2
    
            weight=float(slist[0]) 
            try:
                LHS=slist[1]
            except:
                print('error reading grammar line: improper format')
                return None
            if L==3: #RHS is kept as a list of strings. 
                RHS=[slist[2]] # if L==3 it means that RHS contains only one string. i.e. len(RHS)=1
            else:
                RHS=slist[2:]
                if not(RHS):
                    print('error reading grammar line: RHS is empty')
                    return None
            return (weight,LHS,RHS)
        else:
            return ()
    else:
        return ()


    
def read_grammar(filename):
    with open(filename,'r') as grm:
        rule_dictionary={} #holds grammar rules
        weight_dictionary={} #hold wieghts associated with corresponding rules
        for line in grm:
            retval=filterline(line)
            if retval:
                (weight,LHS,RHS)=retval
                RHS_rules_list=rule_dictionary.get(LHS,False)
                RHS_weight_list=weight_dictionary.get(LHS,False)

                if not(RHS_rules_list):
                    rule_dictionary[LHS]=[RHS]
                    weight_dictionary[LHS]=[weight]
                else:
                    RHS_rules_list.append(RHS)
                    RHS_weight_list.append(weight)
            else:
                continue
        return rule_dictionary,weight_dictionary
    
def rule_select(LHS,rule_dictionary,weight_dictionary,term_symbol=None):
    '''
    receives a LHS (string) and selects, randomly, a RHS row vector out of matrix possible RHS expansions.
    random selection is relatively weighted according to the relative weights of all RHS possible expansions for given LHS.
    
    return value: selected RHS vector (list of strings) or term_symbol indicating the requested LHS was terminal.
    '''
    try:
        RHS_list=rule_dictionary.get(LHS,term_symbol) # if no such LHS is found it means it is terminal
        if RHS_list==term_symbol:
            return term_symbol
        
        weight_list=weight_dictionary.get(LHS)
    #    #rng=random.Random() #this is globaly defined
    #    choice=rng.choices(RHS_list,weight_list)
        choice=wchoices(RHS_list,weight_list) #as implemented to support older python 2.7
    
        return choice
    except:
        print('error')

#
def isterminal(LHS,rule_dictionary,weight_dictionary,term_symbol=None):
    retval=rule_select(LHS,rule_dictionary,weight_dictionary,term_symbol)
    if retval==term_symbol:
        return True
    else:
        return False



def rule_expantion_step(LHS,rule_dictionary,weight_dictionary,disp_tree=False,term_symbol=None):
    '''
    given a LHS (string), randomly expand until all terminal values reached.
    
    return string containing all terminals as traversed Left-searched
    retrun string tree
    
    '''
    curr_RHS=rule_select(LHS,rule_dictionary,weight_dictionary,term_symbol)
    if curr_RHS==term_symbol: 
        if disp_tree:
            return ''
        else:
            return LHS+' ' # this is a terminal string
    next_LHS_list=curr_RHS[0]
    substr=''
    
    for LHS_candidate in next_LHS_list:

        cont_str=rule_expantion_step(LHS_candidate,rule_dictionary,weight_dictionary,disp_tree,term_symbol)
        if disp_tree:
            if not(isterminal(LHS_candidate,rule_dictionary,weight_dictionary,term_symbol)):
                substr=substr+'('+LHS_candidate+' '+cont_str+')'
            else:
                substr=substr+LHS_candidate+' '+cont_str
        else:
            substr=substr+cont_str

    
    return substr


def rule_expantion(rule_dictionary,weight_dictionary,disp_tree=False,term_symbol=None):
    if disp_tree:
        fullstr='(START '+rule_expantion_step('START',rule_dictionary,weight_dictionary,disp_tree,term_symbol)+')'
    else:
        fullstr=rule_expantion_step('START',rule_dictionary,weight_dictionary,disp_tree,term_symbol)
    return fullstr

#### un-grammafier functions:
    
def alter_line(line, pre_terms_dict,vocab_to_preterm_dict,rule_dictionary,weight_dictionary):
    slist=line.split()
    alter_index=wchoices([i for i in range(len(slist)-1)])
    word=slist[alter_index[0]]
    term_list=word2preterms(word,vocab_to_preterm_dict) #returns a list of all pre-terminals
    while term_list==None: # the original sentence may contain 'hard wired words, which do not correspond to any preterminal'
        alter_index=wchoices([i for i in range(len(slist)-1)])
        word=slist[alter_index[0]]
        term_list=word2preterms(word,vocab_to_preterm_dict) #returns a list of all pre-terminals
    
    altered_word=choose_disjoint_word(term_list,pre_terms_dict,rule_dictionary,weight_dictionary)
            # last element - verify belongs to line-endings and discard
            #if not (slist.pop() in line_endings):
    slist[alter_index[0]]=altered_word
    return " ".join(slist)

def word2preterms(word,vocab_to_preterm_dict):
    return vocab_to_preterm_dict.get(word)

def choose_disjoint_word(term_list,pre_terms_dict,rule_dictionary,weight_dictionary):
#    key_list=[key for key in pre_term_dict.keys()]
    cand=wchoices([key for key in pre_terms_dict.keys()])
    while cand in term_list:
        cand=wchoices([key for key in pre_terms_dict.keys()])
    word=rule_select(cand[0],rule_dictionary,weight_dictionary)[0][0]
    return word
 
def get_terminal_structures(rule_dictionary,weight_dictionary):
    '''
    for a given rule_dictionary
    return:
        pre_terms_dict: dictionary containing all LHS which are pre-terminal {LHS: #terminals}
        vocab_to_preterm_dict: dictionary {termianl: [pre_term1,pre_term2,...]} - containing 
        all the potential preterms leading to a (terminal) word in the vocabulary 
    '''
    key_list=[key for key in rule_dictionary.keys()]
    pre_terms_dict={}
    vocab_to_preterm_dict={}
    for LHS in key_list:
        RHS_list=rule_dictionary[LHS]
        for RHS in RHS_list:
            cand_term=RHS[0]
            if not(len(RHS)==1): 
                break #the LHS leading to this RHS can not be is a preterminal 
            else:
                if isterminal(cand_term,rule_dictionary,weight_dictionary): #Indeed the LHS is a pre-term        
                    
                    val=pre_terms_dict.get(LHS,False)
                    if val==False:
                        pre_terms_dict[LHS]=1
                    else:
                        pre_terms_dict[LHS]+=1
                    
                    pre_term_list=vocab_to_preterm_dict.get(cand_term,False)
                    if pre_term_list==False:
                        vocab_to_preterm_dict[cand_term]=[LHS]
                    else:
                        vocab_to_preterm_dict[cand_term].append(LHS) #add the preterminal that led to this terminal
    return pre_terms_dict,vocab_to_preterm_dict

def gen_corpus(output_filename,rule_dictionary,weight_dictionary,corpus_size):
    
    with open(output_filename,'w') as fout:
        for it in range(corpus_size):
            line_out=rule_expantion(rule_dictionary,weight_dictionary)+'\n'
            fout.write(line_out)
    
def gen_ungram_corpus(corpus_filename,output_filename,rule_dictionary,weight_dictionary):

    pre_terms_dict,vocab_to_preterm_dict = get_terminal_structures(rule_dictionary,weight_dictionary)

    with open(corpus_filename,'r') as fin, open(output_filename,'w') as fout:
        for line in fin:
            line_out=alter_line(line,pre_terms_dict,vocab_to_preterm_dict,rule_dictionary,weight_dictionary)+'\n'
            fout.write(line_out)
def filter_ungram(filt_fn,ug_cand_fn,out_fn):
    with open(filt_fn,'r') as fflt, open(ug_cand_fn,'r') as fcand, open(out_fn,'w') as fout:
        for linef, linec in zip(fflt, fcand):
            if linef[0]=='f':
                fout.write(linec)