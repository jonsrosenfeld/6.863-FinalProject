#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 14:07:15 2018

@author: jonathan
"""
import random
import numpy as np
seed = 0
if seed!=0:
    rng=random.Random(seed)
else:
    rng=random.Random()


def wchoices(population,weights):
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
    RHS_list=rule_dictionary.get(LHS,term_symbol) # if no such LHS is found it means it is terminal
    if RHS_list==term_symbol:
        return term_symbol
    
    weight_list=weight_dictionary.get(LHS)
#    #rng=random.Random() #this is globaly defined
#    choice=rng.choices(RHS_list,weight_list)
    choice=wchoices(RHS_list,weight_list) #as implemented to support older python 2.7

    return choice

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

seed=0
rule_d,weight_d=read_grammar('grammar4')
for iter in range(10):
    print(rule_expantion(rule_d,weight_d,False))
    

    
    
    
        
        
        #        print(line)