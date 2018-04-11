#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:31:37 2018

@author: elena
"""

from random import *
def generate_sequence(max_length):
    n=randrange(1, max_length)
    print (n)
    a=["a" for x in range(n)]
    b=["b" for x in range(n)]
    ab=a+b
    print ("AB "+str(ab))
    return ab
    ""
    
def misscount_sequence(sequence):
    sequence_length=len(sequence)
    print ("Sequence_length "+str(sequence_length))
    random_delete=randrange(0, sequence_length)
    temp_by_value = sequence[:]
    
    misscount_sequence=temp_by_value.pop(random_delete)
    if count_the_numbers(temp_by_value):
        print ("BALANCED NUMBER OF A AND B")
        return temp_by_value
    else:
        return temp_by_value
    
    
def permute_sequence(sequence):
    sequence_length=len(sequence)
    print ("Sequence_length "+str(sequence_length))
    j=randrange(0, sequence_length)
    i=randrange(0, sequence_length)
    print (j , i)
    temp_by_value=sequence[:]
    temp_by_value[i], temp_by_value[j] = temp_by_value[j], temp_by_value[i]
    if check_the_order(temp_by_value):
        print ("IDENTITY_PERMUTATION")
        return temp_by_value
    else: return temp_by_value
    
def count_the_numbers(sequence):
    a_counter=0
    b_counter=0
    for element in sequence:
        if element=="a":
            a_counter+=1
        if element=="b":
            b_counter+=1
    if a_counter==b_counter:
      return True
    else: return False
        
    ""

def check_the_order(sequence):
    switched_to_b=0
    for element in sequence:
        if element=="b":
          switched_to_b=1
          continue
        if switched_to_b==1 and element=="a":
            return False
    return True

def one_change(alliteration_type,sequence):
    change_happend=0
    
    if alliteration_type=="both":
        random_permute=randrange(0,2)
        print ("Random permute "+str(random_permute))
        if random_permute==0:
            if len(sequence)>1: 
                transformed=permute_sequence(sequence)
                return transformed
            else:
                return sequence
        else:
           if len(sequence)>1: 
               transformed=misscount_sequence(sequence)
               return transformed
           else:
               return sequence
            # misscount

    if alliteration_type=="permute":
       if len(sequence)>1:
           transformed=permute_sequence(sequence)
           return transformed
       else: return sequence
    if alliteration_type=="misscount":
       if len(sequence)>1: 
           transformed=misscount_sequence(sequence)
           return transformed
       else: return sequence
    
        
def get_the_number_of(symbol,sequence):
    ""

        
          
 
def get_ungrammaticals(number, max_length,alliteration_number,alliteration_type="both"):  #values both,permute, misscount
    ungrammatical_sequences=[]
    while len(ungrammatical_sequences)<number:
        initial_sequence=generate_sequence(max_length)
        print ("INITIAL SEQUENCE")
        print (initial_sequence)
        for n in range(0,alliteration_number):
            initial_sequence=one_change(alliteration_type,initial_sequence)
        if count_the_numbers(initial_sequence)==False or check_the_order(initial_sequence)==False:
            ungrammatical_sequences.append(initial_sequence)
        else: continue 
    for sequence in ungrammatical_sequences:
       print (sequence)

def get_grammatical(number,max_length):
   grammatical_sequences=[] 
   while len(grammatical_sequences)<number:
       initial_sequence=generate_sequence(max_length)
       grammatical_sequences.append(initial_sequence)
   for sequence in grammatical_sequences:
       print (sequence)
   return grammatical_sequences 
   


def save_sequence(List_of_sentences, address):
    f=open(address, 'a')
    for sequence in List_of_sentences:
        for word in sequence:
            f.write(word+" ")
        f.write("GR ")
        f.write("\n")
    
      
def remove_all_sequences_shorter_then(number) :
    ""
    
    
    
'''
sequence=generate_sequence(10)
t=misscount_sequence(sequence)

print (sequence)
permute_sequence(sequence)
print (t)
'''
#get_ungrammaticals(100,10,1,alliteration_type="misscount")
t=get_grammatical(100,10)
save_sequence(t,"test_output")
