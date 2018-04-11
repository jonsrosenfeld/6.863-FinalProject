# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import random
import shutil

def merge_files(file1,file2,output):
    with open(output,'wb') as wfd:
     for f in [file1,file2]:
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd, 1024*1024*10)


def mark_examples_as(file1, mark): # don't forget to add " " before the mark
    with open(file1, 'r') as f:
      file_lines = [''.join([x.strip(), mark, '\n']) for x in f.readlines()]

    with open(file1, 'w') as f:
      f.writelines(file_lines) 
    
def shuffle_file(file1):
    lines = open(file1).readlines()
    random.shuffle(lines)
    open(file1, 'w').writelines(lines)
    

def process(gramatical,ungrammatical,output):
    mark_examples_as(gramatical," GR")
    mark_examples_as(ungrammatical, " NG")
    merge_files(gramatical,ungrammatical,output)
    shuffle_file(output)
    
    
process("corp1.txt","corp1_ug.txt","train_big.txt")