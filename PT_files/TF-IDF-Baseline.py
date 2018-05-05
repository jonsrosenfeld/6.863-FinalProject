#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:38:08 2018

@author: elena
"""




from torchtext import data
from spacy import *
import torch
import numpy as np
import random

import torch.nn as nn
from torch.autograd import Variable
import pdb

import os
import time
import glob
from argparse import ArgumentParser
import torch.optim as O
from torchtext import datasets
import  torch.nn.functional as F





from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc

def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    import os, errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise
            

def get_batches(batch_nums, train_iterator, dev_iterator, test_iterator="", dset='train'):
    print('getting batches...')
    np.random.seed(13)
    random.seed(13)
    
    # pick data_iterator
    if dset=='train':
        data_iterator = train_iterator
    elif dset=='dev':
        data_iterator = dev_iterator
    elif dset=="test":
        print ("TEST")
        data_iterator=test_iterator
    
    # actually get batches
    num = 0
    batches = {}
    data_iterator.init_epoch() 
    for batch_idx, batch in enumerate(data_iterator):
        if batch_idx == batch_nums[num]:
            batches[batch_idx] = batch
            num +=1 

        if num == max(batch_nums):
            break
        elif num == len(batch_nums):
            print('found them all')
            break
    print ("NUMBER OF EXAMPLES")
    print (len(batches))
    return batches




inputs = data.Field()
answers = data.Field(sequential=False, unk_token=None)


train, dev, test = datasets.SST.splits(inputs, answers, fine_grained = False, train_subtrees = True, filter_pred=lambda ex: ex.label != 'neutral')
inputs.build_vocab(train, dev, test)
answers.build_vocab(train)

batch_nums = list(range(6920))
train_iter_n, dev_iter_n,test_iter_n = data.BucketIterator.splits((train, dev,test), batch_size=1, sort_key=lambda x: len(x.text),repeat=False, device=0)

data =get_batches(batch_nums, train_iter_n, dev_iter_n) 
data_dev=get_batches(batch_nums, train_iter_n, dev_iter_n,test_iterator=test_iter_n, dset='test')  #dev

tokenized_documents=[]
y_train=[]
for ind in range(6919):
    text = data[ind].text.data[:, 0]
    label=data[ind].label.data[0]

  #  print (text)
    words = [inputs.vocab.itos[i] for i in text]
    
    test_stringified=" ".join(words)
    tokenized_documents.append(test_stringified)

    labels=[answers.vocab.itos[label]]
    y_train.append(label)
   #print (test_stringified)
   # print (labels)
  #  print (ind)
  #  print ("__________________________")
   # print (labels)
tokenized_documents_test=[]
y_test=[]
for ind in range(1821):
    text = data_dev[ind].text.data[:, 0]
    label=data_dev[ind].label.data[0]
    
    words = [inputs.vocab.itos[i] for i in text]
    test_stringified=" ".join(words)
    tokenized_documents_test.append(test_stringified)
    y_test.append(label)


tokenize = lambda doc: doc.lower().split(" ")
sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
sklearn_representation_train = sklearn_tfidf.fit_transform(tokenized_documents)


#print (tokenized_documents[-1])
#print (y_train[-1])
#print (sklearn_representation_train[-1].toarray()[0].tolist())



sklearn_representation_dev=sklearn_tfidf.transform(tokenized_documents_test)
#print (tokenized_documents_test[-1])
#print (sklearn_representation_dev[-1].toarray()[0].tolist())


logisticRegr = LogisticRegression()
logisticRegr.fit(sklearn_representation_train, y_train)
score = logisticRegr.score(sklearn_representation_dev, y_test)

print ("Score " + str(score))
print ("DONE")


bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)

vectorizer = CountVectorizer()
BOW_train = bigram_vectorizer.fit_transform(tokenized_documents)
BOW_test=bigram_vectorizer.transform(tokenized_documents_test)

logisticRegr2 = LogisticRegression()
logisticRegr2.fit(BOW_train, y_train)

result=logisticRegr2.predict(BOW_test)
score2 = logisticRegr2.score(BOW_test, y_test)

print ("Score BOW "+str(score2))

print(classification_report(y_test,result))


def create_sum_vector_representation(data, n_el):
  word_vectors='glove.6B.100d'
  vector_cache=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt')
  if word_vectors:
    if os.path.isfile(vector_cache):
        inputs.vocab.vectors = torch.load(vector_cache)
    else:
        inputs.vocab.load_vectors(word_vectors)
        makedirs(os.path.dirname(vector_cache))
        torch.save(inputs.vocab.vectors, vector_cache)

  vector_list=[]

  vocab_size=len(inputs.vocab)
  print ("VOCAB_SIZE is "+str(vocab_size))
  
  
  mapper=nn.Embedding(vocab_size,100)
  mapper.weight.data = inputs.vocab.vectors
  print ("Vectors are initialized")
  for ind in range(n_el):    
    vecs = mapper(data[ind].text)
    
    text = data[ind].text.data[:, 0]
    words = [inputs.vocab.itos[i] for i in text]
    #print (words)
     
    v=vecs.data.cpu().numpy()
   # print (v.shape)
   # print (v)
    representation=np.sum(vecs.data.cpu().numpy(), axis=0)
   # print (representation.shape)
    vector_list.append(representation)
    
  
  array_x=np.vstack([y for y in vector_list])
 # print (array_x.shape)
  #print (representation)
  #print (len(v))
  #print (v[1])
  #print (v[-1])
  return array_x
  
train_av=create_sum_vector_representation(data, 6919)  
test_av=create_sum_vector_representation(data_dev, 1821)  
logisticRegr3=LogisticRegression()
logisticRegr3.fit(train_av, y_train)

result3=logisticRegr3.predict(test_av)
score3 = logisticRegr3.score(test_av, y_test)
print(classification_report(y_test,result3))
print (accuracy_score(y_test, result3))

for ind in range(1821):
    text = data_dev[ind].text.data[:, 0]
    words = [inputs.vocab.itos[i] for i in text]
    answer=answers.vocab.itos[y_test[ind]]

    print ("_________________________________________")
    print (' '.join(word for word in words))
    print ("True "+str(answers.vocab.itos[y_test[ind]])+" "+"Predicted "+str(answers.vocab.itos[result3[ind]]))

   # print (representation)
   # print (len(vecs.data.cpu().numpy()))