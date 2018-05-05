#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:57:42 2018

@author: elena
"""

import os
import pdb
import torch
import numpy as np
from argparse import ArgumentParser
from torchtext import data, datasets
from scipy.special import expit as sigmoid
import random
import  torch.nn.functional as F

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

def get_args():
    parser = ArgumentParser(description='PyTorch/torchtext SST')
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--d_embed', type=int, default=300) #300 128 11
    parser.add_argument('--d_proj', type=int, default=300)
    parser.add_argument('--d_hidden', type=int, default=168)  # 128 100
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=100) # 100
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--dev_every', type=int, default=100) #100
    parser.add_argument('--save_every', type=int, default=100) # 1000
    parser.add_argument('--dp_ratio', type=int, default=0.2)
    parser.add_argument('--no-bidirectional', action='store_false', dest='birnn')
    parser.add_argument('--preserve-case', action='store_false', dest='lower')
    parser.add_argument('--no-projection', action='store_false', dest='projection')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
    parser.add_argument('--word_vectors', type=str, default=False ) #"glove.6B.300d" False
    parser.add_argument('--resume_snapshot', type=str, default='')
    args = parser.parse_args()
    return args



class LSTMSentiment(nn.Module):

    def __init__(self, config):
        super(LSTMSentiment, self).__init__()
        self.hidden_dim = config.d_hidden
        self.vocab_size = config.n_embed
        self.emb_dim = config.d_embed
        self.num_out = config.d_out
        self.batch_size = config.batch_size
        self.use_gpu = False #config.use_gpu
        self.num_labels = 2
        self.embed = nn.Embedding(self.vocab_size, self.emb_dim)
        self.lstm = nn.LSTM(input_size = self.emb_dim, hidden_size = self.hidden_dim)
        self.hidden_to_label = nn.Linear(self.hidden_dim, self.num_labels)

    def forward(self, batch):
        if self.use_gpu:
            self.hidden = (Variable(torch.zeros(1, batch.text.size()[1], self.hidden_dim).cuda()),
                            Variable(torch.zeros(1, batch.text.size()[1], self.hidden_dim).cuda()))
        else:
            self.hidden = (Variable(torch.zeros(1, batch.text.size()[1], self.hidden_dim)),
                            Variable(torch.zeros(1, batch.text.size()[1], self.hidden_dim)))

        vecs = self.embed(batch.text)
        lstm_out, self.hidden = self.lstm(vecs, self.hidden)
        logits = self.hidden_to_label(lstm_out[-1])
        return logits

def get_model(snapshot_file):
    print('loading', snapshot_file)
    #try:  # load onto gpu
    #    model = torch.load(snapshot_file)
   #     print('loaded onto gpu...')
   # except:  # load onto cpu
    model = torch.load(snapshot_file, map_location=lambda storage, loc: storage)
    print('loaded onto cpu...')
    return model

def get_batches(batch_nums, train_iterator, dev_iterator, dset='dev'):
    print('getting batches...')
    np.random.seed(13)
    random.seed(13)
    
    # pick data_iterator
    if dset=='train':
        data_iterator = train_iterator
    elif dset=='dev':
        data_iterator = dev_iterator
    
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
    return batches



args = get_args()


f=open('results_with_interpretation_sem2lm.txt','a')
dataset_type="Custom"


if dataset_type!="Custom":
  TEXT=data.Field(lower=args.lower)
  LABELS=data.Field(sequential=False, unk_token=None)
  train, dev, test = datasets.SST.splits(TEXT, LABELS, fine_grained = False, train_subtrees = True, filter_pred=lambda ex: ex.label != 'neutral')
  print ("Dataset_split done")    
  TEXT.build_vocab(train,dev,test)
  LABELS.build_vocab(train)

  train_iter_n, dev_iter_n, test_iter = data.BucketIterator.splits((train, dev, test), batch_size=1, sort_key=lambda x: len(x.text), device=0)

if dataset_type=="Custom":
     TEXT = data.Field(tokenize=data.get_tokenizer('spacy'), sequential=True, init_token='<SOS>', eos_token='<EOS>',lower=True)
     LABELS = data.Field(sequential=False, unk_token=None)
     
     train, dev =data.TabularDataset.splits(path='', format='tsv', train="train_g2.tsv", validation="dev_g2.tsv", fields=[('text', TEXT), ('label', LABELS)])  #
     print ("Dataset_split done")  # out_train.tsv  out_dev.tsv

     TEXT.build_vocab(train,dev, max_size=500)
     LABELS.build_vocab(train)

     train_iter_n, dev_iter_n = data.BucketIterator.splits((train, dev), batch_size=1, sort_key=lambda x: len(x.text), device=0)
  
  
#train, dev =data.TabularDataset.splits(path='', format='tsv', train="out_train.tsv", validation="out_dev.tsv", fields=[('text', TEXT), ('label', LABELS)])  #
#train_iter_n, dev_iter_n = data.BucketIterator.splits((train, dev), batch_size=1, sort_key=lambda x: len(x.text), device=0)

model=get_model("results/best_snapshot_devacc_87.3_devloss_0.34375759959220886_iter_7900_model.pt")
model.eval()
dev_iter_n.init_epoch()
batch_nums = list(range(2000)) #501 872 2000
print (len(batch_nums))
data_b =get_batches(batch_nums, train_iter_n, dev_iter_n) 
for ind in range(1999): # 500
    print ("START-EXAMPLE")
    print (ind)
    answer = model(data_b[ind]) 
    #print (answer)
    prob=F.softmax(answer)
    text = data_b[ind].text.data[:, 0]
    words = [TEXT.vocab.itos[i] for i in text]
    words_string=" ".join(words)
    #print (words)
    #print ("Soft-maxed predictions")
    #print (prob)
    
    tens=prob.data.numpy()
    #print (tens.shape)
    if (abs(prob.data[0][0]-prob.data[0][1])<0.1):
        print ("The system is not sure about the class")
   # print ("Non-normalized_logits")
   #print (answer)
    
    #print ("Predicted_label "+ str(torch.max(prob, 1)[1])+" True label "+ str(data[ind].label.data[0]))
    print ("-------------------------------")
    write_string=words_string+" "+"Predicted_label "+ str(torch.max(prob, 1)[1].data[0])+" True label "+ str(data_b[ind].label.data[0])+ " Certainty "+str(abs(prob.data[0][0]-prob.data[0][1]))
    print (write_string + "\n")
    f.write(write_string)
    f.write("\n")
f.close()
print ("NUMBER OF TEST BATCHES "+str(len(data_b)))