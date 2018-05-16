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
    parser = ArgumentParser(description='Loading_models')
    parser.add_argument('--dataset_type', type=str,default='Custom_with_test', help='Should always be Custom with text for this project')
    parser.add_argument('--number_of_examples', type=int , default=4000, help='Number of examples to predict, will throw an error if the target file has less examples then listed')
    parser.add_argument('--train_address', type=str, default="Datasets/different_size_train_grammar2/grammar2_train-15000.tsv", help='training set address')
    parser.add_argument('--test_address', type=str, required=True, help='dev(test) set address')
    parser.add_argument('--analysis_file_address',required=True, help='analysis file name, will be in the prediction_results folder',type=str)
    parser.add_argument('--model_address',required=True, type=str, help='Pre-trained model address, usually in pre-trained_models folder' )
    #parser.add_argument('--test_adress', type=str,default="Datasets/different_size_train_grammar2/grammar2_train-15000.tsv", help='test set address, please use the same address as you did for --train_address since we do not have test')
    # NOT USED FOR THIS DATASEt
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
    parser.add_argument('--word_vectors', type=str, default=False ) #"glove.6B.300d" False
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
           # print ("GPU")
            self.hidden = (Variable(torch.zeros(1, batch.text.size()[1], self.hidden_dim).cuda()),
                            Variable(torch.zeros(1, batch.text.size()[1], self.hidden_dim).cuda()))
        else:
            #print ("CPU")
            self.hidden = (Variable(torch.zeros(1, batch.text.size()[1], self.hidden_dim)),
                            Variable(torch.zeros(1, batch.text.size()[1], self.hidden_dim)))

        vecs = self.embed(batch.text)
        lstm_out, self.hidden = self.lstm(vecs, self.hidden)
        logits = self.hidden_to_label(lstm_out[-1])
        return logits



def get_model(snapshot_file):
    print('loading', snapshot_file)
    model = torch.load(snapshot_file, map_location=lambda storage, loc: storage)
    return model

def get_batches(batch_nums, train_iterator, dev_iterator,test_iterator=None,  dset='dev'):
    if dset=='train':
        data_iterator = train_iterator
    if dset=='dev':
        data_iterator = dev_iterator
    if dset=='test':
        data_iterator = test_iterator
    # actually get batches
    num = 0
    batches = {}
    data_iterator.init_epoch() 
    for batch_idx, batch in enumerate(data_iterator):
        if batch_idx == batch_nums[num]:
            batches[batch_idx] = batch
            num +=1 
    return batches

def predict_and_analyze(dataset_type,number_of_example=0, train_adress="",dev_adress="", test_adress="",analysis_log="",model_adress=""):

    if dataset_type=="SST":
       TEXT=data.Field(lower=args.lower)
       LABELS=data.Field(sequential=False, unk_token=None)
       train, dev, test = datasets.SST.splits(TEXT, LABELS, fine_grained = False, train_subtrees = True, filter_pred=lambda ex: ex.label != 'neutral')
       print ("Dataset_split done")    
       TEXT.build_vocab(train,dev,test)
       LABELS.build_vocab(train)

       train_iter_n, dev_iter_n, test_iter = data.BucketIterator.splits((train, dev, test), batch_size=1, sort_key=lambda x: len(x.text), device=None)

    if dataset_type=="Custom":
       TEXT = data.Field(tokenize=data.get_tokenizer('spacy'), sequential=True, init_token='<SOS>', eos_token='<EOS>',lower=True)
       LABELS = data.Field(sequential=False, unk_token=None)
     
       train, dev =data.TabularDataset.splits(path='', format='tsv', train=train_adress, validation=dev_adress, fields=[('text', TEXT), ('label', LABELS)])  #
       print ("Dataset_split done")  # out_train.tsv  out_dev.tsv

       TEXT.build_vocab(train,dev, max_size=500)
       LABELS.build_vocab(train)
     
       #print (LABELS.vocab.itos)

       train_iter_n, dev_iter_n = data.BucketIterator.splits((train, dev), batch_size=1, sort_key=lambda x: len(x.text), device=None)
     

    if dataset_type=="Custom_with_test":
       TEXT = data.Field(tokenize=data.get_tokenizer('spacy'), sequential=True, init_token='<SOS>', eos_token='<EOS>',lower=True)
       LABELS = data.Field(sequential=False, unk_token=None)
     
       train, dev,test =data.TabularDataset.splits(path='', format='tsv', train=train_adress, validation=dev_adress, test=test_adress, fields=[('text', TEXT), ('label', LABELS)])  #
       print ("------------------------START OUTPUT------------------------------")
       print ("Dataset_split done")  # out_train.tsv  out_dev.tsv
     
       TEXT.build_vocab(train,dev,test, max_size=500)
       LABELS.build_vocab(train)

       train_iter_n, dev_iter_n, test_iter_n = data.BucketIterator.splits((train, dev,test), batch_size=1, sort_key=None, sort=None, device=None)
    
    model=get_model(model_adress)
    model.use_gpu=False
    model.eval()
    f=open('prediction_results/'+analysis_log,'a')
    dev_iter_n.init_epoch()
    
    total_GR_NG=0
    total_NG_GR=0
    
   
    batch_nums = list(range(number_of_example))
    print (len(batch_nums))
    #print(TEXT.vocab.itos)   
    data_b =get_batches(batch_nums, train_iter_n, dev_iter_n,test_iterator=test_iter_n, dset='dev') 
    
    dev_iter_n.init_epoch()
    for ind in range(number_of_example-1): # 500
       answer = model(data_b[ind]) 

       prob=F.softmax(answer)
       text = data_b[ind].text.data[:, 0]
       words = [TEXT.vocab.itos[i] for i in text]
       words_string=" ".join(words)
 
    
       write_string=words_string+" "+"Predicted_label "+ str(LABELS.vocab.itos[torch.max(prob, 1)[1].data[0]])+" True_label "+ str(LABELS.vocab.itos[data_b[ind].label.data[0]])+ " Certainty "+str(abs(prob.data[0][0]-prob.data[0][1]))

       if (abs(prob.data[0][0]-prob.data[0][1])<0.1):
           print ("The system is not sure about the class")
           print (write_string)

       if (str(LABELS.vocab.itos[data_b[ind].label.data[0]])=="NG"):

            if str(LABELS.vocab.itos[data_b[ind].label.data[0]])!=str(LABELS.vocab.itos[torch.max(prob, 1)[1].data[0]]): 
                print("System is wrong on:")
                print(words_string)
                
                total_NG_GR+=1
       if (str(LABELS.vocab.itos[data_b[ind].label.data[0]])=="GR"):
            if str(LABELS.vocab.itos[data_b[ind].label.data[0]])!=str(LABELS.vocab.itos[torch.max(prob, 1)[1].data[0]]): 
                print("System is wrong on:")
                print(words_string)
                total_GR_NG+=1
        
       
       #print (write_string + "\n")
       f.write(write_string)
       f.write("\n")
    
    
    f.close()
    print ("NUMBER OF TEST BATCHES "+str(len(data_b)))
    print ("GRAMMATICAL Predicted UNGRAMMATICAL "+ str(total_GR_NG))
    print ("UNGRAMMATICAL Predicted GRAMMATICAL "+str(total_NG_GR))


                    
if __name__ == '__main__':
    args=get_args()
    predict_and_analyze(args.dataset_type, number_of_example=args.number_of_examples, \
                    train_adress=args.train_address, \
                    dev_adress=args.test_address, \
                    test_adress=args.train_address, \
                    analysis_log=args.analysis_file_address,\
                    model_adress=args.model_address)  #"pre-trained_models/grammaticals/15000-perfect_one_unsure.pt")
    
    