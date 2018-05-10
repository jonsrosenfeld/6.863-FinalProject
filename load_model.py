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
    parser.add_argument('--d_embed', type=int, default=5) #300 128 11
    parser.add_argument('--d_proj', type=int, default=300)
    parser.add_argument('--d_hidden', type=int, default=5)  # 128 100
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


'''
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
'''

class LSTMSentiment(nn.Module):

    def __init__(self, config):
        super(LSTMSentiment, self).__init__()
        self.hidden_dim = config.d_hidden
        self.vocab_size = config.n_embed
        self.emb_dim = config.d_embed
        self.num_out = config.d_out
        self.batch_size = config.batch_size
        self.use_gpu = True #config.use_gpu
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
    #try:  # load onto gpu
    #    model = torch.load(snapshot_file)
   #     print('loaded onto gpu...')
   # except:  # load onto cpu
    model = torch.load(snapshot_file, map_location=lambda storage, loc: storage)
    print('loaded onto cpu...')
    return model

def get_batches(batch_nums, train_iterator, dev_iterator,test_iterator=None,  dset='dev'):
    print('getting batches...')
    np.random.seed(13)
    random.seed(13)
    
    # pick data_iterator
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

        if num == max(batch_nums):
            break
        elif num == len(batch_nums):
            print('found them all')
            break
    return batches


def CD(batch, model, start, stop):
    weights = model.lstm.state_dict()

    # Index one = word vector (i) or hidden state (h), index two = gate
    W_ii, W_if, W_ig, W_io = np.split(weights['weight_ih_l0'], 4, 0)
    W_hi, W_hf, W_hg, W_ho = np.split(weights['weight_hh_l0'], 4, 0)
    b_i, b_f, b_g, b_o = np.split(weights['bias_ih_l0'].cpu().numpy() + weights['bias_hh_l0'].cpu().numpy(), 4)
    word_vecs = model.embed(batch.text)[:,0].data
    T = word_vecs.size(0)
    relevant = np.zeros((T, model.hidden_dim))
    irrelevant = np.zeros((T, model.hidden_dim))
    relevant_h = np.zeros((T, model.hidden_dim))
    irrelevant_h = np.zeros((T, model.hidden_dim))
    for i in range(T):
        if i > 0:
            prev_rel_h = relevant_h[i - 1]
            prev_irrel_h = irrelevant_h[i - 1]
        else:
            prev_rel_h = np.zeros(model.hidden_dim)
            prev_irrel_h = np.zeros(model.hidden_dim)

        rel_i = np.dot(W_hi, prev_rel_h)
        rel_g = np.dot(W_hg, prev_rel_h)
        rel_f = np.dot(W_hf, prev_rel_h)
        rel_o = np.dot(W_ho, prev_rel_h)
        irrel_i = np.dot(W_hi, prev_irrel_h)
        irrel_g = np.dot(W_hg, prev_irrel_h)
        irrel_f = np.dot(W_hf, prev_irrel_h)
        irrel_o = np.dot(W_ho, prev_irrel_h)

        if i >= start and i <= stop:
            rel_i = rel_i + np.dot(W_ii, word_vecs[i])
            rel_g = rel_g + np.dot(W_ig, word_vecs[i])
            rel_f = rel_f + np.dot(W_if, word_vecs[i])
            rel_o = rel_o + np.dot(W_io, word_vecs[i])
        else:
            irrel_i = irrel_i + np.dot(W_ii, word_vecs[i])
            irrel_g = irrel_g + np.dot(W_ig, word_vecs[i])
            irrel_f = irrel_f + np.dot(W_if, word_vecs[i])
            irrel_o = irrel_o + np.dot(W_io, word_vecs[i])

        rel_contrib_i, irrel_contrib_i, bias_contrib_i = decomp_three(rel_i, irrel_i, b_i, sigmoid)
        rel_contrib_g, irrel_contrib_g, bias_contrib_g = decomp_three(rel_g, irrel_g, b_g, np.tanh)

        relevant[i] = rel_contrib_i * (rel_contrib_g + bias_contrib_g) + bias_contrib_i * rel_contrib_g
        irrelevant[i] = irrel_contrib_i * (rel_contrib_g + irrel_contrib_g + bias_contrib_g) + (rel_contrib_i + bias_contrib_i) * irrel_contrib_g

        if i >= start and i < stop:
            relevant[i] += bias_contrib_i * bias_contrib_g
        else:
            irrelevant[i] += bias_contrib_i * bias_contrib_g

        if i > 0:
            rel_contrib_f, irrel_contrib_f, bias_contrib_f = decomp_three(rel_f, irrel_f, b_f, sigmoid)
            relevant[i] += (rel_contrib_f + bias_contrib_f) * relevant[i - 1]
            irrelevant[i] += (rel_contrib_f + irrel_contrib_f + bias_contrib_f) * irrelevant[i - 1] + irrel_contrib_f * relevant[i - 1]

        o = sigmoid(np.dot(W_io, word_vecs[i]) + np.dot(W_ho, prev_rel_h + prev_irrel_h) + b_o)
        rel_contrib_o, irrel_contrib_o, bias_contrib_o = decomp_three(rel_o, irrel_o, b_o, sigmoid)
        new_rel_h, new_irrel_h = decomp_tanh_two(relevant[i], irrelevant[i])
        #relevant_h[i] = new_rel_h * (rel_contrib_o + bias_contrib_o)
        #irrelevant_h[i] = new_rel_h * (irrel_contrib_o) + new_irrel_h * (rel_contrib_o + irrel_contrib_o + bias_contrib_o)
        relevant_h[i] = o * new_rel_h
        irrelevant_h[i] = o * new_irrel_h

    W_out = model.hidden_to_label.weight.data
    
    # Sanity check: scores + irrel_scores should equal the LSTM's output minus model.hidden_to_label.bias
    scores = np.dot(W_out, relevant_h[T - 1])
    irrel_scores = np.dot(W_out, irrelevant_h[T - 1])

    return scores, irrel_scores
    
def decomp_three(a, b, c, activation):
    a_contrib = 0.5 * (activation(a + c) - activation(c) + activation(a + b + c) - activation(b + c))
    b_contrib = 0.5 * (activation(b + c) - activation(c) + activation(a + b + c) - activation(a + c))
    return a_contrib, b_contrib, activation(c)

def decomp_tanh_two(a, b):
    return 0.5 * (np.tanh(a) + (np.tanh(a + b) - np.tanh(b))), 0.5 * (np.tanh(b) + (np.tanh(a + b) - np.tanh(a)))




def predict_and_analyze(dataset_type, train_adress="",dev_adress="", test_adress=""):
    args = get_args()
    #dataset_type="Custom_with_test"


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
     
       print (LABELS.vocab.itos)

       train_iter_n, dev_iter_n = data.BucketIterator.splits((train, dev), batch_size=1, sort_key=lambda x: len(x.text), device=None)
     
    if dataset_type=="Custom_with_test":
       TEXT = data.Field(tokenize=data.get_tokenizer('spacy'), sequential=True, init_token='<SOS>', eos_token='<EOS>',lower=True)
       LABELS = data.Field(sequential=False, unk_token=None)
     
       train, dev,test =data.TabularDataset.splits(path='', format='tsv', train=train_adress, validation=dev_adress, test=test_adress, fields=[('text', TEXT), ('label', LABELS)])  #
       print ("Dataset_split done")  # out_train.tsv  out_dev.tsv
     
       TEXT.build_vocab(train,dev,test, max_size=500)
       LABELS.build_vocab(train)
     
     # print (LABELS.vocab.itos)
     # LABELS.vocab.itos=['GR','UGR']
     # print (LABELS.vocab.itos)
    
       train_iter_n, dev_iter_n, test_iter_n = data.BucketIterator.splits((train, dev,test), batch_size=1, sort_key=lambda x: len(x.text), device=None)
     
    ""  
    model=get_model("models/best_snapshot_devacc_100.0_devloss_0.00842789746821_iter_13600_model.pt")
    model.use_gpu=False
    model.eval()
    f=open('predictions_explained_iter-13600.txt','a')
    dev_iter_n.init_epoch()
   
    batch_nums = list(range(2000)) #501 872 2000 60000 8000
    print (len(batch_nums))
    data_b =get_batches(batch_nums, train_iter_n, dev_iter_n,test_iterator=test_iter_n, dset='dev') 
    
    for ind in range(1999): # 500
       print ("START-EXAMPLE")
       print (ind)
       answer = model(data_b[ind]) 
     #print (answer)
       prob=F.softmax(answer)
       text = data_b[ind].text.data[:, 0]
       words = [TEXT.vocab.itos[i] for i in text]
       words_string=" ".join(words)
 
       #tens=prob.data.numpy()
       
    
       if (abs(prob.data[0][0]-prob.data[0][1])<0.1):
           print ("The system is not sure about the class")

       print ("-------------------------------")
       write_string=words_string+" "+"Predicted_label "+ str(LABELS.vocab.itos[torch.max(prob, 1)[1].data[0]])+" True_label "+ str(LABELS.vocab.itos[data_b[ind].label.data[0]])+ " Certainty "+str(abs(prob.data[0][0]-prob.data[0][1]))
       print (write_string + "\n")
       f.write(write_string)
       f.write("\n")
    
    print (LABELS.vocab.itos)
    f.close()
    print ("NUMBER OF TEST BATCHES "+str(len(data_b)))



#predict_and_analyze("Custom_with_test",train_adress="train_g2.tsv",dev_adress="dev_g2.tsv", test_adress="train_g2.tsv")
predict_and_analyze("Custom_with_test",train_adress="grammar2_train.tsv",dev_adress="grammar2_dev.tsv", test_adress="grammar2_train.tsv")

'''
for ind in range(7999):
    text = data_b[ind].text.data[:, 0]
    words = [TEXT.vocab.itos[i] for i in text]
    #if words==['<SOS>','a','a','b','b', '<EOS>']:
    if words==['<SOS>', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a','a', 'a', 'a', 'a', 'b', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', '<EOS>'] :
        high_level_comp_ind = ind
        print (high_level_comp_ind)
        print (words)
        break
    
pos, pos_irrel = CD(data_b[high_level_comp_ind], model, start = 0, stop = 35)
print(' '.join(words[:35]), pos[0] - pos[1])
neg, neg_irrel = CD(data_b[high_level_comp_ind], model, start = 36, stop = len(words)-1)
print(' '.join(words[35:]), neg[0] - neg[1])
'''    
    
    