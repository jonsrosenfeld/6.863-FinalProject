#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:13:58 2018

@author: elena
"""
from __future__ import division
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
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data.dataset import Dataset
import itertools
from sklearn import preprocessing

def get_args():
    parser = ArgumentParser(description='PyTorch/torchtext NewDatset')
    parser.add_argument('--epochs', type=int, default=700)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--d_hidden', type=int, default=200)  # 128 100
    parser.add_argument('--n_layers', type=int, default=1) # 100
    parser.add_argument('--dev_every', type=int, default=100) #100
    parser.add_argument('--save_every', type=int, default=100) # 1000
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--save_path', type=str, default='model_folder')
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
    parser.add_argument('--word_vectors', type=str, default=False ) #"glove.6B.300d" False
    parser.add_argument('--resume_snapshot', type=str, default='')   
    #parser.add_argument('--train_dataset', type=str, required=True)
   # parser.add_argument('--test_dataset', type=str, required=True)
    
    args = parser.parse_args()
    return args



def makedirs(name):
    # 
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

class MyLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super(MyLabelBinarizer,self).transform([y])
        if self.y_type_ == 'binary':
            #print ('b')
            ch_st=np.hstack((Y, 1-Y)) 
            ch_st=np.flip(ch_st,1)
            return ch_st

        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)
# https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/7        
def pad_tensor(vec, pad, dim,use_gpu=1):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    
    #print (vec.type())
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
   # print (torch.zeros(*pad_size).type())
    if use_gpu==-1: return torch.cat([vec, torch.zeros(*pad_size).type(torch.FloatTensor)], dim=dim)
    else: return torch.cat([vec, torch.zeros(*pad_size).type(torch.FloatTensor).cuda()], dim=dim)

class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        #print (batch[0])
        #print (len(batch))
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        #print (max_len)
        # pad according to max_len
        batch = map(lambda (x, y):
                    (pad_tensor(x, pad=max_len, dim=self.dim,), y), batch)
        # stack all
        xs = torch.stack(map(lambda x: x[0], batch), dim=1)
        ys = torch.stack(map(lambda x: x[1], batch),dim=0)
        # print (xs.shape)
        # print (ys.shape)
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)

def binarize(list_of_sequences,binarizer="dataset"):
          if binarizer=="dataset" :label_binarizer = MyLabelBinarizer()
          if binarizer=="labels": label_binarizer = preprocessing.LabelBinarizer()
          
          
          if binarizer=="dataset" : label_to_index={'a':0,'b':1}
          if binarizer=="labels": label_to_index={'GR':0,'UGR':1}    
          
    
    
          if binarizer=="dataset" : label_binarizer.fit(label_to_index.values())
          if binarizer=="labels": label_binarizer.fit(label_to_index.values())
          
          #print (label_binarizer)
          
          list_of_numpy_objects=[]
          for sequence in list_of_sequences:
             # print (sequence)
              
              if binarizer=="dataset":
                string=[label_to_index[token] for token in sequence]
                transformed_sequence=[]
                for character in string:
                  # print(repr(character))
                  # print(label_binarizer.transform(character))
                  transformed_sequence.append(label_binarizer.transform(character))
                list_of_numpy_objects.append(transformed_sequence)
              if binarizer=="labels":
                  sequence=label_to_index[sequence]
                  
                 #  print (label_binarizer.transform(sequence))
                  list_of_numpy_objects.append(label_binarizer.transform([sequence]))
         # print (len(list_of_numpy_objects))
          #print (list_of_numpy_objects[-1])
          return list_of_numpy_objects
      
def read_tsv_to_list_of_sequences(address):
    f=open(address,'r')
    list_of_sequences=[]
    list_of_labels=[]
    for line in f:
        sequence_labels=line.split("\t")
        sequence=sequence_labels[0].split(" ")
        list_of_sequences.append(sequence)
        list_of_labels.append(sequence_labels[-1].strip())
    return list_of_sequences, list_of_labels        

class CustomDataset(Dataset):
    def __init__(self, datasets_paths,use_gpu=-1):
        l_s,l_l=read_tsv_to_list_of_sequences(datasets_paths[0])
        numpy_examples_list=binarize(l_s)       
        numpy_examples_list=[np.vstack(i) for i in numpy_examples_list]
        if use_gpu==-1: self.tensors=[torch.from_numpy(array).type(torch.FloatTensor) for array in numpy_examples_list]
        else: self.tensors=[torch.from_numpy(array).type(torch.FloatTensor).cuda() for array in numpy_examples_list]
       # print(len(self.tensors))
        print (self.tensors[0].shape)
        #print(self.tensors[0].type())
        
        numpy_labels_list=binarize(l_l,binarizer="labels")
       # print (numpy_labels_list[0].shape)
        if use_gpu==-1: self.tensor_labels=[torch.squeeze(torch.from_numpy(np.flip(numpy_labels_list[i],axis=0).copy()).type(torch.LongTensor)) for i,el in enumerate(numpy_labels_list)]
        else: self.tensor_labels=[torch.squeeze(torch.from_numpy(np.flip(numpy_labels_list[i],axis=0).copy()).type(torch.LongTensor).cuda()) for i,el in enumerate(numpy_labels_list)]
       # self.tensor_labels= torch.squeeze(self.tensor_labels)
       # print (len(numpy_labels_list))
        
       # print(numpy_labels_list)
       # print(self.tensor_labels)

    def __getitem__(self, index): 
         single_label = self.tensor_labels[index]
         single_string=self.tensors[index]
         return (single_string, single_label)
         ""
    def __len__(self):
        return len(self.tensors)
    
    

class LSTM_vanila(nn.Module):
   def __init__(self, config):
        super(LSTM_vanila, self).__init__()
        self.use_embeddings=False
        
        self.hidden_dim = config.d_hidden
        self.num_out = config.d_out
        self.batch_size = config.batch_size
        self.use_gpu = config.use_gpu
        self.num_labels = 2
        
        self.representation_dimension = 2

        self.lstm = nn.LSTM(input_size = self.representation_dimension, hidden_size = self.hidden_dim)
        self.hidden_to_label = nn.Linear(self.hidden_dim, self.num_labels)
        
        print (self.use_gpu)
        self.test=0
   def forward(self, batch):

       
       
       
       
        if self.use_embeddings==True:
          if self.use_gpu:
           # print ("GPU")
            self.hidden = (Variable(torch.zeros(1, batch.text.size()[1], self.hidden_dim).cuda()),
                            Variable(torch.zeros(1, batch.text.size()[1], self.hidden_dim).cuda()))
          else:
            #print ("CPU")
            self.hidden = (Variable(torch.zeros(1, batch.text.size()[1], self.hidden_dim)),
                            Variable(torch.zeros(1, batch.text.size()[1], self.hidden_dim)))
          
          lstm_out, self.hidden = self.lstm(batch[0], self.hidden)
          logits = self.hidden_to_label(lstm_out[-1])
          return logits
        else:
            
            # print ("THERE")
           #  print (batch)
           
             if self.use_gpu:
                # print ("GPU")
                 self.hidden =(Variable(torch.zeros(1, len(batch[0]), self.hidden_dim).cuda()),
                               Variable(torch.zeros(1, len(batch[0]), self.hidden_dim).cuda()))
             else:
                 self.hidden =(Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                               Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))
             lstm_out, self.hidden = self.lstm(batch, self.hidden)
             logits = self.hidden_to_label(lstm_out[-1])
             
             
             
             if self.test==0:
             #  print (batch.size())
              # print (batch)
               self.test=1
               #print (self.hidden[0].shape)
             
             return logits
  
   
    
args = get_args()    
torch.cuda.set_device(1)  
D=CustomDataset(['new_train-15000.tsv'],use_gpu=1) 
mn_dataset_loader = torch.utils.data.DataLoader(dataset=D,collate_fn=PadCollate(dim=0),
                                                    batch_size=args.batch_size,
                                                    shuffle=False)   
T=CustomDataset(['new_dev.tsv'],use_gpu=1)
test_dataset_loader= torch.utils.data.DataLoader(dataset=D,collate_fn=PadCollate(dim=0),
                                                    batch_size=args.batch_size,
                                                    shuffle=False)  
#new_data_loader= torch.utils.data.DataLoader(dataset=D,collate_fn=PadCollate(dim=0),
#                                                    batch_size=args.batch_size,
#                                                shuffle=False)  


config = args
config.d_out = 2
config.use_gpu=True
model = LSTM_vanila(config)
model.cuda()

criterion = nn.CrossEntropyLoss()
opt = O.Adam(model.parameters())



#opt=O.SGD(model.parameters(),lr=0.1)


m = nn.Softmax()


header = '  Time Epoch Iteration Progress    (%Epoch)  Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:12.4f},{:12.4f}'.split(','))
print (header)
makedirs(args.save_path)

general_batch_index=0



best_dev_acc=0


start = time.time()
for epoch in range(args.epochs):
    #print (epoch)
    cum_loss=0.
    
    correct = 0
    total = 0
    
    for batch_idx, (example, target) in enumerate(mn_dataset_loader):
#        print (example)
#        print (target)
#        print (batch_idx)
         general_batch_index+=1
         model.train(); opt.zero_grad()
        # print (example)
         answer = model(example)
         #print (answer)
         loss = criterion(answer, target)
         loss.backward(); opt.step()
         
         
         _, predicted = torch.max(answer, 1)
         total += target.size(0)
         correct += (predicted == target).sum().item()
         
  
         
         if general_batch_index % 100 ==0:
             correct_dev=0
             total_dev=0
             model.eval();
             for  batch_idx_2, (test_example,test_target) in enumerate(test_dataset_loader):
                  answer = model(test_example)
                  _, predicted = torch.max(answer, 1)
                  total_dev += test_target.size(0)
                  correct_dev += (predicted == test_target).sum().item()
        
             dev_acc=100 * correct_dev / total_dev
             #print ("Batches seen " +str(general_batch_index)+" "+str(loss))
             #print ('Accuracy of the network on train: %d %%' % (100 * correct / total))
             #print ('Accuracy of the network on dev: %d %%' % (100 * correct_dev / total_dev))
             
             
             print(dev_log_template.format(time.time()-start,
                epoch, general_batch_index, 1+batch_idx, len(mn_dataset_loader),
                100. * (1+batch_idx) / len(mn_dataset_loader), 100 * correct / total, dev_acc))       
             
             
             if dev_acc > best_dev_acc:
                 best_dev_acc = dev_acc
                 snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
                 snapshot_path = snapshot_prefix + '_devacc_{}_iter_{}_model.pt'.format(dev_acc, general_batch_index)
                 torch.save(model, snapshot_path)
                 ""
snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
snapshot_path=snapshot_prefix +"final_convergence"
torch.save(model, snapshot_path)
    
