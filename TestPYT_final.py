
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 17:11:53 2018

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
 

def get_args():
    parser = ArgumentParser(description='PyTorch/torchtext NewDatset')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--d_embed', type=int, default=5) #300 128 20 10
    parser.add_argument('--d_hidden', type=int, default=5)  # 128 100
    parser.add_argument('--n_layers', type=int, default=1) # 100
    parser.add_argument('--dev_every', type=int, default=100) #100
    parser.add_argument('--save_every', type=int, default=100) # 1000
    parser.add_argument('--preserve-case', action='store_false', dest='lower')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='model_folder')
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
    parser.add_argument('--word_vectors', type=str, default=False ) #"glove.6B.300d" False
    parser.add_argument('--resume_snapshot', type=str, default='')
    
    parser.add_argument('--train_dataset', type=str, required=True)
    parser.add_argument('--test_dataset', type=str, required=True)
    
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

class LSTMSentiment(nn.Module):

    def __init__(self, config):
        super(LSTMSentiment, self).__init__()
        self.hidden_dim = config.d_hidden
        self.vocab_size = config.n_embed
        self.emb_dim = config.d_embed
        self.num_out = config.d_out
        self.batch_size = config.batch_size
        self.use_gpu = config.use_gpu
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
    try:  # load onto gpu
        model = torch.load(snapshot_file)
        print('loaded onto gpu...')
    except:  # load onto cpu
     model = torch.load(snapshot_file, map_location=lambda storage, loc: storage)
     print('loaded onto cpu...')
     return model


def main_loop_structure(dataset_type): #  "aaaa_train_dev", "SST.train_dev_test" split
  args = get_args()
  if args.gpu!=None:  torch.cuda.set_device(args.gpu)
# NEW DATASET
  if dataset_type=="aaaa_train_dev":
     TEXT = data.Field(tokenize=data.get_tokenizer('spacy'), sequential=True, init_token='<SOS>', eos_token='<EOS>',lower=True)
     LABELS = data.Field(sequential=False, unk_token=None)    
     train, dev, test = data.TabularDataset.splits(path='', format='tsv', train=args.train_dataset, validation=args.test_dataset, test=args.train_dataset, fields=[('text', TEXT), ('label', LABELS)])  #
     print ("Dataset_split done")  # out_train.tsv  out_dev.tsv
     TEXT.build_vocab(train,dev, test, max_size=500)
     LABELS.build_vocab(train)   
     print ("Unique tokens")
     print(TEXT.vocab.itos)

     train_iter, val_iter, train_retest_iter = data.BucketIterator.splits((train, dev,test), batch_size=50, device=args.gpu, sort_key=None, sort=None)
     print ("Iterators are set")
     print ("Custom dataset")   
     
  if dataset_type=="SST_train_dev_test": #This is to check if it works on a real life data, not used in this prohect
      print ("SST")
      TEXT=data.Field(lower=args.lower)
      LABELS=data.Field(sequential=False, unk_token=None)
      
      train, dev, test = datasets.SST.splits(TEXT, LABELS, fine_grained = False, train_subtrees = True, filter_pred=lambda ex: ex.label != 'neutral')
      
      TEXT.build_vocab(train,dev,test)
      
      if args.word_vectors:
       if os.path.isfile(args.vector_cache):
        TEXT.vocab.vectors = torch.load(args.vector_cache)
       else:
        TEXT.vocab.load_vectors(args.word_vectors)
        makedirs(os.path.dirname(args.vector_cache))
        torch.save(TEXT.vocab.vectors, args.vector_cache)
  
      LABELS.build_vocab(train)
      train_iter, val_iter, test_iter = data.BucketIterator.splits((train, dev, test), batch_size=args.batch_size, device=args.gpu)

  config = args
  config.use_gpu=True
  if args.gpu==None:
      config.use_gpu=False
      
  config.n_embed = len(TEXT.vocab) #TEXT.vocab
  config.d_out = len(LABELS.vocab) # LABELS.vocab
  config.n_cells = config.n_layers

  if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
  else:
    print ("Define model")
    model = LSTMSentiment(config)
    if args.word_vectors:
        print ("WW")
        model.embed.weight.data = TEXT.vocab.vectors
    #print("Pre Cuda Sent")
    if config.use_gpu:  model.cuda() 
    #print ("POST CUDA SENT")
       
    
  criterion = nn.CrossEntropyLoss()
  opt = O.Adam(model.parameters())

  iterations = 0
  start = time.time()
  best_dev_acc = -1
  train_iter.repeat = False
  header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
  dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
  makedirs(args.save_path)
  print(header)

  '''
  n_dev_correct, dev_loss = 0, 0
  model.eval(); val_iter.init_epoch()
  for dev_batch_idx, dev_batch in enumerate(val_iter):
      answer = model(dev_batch)
      # print (answer)
      # print (dev_batch)
      n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum() #labels
      dev_loss = criterion(answer, dev_batch.label) # labels
      dev_acc = 100. * n_dev_correct / len(dev)
      
  print ("PRE_TRAIN_DEV_ACCURACY "+str (dev_acc))
 '''


  all_break = False
  for epoch in range(args.epochs):
    if all_break:
        break
    train_iter.init_epoch()
    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(train_iter):
        # switch model to training mode, clear gradient accumulators
        model.train(); opt.zero_grad()

        iterations += 1

        # forward pass
        answer = model(batch)
        # if (batch_idx==1 or batch_idx==0):
        #  print ("Labels for the batch")
        #  print ("Batch index "+str(batch_idx))
        #  print (batch.label.data)

        # calculate accuracy of predictions in the current batch
        n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()  # labels
        n_total += batch.batch_size
        train_acc = 100. * n_correct/n_total

        # calculate loss of the network output with respect to training labels
        loss = criterion(answer, batch.label)
        loss.backward(); opt.step()
        
        

        # checkpoint model periodically
        if iterations % args.save_every == 0:

            snapshot_prefix = os.path.join(args.save_path, 'snapshot')
            snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, loss.data.item(), iterations) #[0]
            torch.save(model, snapshot_path)
            for f in glob.glob(snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)

        # evaluate performance on validation set periodically
          

            # switch model to evaluation mode
            model.eval(); val_iter.init_epoch()

            # calculate accuracy on validation set
            n_dev_correct, dev_loss = 0, 0
            for dev_batch_idx, dev_batch in enumerate(val_iter):
                 answer = model(dev_batch)
                 n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum() #labels
                 dev_loss = criterion(answer, dev_batch.label) # labels
            '''
            print ("SANITY CHECK")
            print (n_dev_correct.item())
            print (len(dev))
            print (n_dev_correct.item()/len(dev))
            print (100. * n_dev_correct.item() / len(dev))
            '''
            dev_acc = float(100. * n_dev_correct.item() / len(dev))
            #print (dev_acc)
            
            
            model.eval(); train_retest_iter.init_epoch()
            n_train_correct=0
            for  train_batch_idx, train_batch in enumerate (train_retest_iter):
                answer=model(train_batch)
                n_train_correct+= (torch.max(answer, 1)[1].view(train_batch.label.size()).data == train_batch.label.data).sum()
           # true_train_acc=float(100. * n_train_correct/len(test))
            true_train_acc = float(100. * n_train_correct.item() / len(test))
            
            '''
            print (true_train_acc)
            print ("SANITY CHECK 2")
            print (n_train_correct.item())
            print (len(test))
            print (n_train_correct.item()/len(test))
            print (100. * n_train_correct.item() / len(test))
            true_train_acc = float(100. * n_train_correct.item() / len(test))
            print (true_train_acc)
            '''


           # print(dev_log_template.format(time.time()-start,
           #     epoch, iterations, 1+batch_idx, len(train_iter),
           #     100. * (1+batch_idx) / len(train_iter), loss.data[0], dev_loss.data[0], train_acc, dev_acc))
           
            print(dev_log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.data[0], dev_loss.data[0], true_train_acc, dev_acc))          
           

            # update best valiation set accuracy
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
                snapshot_path = snapshot_prefix + '_devacc_{}_devloss_{}_iter_{}_model.pt'.format(dev_acc, dev_loss.data[0], iterations)

                # save model, delete previous 'best_snapshot' files
                torch.save(model, snapshot_path)
                #for f in glob.glob(snapshot_prefix + '*'):
                #    if f != snapshot_path:
                 #       os.remove(f)
  
if __name__ == '__main__':
    main_loop_structure("aaaa_train_dev")




