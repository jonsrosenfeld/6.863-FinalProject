#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 17:11:53 2018

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


def space_sep_to_space_sep(file_input,file_output):
    f=open(file_input, 'r')
    f_o=open(file_output, 'a')
    
    for line in f :
       print (repr(line))
       set_line=line.strip("\n").split(" ")
       line2=set_line[:-1]
       print (line2)
       end_line = ' '.join(str(e) for e in line2)
       end_line=end_line+"\t"+set_line[-1]+"\n"
       print (end_line)
       f_o.write(end_line)
       
#space_sep_to_space_sep("train_g2.txt","train_g2.tsv")
#space_sep_to_space_sep("dev_g2.txt","dev_g2.tsv")


def artifically_clued_dataset(file_input,file_output):
    f=open(file_input, 'r')
    f_o=open(file_output, 'a')
    for line in f :
         print (repr(line))
         set_line=line.strip("\n").split(" ")
         line2=set_line[:-1]
         print (line2)
         end_line = ' '.join(str(e) for e in line2)
         if set_line[-1]=="GR":
              end_line=end_line+" pos_clue "
         else:
              end_line=end_line+" negative_clue "
         end_line=end_line+"\t"+set_line[-1]+"\n"
         print (end_line)
         f_o.write(end_line)
         
def delete_longer_then(n,file_input,file_output):
        f=open(file_input, 'r')
        f_o=open(file_output, 'a')
        for line in f :
            set_line=line.strip("\n").split(" ")
            if len(set_line)>n:
                continue
            else:
                f_o.write(line)
                
            
    
         

#artifically_clued_dataset("dev.txt","dev_sanity.tsv")       
       

def get_args():
    parser = ArgumentParser(description='PyTorch/torchtext SST')
    parser.add_argument('--epochs', type=int, default=10)
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




#space_sep_to_space_sep("super_small_train","out_sst.tsv")
#space_sep_to_space_sep("super_small_dev","out_ssd.tsv")

#pace_sep_to_space_sep("train.txt","out_train.tsv")

#space_sep_to_space_sep("train_1000.txt","out_test.tsv")
#space_sep_to_space_sep("out_test.tsv","out_test.tsv")
#space_sep_to_space_sep("toxicity_annotated_comments.tsv","")



args = get_args()


'''

inputs = data.Field(lower=args.lower)
answers = data.Field(sequential=False, unk_token=None)

train, dev, test = datasets.SST.splits(inputs, answers, fine_grained = False, train_subtrees = True,
										filter_pred=lambda ex: ex.label != 'neutral')


inputs.build_vocab(train, dev, test)
answers.build_vocab(train)

'''



# NEW DATASET
'''
TEXT = data.Field(tokenize=data.get_tokenizer('spacy'), sequential=True, init_token='<SOS>', eos_token='<EOS>',lower=True)
LABELS = data.Field(sequential=False, unk_token=None)

train, dev =data.TabularDataset.splits(path='', format='tsv', train="out_train.tsv", validation="out_dev.tsv", fields=[('text', TEXT), ('label', LABELS)])  #
print ("Dataset_split done")  # out_train.tsv  out_dev.tsv

TEXT.build_vocab(train,dev, max_size=500)
LABELS.build_vocab(train.label)


train_iter, val_iter = data.BucketIterator.splits((train, dev), batch_size=50, sort_key=lambda x: len(x.text), device=0)
print ("Iterators are set")
'''
#TEXT=inputs
#LABELS=answers

'''
config = args
config.n_embed = len(TEXT.vocab) #TEXT.vocab
config.d_out = len(LABELS.vocab) # LABELS.vocab
config.n_cells = config.n_layers

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
else:
    model = LSTMSentiment(config)
    if args.word_vectors:
        model.embed.weight.data = TEXT.vocab.vectors # TEXT
'''

# train_iter, val_iter, test_iter = data.BucketIterator.splits((train, dev, test), batch_size=args.batch_size, device=args.gpu)

def main_loop_structure(dataset_type): #  "aaaa_train_dev", "SST.train_dev_test" split
  args = get_args()
  '''
  inputs = data.Field(lower=args.lower)
  answers = data.Field(sequential=False, unk_token=None)

  train, dev, test = datasets.SST.splits(inputs, answers, fine_grained = False, train_subtrees = True, filter_pred=lambda ex: ex.label != 'neutral')
  inputs.build_vocab(train, dev, test)
  answers.build_vocab(train)
  '''
# NEW DATASET
  if dataset_type=="aaaa_train_dev":
     TEXT = data.Field(tokenize=data.get_tokenizer('spacy'), sequential=True, init_token='<SOS>', eos_token='<EOS>',lower=True)
     LABELS = data.Field(sequential=False, unk_token=None)
     
     train, dev =data.TabularDataset.splits(path='', format='tsv', train="train_g2.tsv", validation="dev_g2.tsv", fields=[('text', TEXT), ('label', LABELS)])  #
     print ("Dataset_split done")  # out_train.tsv  out_dev.tsv

     TEXT.build_vocab(train,dev, max_size=500)
     LABELS.build_vocab(train)


     train_iter, val_iter = data.BucketIterator.splits((train, dev), batch_size=50, sort_key=lambda x: len(x.text), device=0)
     print ("Iterators are set")
     
     
  if dataset_type=="SST_train_dev_test":
      print ("SST")
      TEXT=data.Field(lower=args.lower)
      LABELS=data.Field(sequential=False, unk_token=None)
      
      train, dev, test = datasets.SST.splits(TEXT, LABELS, fine_grained = False, train_subtrees = True, filter_pred=lambda ex: ex.label != 'neutral')
      
      TEXT.build_vocab(train,dev,test)
      
      if args.word_vectors:
       print ("WW")
       if os.path.isfile(args.vector_cache):
        print ("1")
        TEXT.vocab.vectors = torch.load(args.vector_cache)
       else:
        print ("2")
        TEXT.vocab.load_vectors(args.word_vectors)
        makedirs(os.path.dirname(args.vector_cache))
        torch.save(TEXT.vocab.vectors, args.vector_cache)
            
      
      
      LABELS.build_vocab(train)
      
      
      train_iter, val_iter, test_iter = data.BucketIterator.splits((train, dev, test), batch_size=args.batch_size, device=args.gpu)
#TEXT=inputs
#LABELS=answers


  config = args
  config.n_embed = len(TEXT.vocab) #TEXT.vocab
  config.d_out = len(LABELS.vocab) # LABELS.vocab
  config.n_cells = config.n_layers

  if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
  else:
    model = LSTMSentiment(config)
    if args.word_vectors:
        print ("WW")
        model.embed.weight.data = TEXT.vocab.vectors    
    
    
    
    
    
  criterion = nn.CrossEntropyLoss()
  opt = O.Adam(model.parameters())

  iterations = 0
  start = time.time()
  best_dev_acc = -1
  train_iter.repeat = False
  header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
  dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
  log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
  makedirs(args.save_path)
  print(header)



  all_break = False
  for epoch in range(args.epochs):
    if all_break:
        break
    train_iter.init_epoch()
    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(train_iter):
        #print (batch)
       # print (batch.label)
        # switch model to training mode, clear gradient accumulators
        model.train(); opt.zero_grad()

        iterations += 1

        # forward pass
        answer = model(batch)

        # calculate accuracy of predictions in the current batch
        n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()  # labels
        n_total += batch.batch_size
        train_acc = 100. * n_correct/n_total

        # calculate loss of the network output with respect to training labels
        loss = criterion(answer, batch.label)

        # backpropagate and update optimizer learning rate
        loss.backward(); opt.step()

        # checkpoint model periodically
        if iterations % args.save_every == 0:

            snapshot_prefix = os.path.join(args.save_path, 'snapshot')
            snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, loss.data[0], iterations)
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
                # print (answer)
                # print (dev_batch)
                 n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum() #labels
                 dev_loss = criterion(answer, dev_batch.label) # labels
            dev_acc = 100. * n_dev_correct / len(dev)

            print(dev_log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.data[0], dev_loss.data[0], train_acc, dev_acc))

            # update best valiation set accuracy
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
                snapshot_path = snapshot_prefix + '_devacc_{}_devloss_{}_iter_{}_model.pt'.format(dev_acc, dev_loss.data[0], iterations)

                # save model, delete previous 'best_snapshot' files
                torch.save(model, snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

        elif iterations % args.log_every == 0:
            # print progress message
            print(log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
            100. * (1+batch_idx) / len(train_iter), loss.data[0], ' '*8, n_correct/n_total*100, ' '*12))
  

  f=open('results_with_interpretation_sem2.txt','a')
  model=get_model("results-84/best_snapshot_devacc_84.51834862385321_devloss_0.5902754068374634_iter_4300_model.pt")
  model.eval()
  val_iter.init_epoch()
  batch_nums = list(range(2000)) #501  872
  print (len(batch_nums))
  if dataset_type=="SST_train_dev_test" : train_iter, val_iter, test_iter = data.BucketIterator.splits((train, dev, test), batch_size=1, sort_key=lambda x: len(x.text), device=0)  
  if dataset_type=="aaaa_train_dev" : train_iter, val_iter = data.BucketIterator.splits((train, dev), batch_size=1, sort_key=lambda x: len(x.text), device=0)
  
  data_b =get_batches(batch_nums, train_iter, val_iter) 

  for ind in range(1999): # 500
    print ("START-EXAMPLE")
    answer = model(data_b[ind]) 
    prob=F.softmax(answer)
    text = data_b[ind].text.data[:, 0]
    words = [TEXT.vocab.itos[i] for i in text]
    words_string=" ".join(words)
    #print (words)
    #print ("Soft-maxed predictions")
    #print (prob)
    
    #tens=prob.data.numpy()
   # print (tens.shape)
    if (abs(prob.data[0][0]-prob.data[0][1])<0.1):
        print ("The system is not sure about the class")
    #print ("Non-normalized_logits")
    #print (answer)
    
    #print ("Predicted_label "+ str(torch.max(prob, 1)[1])+" True label "+ str(data_b[ind].label.data[0]))
    #print (torch.max(answer, 1))
   # print ("True Label")
    #print (data[ind].label.data[0])
    print ("-------------------------------")
    write_string=words_string+" "+"Predicted_label "+ str(torch.max(prob, 1)[1].data[0])+" True label "+ str(data_b[ind].label.data[0])+ " Certainty "+str(abs(prob.data[0][0]-prob.data[0][1]))
    f.write(write_string)
    print (write_string+"\n")
    f.write("\n")
    
  print ("NUMBER OF TEST BATCHES-TEST "+str(len(data_b)))    

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


main_loop_structure("aaaa_train_dev")


'''
train_iter_n, dev_iter_n = data.BucketIterator.splits((train, dev), batch_size=1, sort_key=lambda x: len(x.text), device=0)
model.eval()
dev_iter_n.init_epoch()
batch_nums = list(range(1821)) #501
print (len(batch_nums))
data =get_batches(batch_nums, train_iter_n, dev_iter_n) 
for ind in range(1820): # 500
    answer = model(data[ind]) 
    prob=F.softmax(answer)
    text = data[ind].text.data[:, 0]
    words = [TEXT.vocab.itos[i] for i in text]
    print ("-------------------------------")
    print (words)
    print ("Predictions")
    print (prob)
    
    #print (torch.max(answer, 1))
    print ("True Label")
    print (data[ind].label.data[0])
    print ("-------------------------------")
'''

'''

#print(len(TEXT.vocab.freqs.most_common()))
#print(LABELS.vocab.itos)


#for test in range (0,3):
 # batch = next(iter(train_iter))
 # text = data[0].text.data[:, 0]
  #words = [inputs.vocab.itos[i] for i in text]
#  print(batch.text)
#  print(batch.labels)
  
#bn = list(range(0,10))
#print (bn)
#batch_nums = list(range(6920))
#get_batches(batch_nums, train_iterator=train_iter, dev_iterator=val_iter, dset='train')

'''





