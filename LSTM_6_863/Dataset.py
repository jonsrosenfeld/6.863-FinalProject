#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 22:31:26 2018

@author: elena
"""

# dataset = ds.Dataset()
# z=hd.get_valid_dataset_filepaths({'dataset_text_folder':folder_adress})
#dataset.load_dataset(z)
import codecs
import os
import collections
import numpy as np
import operator
import Helpers as hlp
import sklearn.preprocessing



from sklearn.preprocessing import LabelBinarizer


#https://stackoverflow.com/questions/31947140/sklearn-labelbinarizer-returns-vector-when-there-are-2-classes

class MyLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            ch_st=np.hstack((Y, 1-Y)) 
            ch_st=np.flip(ch_st,1)
            # print ("TRANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN")
            # print (ch_st.shape)
            # print (ch_st)
            return ch_st
            #return np.hstack((Y, 1-Y)) #

        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)

class Dataset(object):
        def __init__(self, dataset, name='',dataset_log_adress='',use_features=False):
         self.name = name
         self.use_features=use_features
         self.dataset=dataset
        
        def _parse_dataset_fake_language(self,dataset_filepath,dataset_type):
            all_lowercase=True
            token_count = collections.defaultdict(lambda: 0)
            label_count = collections.defaultdict(lambda: 0)
            
            max_length=0
            max_sentence=""
            
            tokens=[]
            labels=[]
            if dataset_filepath :
               f = codecs.open(dataset_filepath, 'r', 'UTF-8')  
               for idx,line in enumerate(f):
                   line=line.strip().split(' ')
                   #print (line)
                   new_label=[]
                   new_token_sequence=[]
                   words=line[:-1]
                  # print (words)

                 
               
                   for word in words:
                           token_count[word]+=1
                           
                   if max_length<len(words):
                           max_length=len(words)
                           max_sentence=words
                    
                   label_count[line[-1]] += 1 
                   print (line[-1])
                   new_label.append(line[-1])
                   new_token_sequence.extend(words)
                   
                   
                   labels.append(new_label)
                   tokens.append(new_token_sequence)

          #  print (max_length)
          #  print (max_sentence)
            return labels,tokens,max_length,token_count,label_count            
         
         
        def _parse_dataset_semeval_twitter(self,dataset_filepath, dataset_type):
            all_lowercase=True
            token_count = collections.defaultdict(lambda: 0)
            label_count = collections.defaultdict(lambda: 0)
            
            max_length=0
            max_sentence=""
            
            
            tokens=[]
            labels=[]
           # new_token_sequence=[]
           # new_label = []
            if dataset_filepath :
               f = codecs.open(dataset_filepath, 'r', 'UTF-8')  
               for idx,line in enumerate(f):
                   line=line.strip().split('\t')
                  # print (line)
                   new_label=[]
                   new_token_sequence=[]
                   words=(line[2].strip()).lower().split(" ")
               
                   for word in words:
                           token_count[word]+=1
                           
                   if max_length<len(words):
                           max_length=len(words)
                           max_sentence=words
                    
                   label_count[line[1]] += 1 
                   new_label.append(line[1])
                   new_token_sequence.extend(words)
                   
                   
                   labels.append(new_label)
                   tokens.append(new_token_sequence)

           # print (max_length)
           # print (max_sentence)
            return labels,tokens,max_length,token_count,label_count
        def _parse_dataset(self, dataset_filepath,dataset_type):
            
            all_lowercase=True
            
            
            token_count = collections.defaultdict(lambda: 0) #initialized by a function
            label_count = collections.defaultdict(lambda: 0)
            
            tokens=[]
            labels=[]
            
            
            new_token_sequence=[]
            new_label = []
            
            max_length=0
            max_sentence=""
            
            if dataset_filepath :
               f = codecs.open(dataset_filepath, 'r', 'UTF-8')  
               for idx,line in enumerate(f):
                   line = line.strip().split('\t')
                   if line[0]=="": 
                       new_label=[]
                       new_token_sequence=[]
                       continue
                   if len(line)==1:
                       
                      # print ("NEW_TEXT")
                       ""
                       
                       
                   else:
                       new_label=[]
                       new_token_sequence=[]
                       
                       new_label.append(line[0])
                       words=line[1].strip().split(" ")
                       if all_lowercase==True:
                           words=(line[1].strip()).lower().split(" ")
                       
                       if max_length<len(words):
                           max_length=len(words)
                           max_sentence=words
                        
                        
                       label_count[line[0]] += 1 
                        
                       for word in words:
                           token_count[word]+=1
                       new_token_sequence.extend(words)
                       
                       labels.append(new_label)
                       tokens.append(new_token_sequence)
                      # labels.append(line[0])
          #  print (labels[0])        
         #  print (labels[-1])
          #  print (tokens[0])
           # print (tokens[-1])
            
          #  print (max_length)
           # print (max_sentence)
            return labels,tokens,max_length,token_count,label_count

              
                            
        def load_dataset(self, dataset_filepaths):
            print ("Loading the dataset")
            remap_to_unk_count_threshold = 0
            self.UNK_TOKEN_INDEX = 0
            self.UNK = 'UNK'
            self.unique_labels = []
            
            self.PAD="PADD_TK"
            self.TRIMM_LENGTH=100  #padding length
            self.tokens_mapped_to_unk = []
            
            labels={}
            tokens={}
            
            max_lengths={}
            token_counts={}
            label_counts={}
            
            parameters={}
            parameters['token_pretrained_embedding_filepath']='./glove.6B.100d.txt'
           # all_pretrained_tokens = hlp.load_tokens_from_pretrained_token_embeddings(parameters)
        
            if self.dataset=="PUBMED":
             print ("PUBMED SENTENCE CLASSIFICATION")   
             for dataset_type in ['train']:
                labels[dataset_type], tokens[dataset_type],max_lengths[dataset_type],token_counts[dataset_type],label_counts[dataset_type]=self._parse_dataset(dataset_filepaths.get(dataset_type, None),dataset_type)
            if self.dataset=="TWITTER3":
             print ("TWITTER SENTIMENT CLASSIFICATION, 3 point scale")
             for dataset_type in ['train','dev']:
                 labels[dataset_type], tokens[dataset_type],max_lengths[dataset_type],token_counts[dataset_type],label_counts[dataset_type]=self._parse_dataset_semeval_twitter(dataset_filepaths.get(dataset_type, None), dataset_type)
            if self.dataset=="GRAMMAR":
                print ("GRAMMAR INDUCTION TEST")
            for dataset_type in ['train','dev']:
                 labels[dataset_type], tokens[dataset_type],max_lengths[dataset_type],token_counts[dataset_type],label_counts[dataset_type]=self._parse_dataset_fake_language(dataset_filepaths.get(dataset_type, None), dataset_type)                
             
            # TRIMM of linger then 90
   
            
            Num_long_sent=0
            Num_of_PADD_tokens={}
            Num_of_PADD_tokens['train']=0
            Num_of_PADD_tokens['dev']=0 
            
            
            for subdataset in ['train','dev']:
              for sequence in tokens[subdataset]:
                    padd=self.TRIMM_LENGTH-len(sequence)                 
                    if len(sequence)>self.TRIMM_LENGTH:  # 180.040 sentences less then 1/1800 longer then 90,  15 percent less then 50
                        Num_long_sent+=1
                        sequence=sequence[0:self.TRIMM_LENGTH]
                    else:
                        Num_of_PADD_tokens[subdataset]+=padd
                        sequence.extend([self.PAD] * (padd))
            
            
            
            
            token_counts['all'] = {}
            for token in list(token_counts['train'].keys()) + list(token_counts['dev'].keys()):
              token_counts['all'][token] = token_counts['train'][token]+token_counts['dev'][token]

            token_counts['train']['PADD_TK']=Num_of_PADD_tokens['train']
            token_counts['dev']['PADD_TK']=Num_of_PADD_tokens['dev']
            token_counts['all']['PADD_TK']=Num_of_PADD_tokens['train']+Num_of_PADD_tokens['dev']
            
            label_counts['all'] = {}  
            
            for label in list(label_counts['train'].keys())+list(label_counts['dev'].keys()):
                label_counts['all'][label] = label_counts['train'][label]+label_counts['dev'][label]
                       # print (sequence)
           # print (tokens['train'][0])
                        
          #  print ("Number of sentences "+str(len(tokens['train'])))
           # print ("Number of long  sentences "+str(Num_long_sent))
                    
            token_counts['all'][self.PAD]=Num_of_PADD_tokens
            #print ("PADD COUNT")
            #print (token_counts['train'][self.PAD])
            
            token_counts['all'] = hlp.order_dictionary(token_counts['train'], 'value_key', reverse = True)
            label_counts['all'] = hlp.order_dictionary(label_counts['all'], 'key', reverse = False)

            
            
                    #print ("NEED additional "+str(padd))
                   # sequence.extend([self.PAD] * (padd))

                   # if padd<200:
                    #  Num_long_sent+=1
           # print (Num_long_sent)
            
            
            token_to_index = {}
            label_to_index = {}
            
 #           label_to_index = hlp.order_dictionary(label_to_index, 'value', reverse = False)
 #           print ("________________________________________")
 #           print (label_to_index)
 #           index_to_label = hlp.reverse_dictionary(label_to_index)     
            
            
            token_to_index[self.UNK] = self.UNK_TOKEN_INDEX
            
            iteration_number = 0
            number_of_unknown_tokens = 0
            
            for token, count in token_counts['all'].items() :
               
               if iteration_number == self.UNK_TOKEN_INDEX: iteration_number += 1  
               if (token_counts['train'][token] == 0): # and (not hlp.is_token_in_pretrained_embeddings(token, all_pretrained_tokens, parameters)) :
                 token_to_index[token] =  self.UNK_TOKEN_INDEX
                 number_of_unknown_tokens += 1
                 self.tokens_mapped_to_unk.append(token)
               else:
                token_to_index[token] = iteration_number  #ORDERED BY  frequency
                iteration_number += 1
                
            
            
            infrequent_token_indices = []
            for token, count in token_counts['train'].items():
               if 0 < count <= remap_to_unk_count_threshold:
                infrequent_token_indices.append(token_to_index[token])
            
           # print (len(infrequent_token_indices))
            
            
           # print (token_to_index)
            
           
           
            token_indices={}
            label_indices ={}
            
            
            iteration_number = 0
            for label, count in label_counts['train'].items():
                label_to_index[label] = iteration_number
                iteration_number += 1
                self.unique_labels.append(label)   
            
            for dataset_type in ['train','dev']: 
              token_indices[dataset_type] = []
              for idx,token_sequence in enumerate(tokens[dataset_type]):
                   # print (token_sequence)
                    token_indices[dataset_type].append([token_to_index[token] for token in token_sequence])
                    
              label_indices[dataset_type] = [] 
              
              for label_sequence in labels[dataset_type]:
                # print (label_sequence)
                 label_indices[dataset_type].append([label_to_index[label] for label in label_sequence])  
             
           # print ("JKKKKKKKKKKKKKKKKKKKKKKKK")
           # print (label_indices)
            
            restructured={}
            
            for dataset_type in ['train','dev']:
                restructured[dataset_type]=[item for sublist in label_indices[dataset_type] for item in sublist]
           
            #dev_label_restructure=[item for sublist in label_indices['dev'] for item in sublist]
            
            label_to_index = hlp.order_dictionary(label_to_index, 'value', reverse = False)
           # print ("________________________________________")
           # print (label_to_index)
            index_to_label = hlp.reverse_dictionary(label_to_index)     
            
            
            label_binarizer = MyLabelBinarizer()
            label_binarizer.fit(range(max(label_to_index.values())+1))
            #print(label_to_index.values())
            
            
            
            label_vector_indices = {}
            labels_just_values={}

            for dataset_type in ['train','dev']:
              label_vector_indices[dataset_type] = []
              labels_just_values[dataset_type]=[]
              for label_indices_sequence in label_indices[dataset_type]:
                label_vector_indices[dataset_type].append(label_binarizer.transform(label_indices_sequence)[0])
                labels_just_values[dataset_type].append(label_indices_sequence[0])
               # print ("TEST TRANSD")
               # print (label_binarizer.transform(label_indices_sequence)[0])
               # print (label_indices_sequence[0])
            
            #print("FORM")
           # print (label_vector_indices['train'])
           # print (label_vector_indices['dev'])
            #print ( labels_just_values['train'])
            

            label_indices['train']=label_vector_indices['train']
            label_indices['dev']=label_vector_indices['dev']    

            self.token_indices=token_indices
            self.token_to_index=token_to_index
            
            self.label_indicies=label_indices
            self.label_to_index=label_to_index
            self.index_to_label=index_to_label
            
            self.vocab_size=len(self.token_to_index)
            self.number_of_classes=len(self.label_to_index)
            
            self.labels_untransformed=labels_just_values
            
            
           # print (self.label_to_index)
           # print (self.token_indices)
           # print (self.label_indicies)
           # print (self.vocab_size)
            #print (token_to_index)


#dataset=Dataset_pubmed("TWITTER3") # TWITTER3 PUBMED
#z=hlp.get_valid_dataset_filepaths({'dataset_text_folder':"./TWITTER/2016A"})
#print (z)
#print (z)


#dataset.load_dataset(z)

#dataset=Dataset("GRAMMAR")
#z=hlp.get_valid_dataset_filepaths({'dataset_text_folder':"./GRAMMAR2"})
#dataset.load_dataset(z)