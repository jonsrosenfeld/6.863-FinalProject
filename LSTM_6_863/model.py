#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 18:21:06 2018

@author: elena
"""
import tensorflow as tf
import numpy as np
import codecs
import Dataset as ds
import Helpers as hlp

import sklearn.preprocessing


def bidirectional_GRU(input,hidden_state_dimension,initializer,sequence_length=None, output_sequence=True):
    print ("Biderectional GRU")
    with tf.variable_scope("biderectional_GRU"):
        if sequence_length==None:
            batch_size=1 # ONE WORD(char)
            sequence_length = tf.shape(input)[1]
            sequence_length = tf.expand_dims(sequence_length, axis=0, name='sequence_length')  #NOT SURE IF IT EVER HAPPENS
        else:
            batch_size= tf.shape(sequence_length)[0]
            
            
        gru_cell={}
        initial_state={}
        for direction in ["forward","backward"]: 
            gru_cell[direction] = tf.contrib.rnn.GRUCell(hidden_state_dimension)  
            initial_state[direction]=gru_cell[direction].zero_state(batch_size, tf.float32)           
        outputs,final_states = tf.nn.bidirectional_dynamic_rnn(gru_cell["forward"],gru_cell["backward"],input, sequence_length=sequence_length,initial_state_fw=initial_state["forward"],initial_state_bw=initial_state["backward"])    

        
        if output_sequence==True:
           outputs_forward, outputs_backward = outputs        
           output = tf.concat([outputs_forward, outputs_backward], axis=2, name='output_sequence')

        else:
            final_states_forward, final_states_backward = final_states                      

            output = tf.concat([final_states_forward, final_states_backward], axis=1, name='output') #111

        return output
    
def bidirectional_LSTM(input, hidden_state_dimension, initializer, sequence_length=None, output_sequence=True):
    
    
    print ("Biderectional LSTM")
    with tf.variable_scope("bidirectional_LSTM"):
        if sequence_length == None:
            batch_size = 1
            sequence_length = tf.shape(input)[1]
            sequence_length = tf.expand_dims(sequence_length, axis=0, name='sequence_length')
        else:
            batch_size =tf.shape(input)[0]
            print (batch_size)

        lstm_cell = {}
        initial_state = {}
        for direction in ["forward", "backward"]:
            with tf.variable_scope(direction):
                # LSTM cell
                lstm_cell[direction] = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(hidden_state_dimension, use_peepholes=False, forget_bias=1.0, initializer=initializer, state_is_tuple=True, activation=tf.tanh) # tf.tanh (default to RELU)
               # lstm_cell[direction] = tf.contrib.rnn_cell.GRUCell(hidden_state_dimension,activation=tf.tanh,)
                
                
                # initial state: http://stackoverflow.com/questions/38441589/tensorflow-rnn-initial-state
                initial_cell_state = tf.get_variable("initial_cell_state", shape=[1, hidden_state_dimension], dtype=tf.float32, initializer=initializer)
                initial_output_state = tf.get_variable("initial_output_state", shape=[1, hidden_state_dimension], dtype=tf.float32, initializer=initializer)
                c_states = tf.tile(initial_cell_state, tf.stack([batch_size, 1]))
                h_states = tf.tile(initial_output_state, tf.stack([batch_size, 1]))
                initial_state[direction] = tf.contrib.rnn.LSTMStateTuple(c_states, h_states)
                

        # sequence_length must be provided for tf.nn.bidirectional_dynamic_rnn due to internal bug
        outputs, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_cell["forward"],
                                                                    lstm_cell["backward"],
                                                                    input,
                                                                    dtype=tf.float32,
                                                                    sequence_length=sequence_length,
                                                                    initial_state_fw=initial_state["forward"],
                                                                    initial_state_bw=initial_state["backward"])
        
        if output_sequence == True:
            outputs_forward, outputs_backward = outputs
            output = tf.concat([outputs_forward, outputs_backward], axis=2, name='output_sequence')
        else:
            # max pooling
#             outputs_forward, outputs_backward = outputs
#             output = tf.concat([outputs_forward, outputs_backward], axis=2, name='output_sequence')
#             output = tf.reduce_max(output, axis=1, name='output')
            # last pooling
            final_states_forward, final_states_backward = final_states
            output = tf.concat([final_states_forward[1], final_states_backward[1]], axis=1, name='output')

    return output      
class LSTM(object):
    def __init__(self, dataset, parameters):
        self.input_token_indices = tf.placeholder(tf.int32, [None, None], name="input_token_indices")
        self.input_label_indices_vector = tf.placeholder(tf.float32, [None, dataset.number_of_classes], name="input_label_indices_vector")
        
        self.input_sentence_lengths = tf.placeholder(tf.int32, [None], name="input_token_lengths")
        
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("embedding"):
            self.embedding_weights = tf.get_variable("embedding_weights",shape=[dataset.vocab_size, 10],initializer=initializer)
            embedded_tokens = tf.nn.embedding_lookup(self.embedding_weights, self.input_token_indices, name='embedded_tokens')
            print (embedded_tokens)
        with tf.variable_scope('lstm'):
            lstm_output = bidirectional_LSTM(embedded_tokens, parameters['lstm_hidden_state_dimension'], initializer,sequence_length=self.input_sentence_lengths, output_sequence=False)
            print (lstm_output)
        lstm_flat = tf.reshape(lstm_output, [-1, 80])
        with tf.name_scope("output"):
              W = tf.Variable(tf.truncated_normal([80, dataset.number_of_classes], stddev=0.1), name="W")
              b = tf.Variable(tf.constant(0.1, shape=[dataset.number_of_classes]), name="b")      
              self.unary_scores = tf.nn.xw_plus_b(lstm_flat, W, b, name="scores")
              self.predictions = tf.argmax(self.unary_scores, 1, name="predictions")
              print(self.unary_scores)
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.unary_scores, labels=self.input_label_indices_vector, name='softmax')
            self.loss =  tf.reduce_mean(losses, name='cross_entropy_mean_loss')
        
        
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(0.005)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        
        def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad, -5.0, 5.0)
        grads_and_vars = [(ClipIfNotNone(grad), var) for grad, var in grads_and_vars]
        self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        
def train_step(sess, dataset, sequence_range, model):
    token_indices_sequence = dataset.token_indices['train'][sequence_number]

tf.reset_default_graph ()


def prediction_step(sess, dataset, dataset_type, model,epoch_number):
    print('Evaluate model on the {0} set'.format(dataset_type))
    all_predictions = []
    all_y_true = []
    
    store_at_dev="dev/epoche_"+str(epoch_number)+".txt"
    store_at_train="train/epoche_"+str(epoch_number)+".txt"
    
    f_store=open(store_at_dev,'a') 
    f_store_train=open(store_at_train,'a')
    
    sent_length_modified= np.reshape([100], (1, ))     
    for i in range(len(dataset.token_indices[dataset_type])):
        token_1_b=np.reshape(dataset.token_indices['train'][i], (1, 100))
        label_1_b=np.reshape(dataset.label_indicies['train'][i],(1,2))
        feed_dict = {
        model.input_token_indices:token_1_b,
        model.input_label_indices_vector:label_1_b,
        model.input_sentence_lengths: sent_length_modified
    }
        unary_scores, predictions = sess.run([model.unary_scores, model.predictions], feed_dict)
        predictions = predictions.tolist() 
        
        prediction_labels = [dataset.index_to_label[prediction] for prediction in predictions]
        gold_label = dataset.labels_untransformed[dataset_type][i]
        gold_label=dataset.index_to_label[gold_label]
       # print ("PREDICTED "+str(prediction_labels))
        print ("Gold "+str(gold_label)+" "+"Predicted "+str(prediction_labels[0]))
        all_predictions.extend(prediction_labels[0])
        all_y_true.extend(gold_label)

#dataset.token_indices[dataset_type][i] +
        results=" " + "true " + str(gold_label) + " " +str(prediction_labels[0])

        if dataset_type=="dev":
                f_store.write(results+ "\n")
                
        if dataset_type=="train":
                f_store_train.write(results+ "\n")
                
    if dataset_type=="dev":
            f_store.write("\n")
    if dataset_type=="train":
            f_store_train.write("\n")




# 50 batch size

dataset=ds.Dataset("GRAMMAR")
z=hlp.get_valid_dataset_filepaths({'dataset_text_folder':"./GRAMMAR2"})
dataset.load_dataset(z)
sess = tf.Session()

with sess.as_default():      
  parameters={}
  parameters['lstm_hidden_state_dimension']=40
  model=LSTM(dataset,parameters)
  sess.run(tf.global_variables_initializer())
  for n in range (0,40):
   print ("EPOCHE "+str(n))
   for i in range(0,200000,20):
    print ("Example_set "+str(i))
    L=dataset.token_indices['train'][i:i+20]
    L=np.array(L)  
      
    L2=dataset.label_indicies['train'][i:i+20]
    L2=np.array(L2)
    print(L2.shape)
    
    print (L)
    print (L2)
    
    feed_dict = {
      model.input_token_indices: L,
      model.input_label_indices_vector: L2,
      model.input_sentence_lengths: [100] * 20
      #model.input_label_indices_flat: dataset.label_indices['train'][sequence_number],
      
    }
    _,_,loss=sess.run([model.train_op, model.global_step, model.loss],feed_dict)
    print (loss)
   prediction_step(sess,dataset,"train",model,n)
   prediction_step(sess, dataset, "dev", model,n)
    
