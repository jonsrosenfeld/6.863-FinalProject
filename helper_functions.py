#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 16:10:34 2018

@author: elena
"""
import numpy as np
import random
import os
import shutil
import collections
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

"""
Change space separated fotmat to tsv (tab separated) for it to work with torchtext

"""

seed = 0
if seed!=0:
    rng=random.Random(seed)
else:
    rng=random.Random()


def wchoices(population,weights=None):
    if weights==None:
        weights=np.ones(len(population)).tolist()
    cumsum=[0]
    for w in weights:
        cumsum.append(w+cumsum[len(cumsum)-1])
    
    cumsum_vec= np.array(cumsum)
    point=rng.uniform(0,cumsum[len(cumsum)-1])
    ind_leq_vec=(cumsum_vec<=point)
    c=int(np.sum(np.ones(len(population)+1)[ind_leq_vec])-1)
    return [population[c]]
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def filterline(line):
    slist=line.split()
    if len(slist)>0:
        if is_number(slist[0]):    #grammer rules iff start with a wieght.
            s_comments_removed=line.split('#')[0] # remove comments
            slist=s_comments_removed.split() #note that name re-use is adecuate in this case as slist is identical apart from removed comment section
            L=len(slist) # note that L > 2
    
            weight=float(slist[0]) 
            try:
                LHS=slist[1]
            except:
                print('error reading grammar line: improper format')
                return None
            if L==3: #RHS is kept as a list of strings. 
                RHS=[slist[2]] # if L==3 it means that RHS contains only one string. i.e. len(RHS)=1
            else:
                RHS=slist[2:]
                if not(RHS):
                    print('error reading grammar line: RHS is empty')
                    return None
            return (weight,LHS,RHS)
        else:
            return ()
    else:
        return ()


    
def read_grammar(filename):
    with open(filename,'r') as grm:
        rule_dictionary={} #holds grammar rules
        weight_dictionary={} #hold wieghts associated with corresponding rules
        for line in grm:
            retval=filterline(line)
            if retval:
                (weight,LHS,RHS)=retval
                RHS_rules_list=rule_dictionary.get(LHS,False)
                RHS_weight_list=weight_dictionary.get(LHS,False)

                if not(RHS_rules_list):
                    rule_dictionary[LHS]=[RHS]
                    weight_dictionary[LHS]=[weight]
                else:
                    RHS_rules_list.append(RHS)
                    RHS_weight_list.append(weight)
            else:
                continue
        return rule_dictionary,weight_dictionary
    
def rule_select(LHS,rule_dictionary,weight_dictionary,term_symbol=None):
    '''
    receives a LHS (string) and selects, randomly, a RHS row vector out of matrix possible RHS expansions.
    random selection is relatively weighted according to the relative weights of all RHS possible expansions for given LHS.
    
    return value: selected RHS vector (list of strings) or term_symbol indicating the requested LHS was terminal.
    '''
    try:
        RHS_list=rule_dictionary.get(LHS,term_symbol) # if no such LHS is found it means it is terminal
        if RHS_list==term_symbol:
            return term_symbol
        
        weight_list=weight_dictionary.get(LHS)
    #    #rng=random.Random() #this is globaly defined
    #    choice=rng.choices(RHS_list,weight_list)
        choice=wchoices(RHS_list,weight_list) #as implemented to support older python 2.7
    
        return choice
    except:
        print('error')

#
def isterminal(LHS,rule_dictionary,weight_dictionary,term_symbol=None):
    retval=rule_select(LHS,rule_dictionary,weight_dictionary,term_symbol)
    if retval==term_symbol:
        return True
    else:
        return False



def rule_expantion_step(LHS,rule_dictionary,weight_dictionary,disp_tree=False,term_symbol=None):
    '''
    given a LHS (string), randomly expand until all terminal values reached.
    
    return string containing all terminals as traversed Left-searched
    retrun string tree
    
    '''
    curr_RHS=rule_select(LHS,rule_dictionary,weight_dictionary,term_symbol)
    if curr_RHS==term_symbol: 
        if disp_tree:
            return ''
        else:
            return LHS+' ' # this is a terminal string
    next_LHS_list=curr_RHS[0]
    substr=''
    
    for LHS_candidate in next_LHS_list:

        cont_str=rule_expantion_step(LHS_candidate,rule_dictionary,weight_dictionary,disp_tree,term_symbol)
        if disp_tree:
            if not(isterminal(LHS_candidate,rule_dictionary,weight_dictionary,term_symbol)):
                substr=substr+'('+LHS_candidate+' '+cont_str+')'
            else:
                substr=substr+LHS_candidate+' '+cont_str
        else:
            substr=substr+cont_str

    
    return substr


def rule_expantion(rule_dictionary,weight_dictionary,disp_tree=False,term_symbol=None):
    if disp_tree:
        fullstr='(START '+rule_expantion_step('START',rule_dictionary,weight_dictionary,disp_tree,term_symbol)+')'
    else:
        fullstr=rule_expantion_step('START',rule_dictionary,weight_dictionary,disp_tree,term_symbol)
    return fullstr

#### un-grammafier functions:
    
def alter_line(line, pre_terms_dict,vocab_to_preterm_dict,rule_dictionary,weight_dictionary):
    slist=line.split()
    alter_index=wchoices([i for i in range(len(slist)-1)])
    word=slist[alter_index[0]]
    term_list=word2preterms(word,vocab_to_preterm_dict) #returns a list of all pre-terminals
    while term_list==None: # the original sentence may contain 'hard wired words, which do not correspond to any preterminal'
        alter_index=wchoices([i for i in range(len(slist)-1)])
        word=slist[alter_index[0]]
        term_list=word2preterms(word,vocab_to_preterm_dict) #returns a list of all pre-terminals
    
    altered_word=choose_disjoint_word(term_list,pre_terms_dict,rule_dictionary,weight_dictionary)
            # last element - verify belongs to line-endings and discard
            #if not (slist.pop() in line_endings):
    slist[alter_index[0]]=altered_word
    return " ".join(slist)

def word2preterms(word,vocab_to_preterm_dict):
    return vocab_to_preterm_dict.get(word)

def choose_disjoint_word(term_list,pre_terms_dict,rule_dictionary,weight_dictionary):
#    key_list=[key for key in pre_term_dict.keys()]
    cand=wchoices([key for key in pre_terms_dict.keys()])
    while cand in term_list:
        cand=wchoices([key for key in pre_terms_dict.keys()])
    word=rule_select(cand[0],rule_dictionary,weight_dictionary)[0][0]
    return word
 
def get_terminal_structures(rule_dictionary,weight_dictionary):
    '''
    for a given rule_dictionary
    return:
        pre_terms_dict: dictionary containing all LHS which are pre-terminal {LHS: #terminals}
        vocab_to_preterm_dict: dictionary {termianl: [pre_term1,pre_term2,...]} - containing 
        all the potential preterms leading to a (terminal) word in the vocabulary 
    '''
    key_list=[key for key in rule_dictionary.keys()]
    pre_terms_dict={}
    vocab_to_preterm_dict={}
    for LHS in key_list:
        RHS_list=rule_dictionary[LHS]
        for RHS in RHS_list:
            cand_term=RHS[0]
            if not(len(RHS)==1): 
                break #the LHS leading to this RHS can not be is a preterminal 
            else:
                if isterminal(cand_term,rule_dictionary,weight_dictionary): #Indeed the LHS is a pre-term        
                    
                    val=pre_terms_dict.get(LHS,False)
                    if val==False:
                        pre_terms_dict[LHS]=1
                    else:
                        pre_terms_dict[LHS]+=1
                    
                    pre_term_list=vocab_to_preterm_dict.get(cand_term,False)
                    if pre_term_list==False:
                        vocab_to_preterm_dict[cand_term]=[LHS]
                    else:
                        vocab_to_preterm_dict[cand_term].append(LHS) #add the preterminal that led to this terminal
    return pre_terms_dict,vocab_to_preterm_dict

def gen_corpus(output_filename,rule_dictionary,weight_dictionary,corpus_size):
    
    with open(output_filename,'w') as fout:
        for it in range(corpus_size):
            line_out=rule_expantion(rule_dictionary,weight_dictionary)+'\n'
            fout.write(line_out)
    
def gen_ungram_corpus(corpus_filename,output_filename,rule_dictionary,weight_dictionary):

    pre_terms_dict,vocab_to_preterm_dict = get_terminal_structures(rule_dictionary,weight_dictionary)

    with open(corpus_filename,'r') as fin, open(output_filename,'w') as fout:
        for line in fin:
            line_out=alter_line(line,pre_terms_dict,vocab_to_preterm_dict,rule_dictionary,weight_dictionary)+'\n'
            fout.write(line_out) 
            
def run_a_script():
    shell_command = "/mit/6.863/spring2018/Software/parse -g grammar2 -i ungrammatical -o output_parses"
    os.system(shell_command)
            
def filter_ungram(filt_fn,ug_cand_fn,out_fn):
    with open(filt_fn,'r') as fflt, open(ug_cand_fn,'r') as fcand, open(out_fn,'w') as fout:
        for linef, linec in zip(fflt, fcand):
            if linef[0]=='f':
               fout.write(linec)
               
def chain_commands_for_ungrammatical():
    rule_d,weight_d=read_grammar("grammar2")
    gen_corpus("grammatical", rule_d,weight_d,30000)
    gen_ungram_corpus("grammatical","ungrammatical",rule_d,weight_d)    
    run_a_script()
    filter_ungram("output_parses","ungrammatical","filtered_result")


''' End grammar2  helpers  '''


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


"""
Add the class indicator feature to the input: for testing purposes, the system should be able to get 100% rather fast

"""       
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
"""
Delete all longer then n inputs 
"""         
         
def delete_longer_then(n,file_input,file_output):
        f=open(file_input, 'r')
        f_o=open(file_output, 'a')
        for line in f :
            set_line=line.strip("\n").split(" ")
            if len(set_line)>n:
                continue
            else:
                f_o.write(line)
                
def dec2binstr(num_dec,N):
    s=format(num_dec, '0'+str(N)+'b')
    return s
def dec2ab(num_dec,N):
    s= dec2binstr(num_dec,N)#str.format('%'+str(N)+'s', int.toBinaryString(num_dec)).replace(" ", "0")
    return alterbin2ab(s)

def alterbin2ab(num_bin_str):
    s=num_bin_str
    s=s.replace("0","a ")
    s=s.replace("1","b ")   
    return s

def get_number_sets(sent_len):
    
    N=sent_len
    ungram_num_list=[i for i in range(2**N)] #init as all numbers 0..2^N-1
    if N%2 !=0:
        gram_num_list=[]
    else:
       # print('N='+str(N)+',=>'+str(2**(int(N/2))-1))
        gram_num_list=[2**(int(N/2))-1] # this is the only grammatical sequence corresponding number for sequence length of length N
        ungram_num_list.remove(gram_num_list[-1]) # remove the grammatical corresponding number to get the ungrammatical number set
    return gram_num_list,ungram_num_list

def get_number_gr(sent_len):
   # print (sent_len)
    N=sent_len
    if N%2 !=0:
        gram_num_list=[]
    else:
#        print('N='+str(N)+',=>'+str(2**(int(N/2))-1))
        gram_num_list=[2**(int(N/2))-1] # this is the only grammatical sequence corresponding number for sequence length of length N
    return gram_num_list

def gen_full_data(max_sent_len,output_gr_filename,output_ungr_filename):
    with open(output_gr_filename,'w') as fgr, open(output_ungr_filename,'w') as fug:
        for N in np.arange(1,max_sent_len+1):
            gram_num_list,ungram_num_list=get_number_sets(N)
            if gram_num_list:
                line_gr=dec2ab(gram_num_list[-1],N)+'GR\n'
                fgr.write(line_gr)
            for entry in range(len(ungram_num_list)):
                line_ug=dec2ab(ungram_num_list[entry],N)+'UGR\n'
                fug.write(line_ug)

def gen_gr_only(max_sent_len,output_gr_filename):
    with open(output_gr_filename,'w') as fgr:
        for N in np.arange(1,max_sent_len+1):
            gram_num_list=get_number_gr(N)
            if gram_num_list:
                line_gr=dec2ab(gram_num_list[-1],N)+'GR\n'
                fgr.write(line_gr)                
            
def gen_ug_template(max_sent_len,output_ug_filename,ungram_func):
    with open(output_ug_filename,'w') as fug:
        for N in np.arange(1,max_sent_len+1):
            gram_num_list=get_number_gr(N)
            if gram_num_list:
                gr_binstr=dec2binstr(gram_num_list[-1],N)
                ug_binstr_list=ungram_func(gr_binstr)
                for ug_binstr in ug_binstr_list:
                    line_ug=alterbin2ab(ug_binstr)+'UGR\n'
                    fug.write(line_ug)

def gen_ug_hammdist(max_sent_len,output_ug_filename,dist):
    def hammdist(s):
        return hamming(s, dist)
    gen_ug_template(max_sent_len,output_ug_filename,hammdist)
    
def gen_ug_subset(max_sent_len,output_ug_filename):
    with open(output_ug_filename,'w') as fug:
        gr_binstr=dec2binstr(get_number_gr(max_sent_len)[-1],max_sent_len)
        ug_binstr_list=substring(gr_binstr)
        for N in np.arange(1,max_sent_len+1):
           # print (N)
            gram_num_list=get_number_gr(N)
            if gram_num_list:
                gr_binstr=dec2binstr(gram_num_list[-1],N)
                if gr_binstr in ug_binstr_list:
                    ug_binstr_list.remove(gr_binstr)
                       
        for ug_binstr in ug_binstr_list:
            line_ug=alterbin2ab(ug_binstr)+'UGR\n'
            fug.write(line_ug)

def flip(c):
    return str(1-int(c))

def flip_s(s, i):
    t =  s[:i]+flip(s[i])+s[i+1:]
    return t

def hamming(s, k):
    if k>1:
        c = s[-1]
        s1 = [y+c for y in hamming(s[:-1], k)] if len(s) > k else []
        s2 = [y+flip(c) for y in hamming(s[:-1], k-1)]
        r = []
        r.extend(s1)
        r.extend(s2)
        return r
    else:
        return [flip_s(s,i) for i in range(len(s))]

def substring(s,inclusive=False):
    substr_list=[]
    for startind in range(len(s)-1):
        for endind in np.arange(startind,len(s)+1):
            cand_str=s[startind:endind]
            if cand_str not in substr_list:
                substr_list.append(cand_str)
    if not inclusive:
        substr_list.remove('')
        substr_list.remove(s)
    return substr_list

def shuffle_split(infilename, outfilename1, outfilename2):
    from random import shuffle

    with open(infilename, 'r') as f:
        lines = f.readlines()

    # append a newline in case the last line didn't end with one
    lines[-1] = lines[-1].rstrip('\n') + '\n'

    shuffle(lines)

    with open(outfilename1, 'w') as f:
        f.writelines(lines[:len(lines) // 2])
    with open(outfilename2, 'w') as f:
        f.writelines(lines[len(lines) // 2:])
        
def upsample(size_upsample_to,original_file,upsampled_file):
    with open(original_file) as f:
     content = f.readlines()
    upsampled_f=open(upsampled_file,'a')
    
    original_content_length=len(content)
    original_content = content[:]

    while (len(content)<size_upsample_to):
        t=random.randint(0,original_content_length-1)
       # print(t)
        content.append(original_content[t])    
      #  print (len(content))
    for element in content:
        upsampled_f.write(element)

def filter_out(mode, value, input_filename,output_filename,filtered_filename):
    f_filtered=open(output_filename, 'a')
    f_output=open(filtered_filename, 'a')
    
    
    if mode=="shorter_then":
          with open(input_filename) as f:
           content = f.readlines()
          for line in content:
              line_long=line.split(" ")
              if len(line_long)-1<value :
                  #print (line)
                  f_output.write(line)
              else:
                  f_filtered.write(line)
        
    if mode=="list_f":
        with open(input_filename) as f:
           content = f.readlines()
        for idx, line in enumerate(content):
            if idx in value:
                f_filtered.write(line)
            else: 
                f_output.write(line)
                
def random_sample(input_file, sample_up_to, result):  
    f_result=open(result, 'a')
    with open(input_file) as f:
     content = f.readlines()
    content_lines=[]
    #original_content_length=len(content_lines)
    while len(content_lines)<sample_up_to:
         #print (len(content_lines))
         t=random.randint(0,len(content)-1)
         content_lines.append(content[t])  
    for element in content_lines:
        f_result.write(element)              
        
        
def get_number_of_examples(file):
    f=open(file,'r')
    number=0
    for line in f:
        number+=1
    f.close()
    return (number)

def merge_files(file1,file2,output):
    with open(output,'wb') as wfd:
     for f in [file1,file2]:
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd, 1024*1024*10)
            
def merge_file_multiple(filelist,output):
    with open(output,'wb') as wfd:
      for f in filelist:
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd, 1024*1024*10)
            
def shuffle_file(file1):
    lines = open(file1).readlines()
    print ("Shuffle")
    random.shuffle(lines)
    open(file1, 'w').writelines(lines)

def generate_dataset(max_lengh_general,generate_test=False,types_of_ungr="both",type_prop={"grammatical":1/2,"switch":1/4,"misscount":1/4}, \
                      final_sizes_dictionary={"train":60000,"dev":8000,"test":8000}, construction_type="all_then_upsample"):  #types_of_ungr: misscount, switch, both 
    if generate_test:
        dataset_types={"train","dev","test"}
    
    example_types={}
    gen_gr_only(max_lengh_general,'gr_data.txt')
    filter_out("shorter_then",250,"gr_data.txt","gr_bigger.txt","gr_less.txt")
    
    example_types['grammatical']={}
    example_types['grammatical']['short']="gr_less.txt"
    example_types['grammatical']['long']="gr_bigger.txt"
    
    if types_of_ungr=="both" or types_of_ungr=="misscount":
      gen_ug_subset(max_lengh_general,'ug_data_subset.txt')
      filter_out("shorter_then",250,"ug_data_subset.txt","subset_bigger.txt","subset_less.txt")
      
      example_types['misscount']={}
      example_types['misscount']['short']="subset_less.txt"
      example_types['misscount']['long']="subset_bigger.txt"
      
    if types_of_ungr=="both" or types_of_ungr=="switch":
      gen_ug_hammdist(max_lengh_general,'ug_data_hamm1.txt',1)
      filter_out("shorter_then",250,"ug_data_hamm1.txt","switch_bigger.txt","switch_less.txt")
      
      example_types['switch']={}
      example_types['switch']['short']='switch_less.txt'
      example_types['switch']['long']='switch_bigger.txt'

    total_lengths={}
    for elements in example_types.keys():
        total_lengths[elements]={}
        for length in {'short','long'}:
          total_lengths[elements][length]=get_number_of_examples(example_types[elements][length])
    print (total_lengths)
        
      
    if generate_test==True:
        split_type={}
        for split in {"train","dev","test"}:
            split_type[split]={}
            for example in {"misscount","switch","grammatical"}:
                print (final_sizes_dictionary[split])
                print (float(type_prop[example]))
                split_type[split][example]=final_sizes_dictionary[split]*type_prop[example]
    print (split_type)
    
    
    merge_files={}
    if construction_type=="all_then_upsample":
        for dataset_type in  dataset_types:
                merge_files[dataset_type]=[]
                for datatype in example_types.keys():
                    if dataset_type=='train':
                        get_samples_from=example_types[datatype]['short']
                        data_length_type='short'
                    else :
                        get_samples_from=example_types[datatype]['long']
                        data_length_type='long'
                        
                    filename="balanced_sample_"+str(datatype)+"_"+dataset_type
                    merge_files[dataset_type].append(filename)
                    print (filename)
                    if total_lengths[datatype][data_length_type]>split_type[dataset_type][datatype]:
                        random_sample(get_samples_from, split_type[dataset_type][datatype],filename)
                    else:
                        upsample(split_type[dataset_type][datatype],get_samples_from,filename)
    for element in merge_files:      
           output_string='merged_'+str(element)
           merge_file_multiple( merge_files[element] ,output_string)            
    shuffle_file("merged_train")           


         
                        
def get_all_errors(input_file,output_file):
    f=open (input_file,'r')
    for line in f:
        split=line.split(" ")
        if split[-3]!=split[-5]:
            counter=collections.Counter(split)
            print (line)
            print ("a "+ str(counter["a"])+" "+"b "+ str(counter["b"]))
           
def compute_dataset_overlap(data_1,data_2):
  with open(data_1, 'r') as file1:
    with open(data_2, 'r') as file2:
        same = set(file1).intersection(file2)

  same.discard('\n')

  with open('same.txt', 'w') as file_out:
    for line in same:
        file_out.write(line)
        
def parse_results_file(filename):
    f=open(filename,'r')
    experiments={}
    one_experiment=[]
    
    number_of_batches_seen=[]
    dev_accuracy=[]
    train_accuracy=[]
    
    key=""
    
    for line in f:
        # MARKER 20-10-(sorted)
        line_tokens=line.split()
        if len(line_tokens)==0:
            continue
        elif line_tokens[0]=="EXP_END":
            #print ("EE")
            one_experiment.append(number_of_batches_seen)
            one_experiment.append(dev_accuracy)
            one_experiment.append(train_accuracy)
            
            experiments[key]=one_experiment
            
            one_experiment=[]
            number_of_batches_seen=[]
            dev_accuracy=[]
            train_accuracy=[]

        elif line_tokens[0]=="MARKER":
            key=line_tokens[1]
           # print (key)
        else: 
            #print (line_tokens)
            number_of_batches_seen.append(int(line_tokens[2]))
            dev_accuracy.append(float(line_tokens[-1]))
            train_accuracy.append(float(line_tokens[-2]))
    return experiments        
            

def create_plot(report_adress,save_adress):       
  experiments=parse_results_file(report_adress)        
# "5-5-100-log-forma.txt"
# "fixed-hidden_layer-50"
# "Fixed_embed20-dif-hidden.txt"
  colors=['b','g','r','c','m','y','k']
  color_index=0
  for keys in experiments: 
    print (keys)
    print ("--")
    parsed_key=keys.split("-")
    print (parsed_key)
    
    t = np.asarray(experiments[keys][0])
    test = np.asarray(experiments[keys][1])
    train =np.asarray(experiments[keys][2])

    plt.plot(t, test, color=colors[color_index], label=str(parsed_key[0])+"-"+str(parsed_key[1])+'-test')
    plt.plot(t, train, color=colors[color_index], linestyle='--', label=str(parsed_key[0])+"-"+str(parsed_key[1])+'-train')
    plt.legend()
    color_index+=1
    plt.savefig(save_adress, format='eps',dpi=1000)


def split_to_sentences_and_labels(input_file,sentences,labels):
    ""
    input_f=open(input_file,'r')
    out_sentence=open(sentences, 'a')
    out_labels=open(labels, 'a')
    for line in input_f:
        items=line.split()
        out_labels.write(items[-1]+" "+"\n")
        out_sentence.write(' '.join(items[0:-1])+" "+"\n")
    out_sentence.close()
    out_labels.close()
    input_f.close()


""" Two helpers for merging GRAMMAR2 like output """   
def mark_examples_as(file1, mark): # don't forget to add " " before the mark
    with open(file1, 'r') as f:
      file_lines = [''.join([x.strip(), mark, '\n']) for x in f.readlines()]
    with open(file1, 'w') as f:
      f.writelines(file_lines) 
    

def process(gramatical,ungrammatical,output):
    mark_examples_as(gramatical," GR")
    mark_examples_as(ungrammatical, " NG")
    merge_files(gramatical,ungrammatical,output)
    shuffle_file(output)    
    

  
#split_to_sentences_and_labels("dev_g2.txt","s","l")
#    
    
#run_a_script()


#red_patch = mpatches.Patch(color='red', label='The red data')

#os.chdir("train_sizes")
#generate_dataset(500,generate_test=True,types_of_ungr="both",type_prop={"grammatical":0.5,"switch":0.25,"misscount":0.25}, \
#                     final_sizes_dictionary={"train":2000,"dev":8000,"test":8000}, construction_type="all_then_upsample")
        
#compute_dataset_overlap("train_g2.tsv","dev_g2.tsv")
           
#get_all_errors("train_new_presictions_20-168-E15.txt","")
#get_all_errors("dev_new_presictions_20-168-E15.txt", "")
                        
                    
                    
            
        

#os.chdir("remove")
#generate_dataset(500,generate_test=True)

#space_sep_to_space_sep("merged_train-2000","new_train-2000.tsv")
#space_sep_to_space_sep("merged_dev","new_dev.tsv")
#space_sep_to_space_sep("merged_test","new_test.tsv")         