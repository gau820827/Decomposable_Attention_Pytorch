
# coding: utf-8

# Dealing with unseen words, initailize with random vector
# https://www.researchgate.net/publication/309037295_Language_Models_with_Pre-Trained_GloVe_Word_Embeddings
# 
# 
# report:
# 1. try and catch the exception(can not find the words): 
#  2. using a function to do: ignore true or false :
#      if ignore true: the unseen words to the same random vec; if ignore false: the unseen words to the different random vec;
#  3. count the OOV number
# 2. sent the list into the model: [P1, P2, phi(P1): these are several vectors, phi(P2), Label]: l1*d1 and l2*d2
# 3. deal with the '?' '!' in the end of the sentence and lowercase to tokenize
# ------------
# 4. PARALEX: http://knowitall.cs.washington.edu/paralex/README_experiments.html
# ------------
# Question:
# 1. False should only add OOV if OOV is usually
# 
# First, use the blank split then do the tokne(find out the http and www)

# # process
# N-GRAM encoding: 
# 
# paralex pretrain:(noisy-pretrain)

# In[34]:

# %%writefile pretrain_glove.py
import pprint
import numpy as np
from operator import itemgetter
import re
import nltk
# nltk.download('punkt')


# In[20]:

emb_size = 50
pretrain_filename = 'glove.6B.50d.txt'
document_filename = 'a.txt'
document_filename = 'quora_duplicate_questions.tsv'
ignore = True 


# In[21]:

glove_home = './'
sst_home = './'
words_to_load = 50000
with open(glove_home + pretrain_filename) as f:
    loaded_embeddings = np.zeros((words_to_load, emb_size))
    words = {}
    idx2words = {}
    ordered_words = []
    for i, line in enumerate(f):
        if i >= words_to_load: 
            break
        s = line.split()
        loaded_embeddings[i, :] = np.asarray(s[1:])
        words[s[0]] = i
        idx2words[i] = s[0]
        ordered_words.append(s[0])
        
def load_sst_data(path):
    data = []
    with open(path) as f:
        text = f.read().splitlines() 
        for i, line in enumerate(text):
            if i == 0: continue # ignore the first input
            example = {}
            text = re.sub(r'\s*(\(\d)|(\))\s*', '', line)
            text = re.split(r'\t+', line)
            example['text'] = text[1:]
            data.append(example)
            
    return data
# add UNK to loaded_embeddings
words['UNK'] = len(words)
loaded_embeddings = np.vstack((loaded_embeddings, (2 * np.random.random_sample((1, emb_size)) - 1)))
data_set = load_sst_data(sst_home + document_filename)


# In[ ]:

# data_set


# In[23]:

import numpy as np
from sklearn.model_selection import train_test_split
def train_test_sp(data_set):
    train_set, test_set = train_test_split(data_set, test_size=0.17)
    return train_set, test_set
train_set, test_set = train_test_sp(data_set)
train_set, validation_set = train_test_sp(train_set)


# Give the model embedding word vectors: token and to vector

# In[28]:

def ignore_OOV(toke, ignore, emb_size, loaded_embeddings, words): 
    '''decide how to deal with OOV'''
    '''if ignore is Flase, each OOV assign different vector'''
    '''if ignore is True, each OOV assign the same vector'''
    if ignore is False:
        words[toke] = len(words)
#         loaded_embeddings = np.vstack((loaded_embeddings, (2 * np.random.random_sample((1, emb_size)) - 1)))
        loaded_embeddings = np.concatenate((loaded_embeddings, (2 * np.random.random_sample((1, emb_size)) - 1)), axis=0)
    else:
        toke = 'UNK'
    return loaded_embeddings[words[toke]], loaded_embeddings

def sentence2vec(data_set, ignore, emb_size, loaded_embeddings):
    '''add UNK to the embedding'''
    '''numpy would only copy the array and not modify the reference'''
    sentence2vec = []
    for pair in data_set:
        try:#
            p_1, p_2, label = pair['text'][2], pair['text'][3],pair['text'][4]
            p_1_tok, p_2_tok = nltk.word_tokenize(p_1),nltk.word_tokenize(p_2)
            p_1_vec = []
            p_2_vec = []

            for toke_1 in p_1_tok:
                toke_1 = toke_1.lower()
                try:
                    p_1_vec.append(loaded_embeddings[words[toke_1]])
                except KeyError:
                    x, loaded_embeddings = ignore_OOV(toke_1, ignore, emb_size, loaded_embeddings, words)
                    p_1_vec.append(x)

            for toke_2 in p_2_tok:
                toke_2 = toke_2.lower()
                try:
                    p_2_vec.append(loaded_embeddings[words[toke_2.lower()]])
                except KeyError:
                    x, loaded_embeddings = ignore_OOV(toke_2, ignore, emb_size, loaded_embeddings, words)
                    p_2_vec.append(x)

            sentence2vec.append([p_1,p_1_vec,p_2,p_2_vec,label])
            
        except IndexError:
            continue
            
    return sentence2vec, loaded_embeddings

matrix, loaded_embeddings = sentence2vec(train_set, ignore, emb_size, loaded_embeddings)


# In[ ]:

print(matrix.shape)


# Pipeline, that is batch

# In[ ]:

def  batch_iter(nonzero_pairs, cooccurrences, batch_size):
    start = -1 * batch_size
    dataset_size = len(nonzero_pairs)
    order = list(range(dataset_size))
    random.shuffle(order)

    while True:
        start += batch_size
        word_i = []
        word_j = []
        counts = []
        
        if start > dataset_size - batch_size:
            # Start another epoch.
            #a = 3
            start = 0
            random.shuffle(order)
        batch_indices = order[start:start + batch_size]
        batch = [nonzero_pairs[index] for index in batch_indices]
        for k in batch:
            counts.append(cooccurrences[k])
            word_i.append(k[0])
            word_j.append(k[1])
        yield [counts, word_i, word_j]


# In[ ]:

# data_iter = batch_iter(nonzero_pairs, cooccurrences, batch_size)


# In[ ]:



