
# coding: utf-8

# In[1]:

import pprint
import numpy as np
from operator import itemgetter
import re
import nltk
import random 
import numpy as np 
from sklearn.model_selection import train_test_split
import torch


# In[27]:

class pretrain():
#     pretrain(emb_size, pretrain_filename, document_filename, ignore = True, batch_size = 32):
    '''emb_size = 50
    pretrain_filename = 'glove.6B.50d.txt'
    document_filename = 'quora_duplicate_questions.tsv'
    ignore = True 
    batch_size = 32'''
    def __init__(self, emb_size, pretrain_filename, document_filename, ignore, batch_size):
        
        glove_home = './'
        sst_home = './'
        
        self.emb_size = emb_size
        self.pretrain_filename = pretrain_filename
        self.document_filename = document_filename
        self.ignore = ignore
        self.batch_size = batch_size
        self.data_set = self.load_sst_data(sst_home + document_filename)
        ''' read pretrain glove file '''
        ''' return train_set, validation_set, test_set'''
        
       
        with open(glove_home + self.pretrain_filename) as f:
            load = f.readlines()
            words_to_load = len(load)
            words_to_load = 5000
            self.loaded_embeddings = np.zeros((words_to_load, self.emb_size))
            self.words = {}
            self.idx2words = {}
            self.ordered_words = []
            for i, line in enumerate(load):
                if i >= words_to_load: 
                    break
                s = line.split()
                self.loaded_embeddings[i, :] = np.asarray(s[1:])
                self.words[s[0]] = i
                self.idx2words[i] = s[0]
                self.ordered_words.append(s[0])
            self.words['UNK'] = len(self.words)
            self.loaded_embeddings = np.vstack((self.loaded_embeddings, (2 * np.random.random_sample((1, self.emb_size)) - 1)))
        
        
        train_set, self.test_set = self.train_test_sp(self.data_set)
        self.train_set, self.validation_set = self.train_test_sp(train_set)
        self.matrix, self.loaded_embeddings = self.sentence2vec(self.train_set, self.ignore, self.emb_size, self.loaded_embeddings)
    
    def load_sst_data(self, path):
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
    
    def train_test_sp(self, data_set):
        train_set, test_set = train_test_split(data_set, test_size=0.17)
        return train_set, test_set
    
    def ignore_OOV(self, toke, ignore, emb_size, loaded_embeddings, words): 
        '''decide how to deal with OOV'''
        '''if ignore is Flase, each OOV assign different vector'''
        '''if ignore is True, each OOV assign the same vector'''
        if ignore is False:
            words[toke] = len(words)
            loaded_embeddings = np.concatenate((loaded_embeddings, (2 * np.random.random_sample((1, emb_size)) - 1)), axis=0)
        else:
            toke = 'UNK'
        return loaded_embeddings[words[toke]], loaded_embeddings
    
    def sentence2vec(self, data_set, ignore, emb_size, loaded_embeddings):
        '''add UNK to the embedding'''
        '''numpy would only copy the array and not modify the reference'''
        '''return the matrix which has already been change to vector'''
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
                        p_1_vec.append(loaded_embeddings[self.words[toke_1]])
                    except KeyError:
                        x, loaded_embeddings = self.ignore_OOV(toke_1, ignore, emb_size, loaded_embeddings, self.words)
                        p_1_vec.append(x)

                for toke_2 in p_2_tok:
                    toke_2 = toke_2.lower()
                    try:
                        p_2_vec.append(loaded_embeddings[self.words[toke_2.lower()]])
                    except KeyError:
                        x, loaded_embeddings = self.ignore_OOV(toke_2, ignore, emb_size, loaded_embeddings, self.words)
                        p_2_vec.append(x)

                sentence2vec.append([p_1,p_1_vec,p_2,p_2_vec,label])

            except IndexError:
                continue
        return sentence2vec, loaded_embeddings

    def  batch_iter(self, matrix, batch_size):
        '''please input: pt.matrix and pt.batch_size'''
        '''This is for batch'''
        '''Out put is: [(p1_pad_list), (p2_pad_list), p1_str, p2_str, (label)]'''
        start = -1 * batch_size
        dataset_size = len(matrix)
        order = list(range(dataset_size))
        random.shuffle(order)

        while True:
            start += batch_size
            p1 = []
            p2 = []
            p1_2vec = []
            p2_2vec = []
            label = []
            p1_length_list = []
            p2_length_list = []

            if start > dataset_size - batch_size:
                start = 0
                random.shuffle(order)
            batch_indices = order[start:start + batch_size]
            batch = [matrix[index] for index in batch_indices]
            for i,k in enumerate(batch):

                p1_length_list.append(len(k[1]))
                p2_length_list.append(len(k[3]))

                p1.append(k[0])
                p1_2vec.append(k[1])

                p2.append(k[2])
                p2_2vec.append(k[3])
                #
                label.append(int(k[4]))

            max_length_p1 = np.max(p1_length_list)
            max_length_p2 = np.max(p2_length_list)

            p1_pad_list = []
            p2_pad_list = []
            for i,k in enumerate(batch):
                p1_padded_vec = np.pad(np.array(k[1]), 
                                        pad_width=((0,max_length_p1-len(k[1]))), 
                                        mode="constant", constant_values=0)
                p1_pad_list.append(p1_padded_vec)
#                 p1_pad_list.append(torch.from_numpy(p1_padded_vec))

                p2_padded_vec = np.pad(np.array(k[3]), 
                            pad_width=((0,max_length_p2-len(k[3]))), 
                                mode="constant", constant_values=0)
#                 p2_pad_list.append(torch.from_numpy(p2_padded_vec))
                p2_pad_list.append(p2_padded_vec)

            yield [p1_pad_list, p2_pad_list, p1, p2, torch.LongTensor(label)]
    
#             yield [torch.LongTensor(p1_pad_list), torch.LongTensor(p2_pad_list), p1, p2, torch.LongTensor(label)]
#             yield [(np.array(p1_pad_list)), (np.array(p2_pad_list)), p1, p2, torch.LongTensor(label)]
#             yield [torch.from_numpy(np.array(p1_pad_list)), torch.from_numpy(np.array(p2_pad_list)), p1, p2, torch.LongTensor(label)]
#             yield [torch.from_numpy(p1_pad_list), torch.from_numpy(p2_pad_list), p1, p2, torch.LongTensor(label)]
#             yield [torch.FloatTensor((p1_pad_list)), torch.FloatTensor((p2_pad_list)), p1, p2, torch.LongTensor(label)]


# In[28]:

emb_size = 50
pretrain_filename = 'glove.6B.50d.txt'
document_filename = 'quora_duplicate_questions.tsv'
ignore = True 
batch_size = 32


# In[24]:

pt = pretrain(emb_size,pretrain_filename,document_filename,ignore,batch_size)
# path = './' + document_filename
# data_set = pt.load_sst_data


# In[26]:

# train_set, validation_set, test_set = pt.train_test_sp()


# In[27]:

# len(pt.words)
# len(pt.test_set)
# len(pt.loaded_embeddings)
# type(pt.matrix)


# In[25]:

data_iter = pt.batch_iter(pt.matrix, pt.batch_size)


# In[26]:

test = next(data_iter)


# In[11]:

print(type(test[1]))
print(type(test[1][31]))
print(type(test[4]))
print(test[4][0])
print((test[0]))
print((test[4]))
print(type(test[4][31]))


# In[44]:

print(type(torch.LongTensor(test[0])))
print(type(torch.FloatTensor(test[0])))
print(type(torch.from_numpy(test[0][1])))
print(type(torch.from_numpy(test[0][2])))
print(type(torch.from_numpy(test[0][3])))
print(type(torch.from_numpy(test[0][4])))


# In[32]:

print(type((test[0][0])))
print(type((test[0][1])))
print(type((test[0][2])))
print(type((test[0][3])))
print(type((test[0][4])))
print(type(test[0]))


# In[153]:

test[1][0]


# In[ ]:



