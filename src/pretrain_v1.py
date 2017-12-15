import pprint
import numpy as np
from operator import itemgetter
import re
import nltk
import random 
import numpy as np 
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
class pretrain():
#     pretrain(emb_size, pretrain_filename_path, document_filename, ignore = True, batch_size = 32):
    '''
    emb_size = 50
    pretrain_filename = 'glove.6B.50d.txt'
    document_filename = 'quora_duplicate_questions.tsv'
    ignore = True 
    batch_size = 32
    '''
    '''
    Document file is located(path) in the same file, could be modified if you need to do so
    '''
    def __init__(self, emb_size, pretrain_filename, document_filename, ignore, batch_size, use_pretrain):
        ###change this route after test
        glove_home = './pretrainvec/'
        sst_home = './data/'
        # glove_home = './pretrainvec/'
        # sst_home = document_filename
        
        self.emb_size = emb_size
        self.pretrain_filename = pretrain_filename
        self.document_filename = document_filename
        self.ignore = ignore
        self.batch_size = batch_size
        self.use_pretrain = use_pretrain

        ''' read pretrain glove file '''
        ''' return train_set, validation_set, test_set'''
        
        if(use_pretrain is True):
            ###change this route after test
            with open(glove_home + self.pretrain_filename) as f:
            # with open(self.pretrain_filename) as f:
                load = f.readlines()
                words_to_load = len(load)

                self.loaded_embeddings = np.zeros((words_to_load, self.emb_size))
                self.words = {}
                self.idx2words = {}
                self.ordered_words = []
                for i, line in enumerate(load):
                    if i >= words_to_load: 
                        break
                    s = line.split()
                    self.loaded_embeddings[i, :] = np.asarray(s[1:])
                    ###change###
                    self.words[s[0]] = i+1
                    self.idx2words[i+1] = s[0]
                    ###change###
                    self.ordered_words.append(s[0])
                ###change###
                self.words['UNK'] = len(self.words)+1
                self.idx2words[len(self.words)+1] = 'UNK'
                ###change###
                self.loaded_embeddings = np.vstack((self.loaded_embeddings, (2 * np.random.random_sample((1, self.emb_size)) - 1)))
        ###change this route after test
        self.train_set = self.load_sst_data(sst_home + 'train.tsv')
        self.validation_set = self.load_sst_data(sst_home + 'dev.tsv')
        self.test_set = self.load_sst_data(sst_home + 'test.tsv')
        # self.train_set = self.load_sst_data(sst_home)
        # self.validation_set = self.load_sst_data(sst_home)
        # self.test_set = self.load_sst_data(sst_home)

#       Only train return loaded_embeddings; the others would not catch the loaded_embeddings
        if(use_pretrain is True):
            self.matrix_train, self.loaded_embeddings = self.sentence2vec(self.train_set, self.ignore, self.emb_size, self.loaded_embeddings)
            self.matrix_validation, x = self.sentence2vec(self.validation_set, self.ignore, self.emb_size, self.loaded_embeddings)
            self.matrix_test, y = self.sentence2vec(self.test_set, self.ignore, self.emb_size, self.loaded_embeddings)
        else:
            self.matrix_train = self.datatosentence(self.train_set)
            self.matrix_validation = self.datatosentence(self.validation_set)
            self.matrix_test = self.datatosentence(self.test_set)
            return None
            
    def load_sst_data(self, path):
        '''
        load data and add reverse data
        '''
        data = []
        
        with open(path) as f:
            text = f.read().splitlines() 
            for i, line in enumerate(text):
                if i == 0: continue # ignore the first input
                example = {}
                text_1 = re.sub(r'\s*(\(\d)|(\))\s*', '', line)
                text_1 = re.split(r'\t+', text_1)
                example['text'] = text_1[0:-1]
                data.append(example)
                ###should open this part after testing###
                # reverse_example = [text_1[0], text_1[2], text_1[1]]
                # example['text'] = reverse_example
                # data.append(example)
                
        return data
    
    def train_test_sp(self, data_set):
        train_set, test_set = train_test_split(data_set, test_size=0.17)
        return train_set, test_set
    
    def ignore_OOV(self, toke, ignore, emb_size, loaded_embeddings, words): 
        '''decide how to deal with OOV'''
        '''if ignore is Flase, each OOV assign different vector'''
        '''if ignore is True, each OOV assign the same vector'''
        if ignore is False:
            ###len(words) should plus 1, since leaving 0 for padding###
            words[toke] = len(words)+1
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
                p_1, p_2, label = pair['text'][1], pair['text'][2],pair['text'][0]
                p_1_tok, p_2_tok = nltk.word_tokenize(p_1),nltk.word_tokenize(p_2)
                p_1_vec = []
                p_2_vec = []

                for toke_1 in p_1_tok:
                    toke_1 = toke_1.lower()
                    try:
                        ### return index instead returning vector###
                        p_1_vec.append(self.words[toke_1])
                    except KeyError:
                        x, loaded_embeddings = self.ignore_OOV(toke_1, ignore, emb_size, loaded_embeddings, self.words)
                        p_1_vec.append(x)

                for toke_2 in p_2_tok:
                    toke_2 = toke_2.lower()
                    try:
                        p_2_vec.append(self.words[toke_2.lower()])
                    except KeyError:
                        x, loaded_embeddings = self.ignore_OOV(toke_2, ignore, emb_size, loaded_embeddings, self.words)
                        p_2_vec.append(x)
                ### return index instead returning vector###
                sentence2vec.append([p_1,p_1_vec,p_2,p_2_vec,label])
            
            except IndexError:
                continue
        return sentence2vec, loaded_embeddings

    def datatosentence(self, data_set):
            '''add UNK to the embedding'''
            '''numpy would only copy the array and not modify the reference'''
            '''return the matrix which has already been change to vector'''
            sentence2vec = []
            print('I am here')
            for pair in data_set:
                try:#
                    p_1, p_2, label = pair['text'][1], pair['text'][2],pair['text'][0]
                    # p_1_tok, p_2_tok = nltk.word_tokenize(p_1),nltk.word_tokenize(p_2)
                    # p_1_vec = []
                    # p_2_vec = []
                    sentence2vec.append([p_1,p_2,label])
                except IndexError:
                    continue
            return sentence2vec
    
    def batch_iter(self, matrix, batch_size, use_pretrain, reuse = True):
        '''
        return a batch: list w and all have been already converted to tensor
        default batch size 32
        w[0,1,2,3,4]
        w[0]: p1_vector
        w[1]: p2_vector
        w[2]: p1_string
        w[3]: p2_string
        w[4]: label
        reuse: True or False, if False, the bacth would runout and throw 'StopIteration'
        '''
        start = -1 * batch_size
        dataset_size = len(matrix)
        order = list(range(dataset_size))
        random.shuffle(order)

        while True:
            if (use_pretrain is True):
                start += batch_size
                p1 = []
                p2 = []
                p1_2vec = []
                p2_2vec = []
                label = []
                p1_length_list = []
                p2_length_list = []
                
                if start > dataset_size - batch_size:
                    if reuse is True:
                        start = 0
                        random.shuffle(order)
                    else:
                        return

                batch_indices = order[start:start + batch_size]
                batch = [matrix[index] for index in batch_indices]
                for i,k in enumerate(batch):

                    p1_length_list.append(len(k[1]))
                    p2_length_list.append(len(k[3]))

                    p1.append(k[0])
                    p1_2vec.append(k[1])


                    p2.append(k[2])
                    p2_2vec.append(k[3])


                    label.append(int(k[4]))

                max_length_p1 = np.max(p1_length_list)
                max_length_p2 = np.max(p2_length_list)
                # print('max_length_p1', max_length_p1)
                # print('max_length_p2', max_length_p2)
                p1_pad_list = []
                p2_pad_list = []
                for i,k in enumerate(batch):
#                     p1_padded_vec = np.pad(np.array(k[1]), 
#                                             pad_width=(((0,max_length_p1-len(k[1]))),(0,0)), 
#                                             mode="constant", constant_values=0)
                    ###change the padding###
                    p1_padded_vec = np.pad(np.array(k[1]), 
                                            pad_width=((0,max_length_p1-len(k[1]))), 
                                            mode="constant", constant_values=0)
                    # print('p1_padded_vec', p1_padded_vec)

                    p1_pad_list.append(p1_padded_vec)

#                     p2_padded_vec = np.pad(np.array(k[3]), 
#                                             pad_width=(((0,max_length_p2-len(k[3]))),(0,0)), 
#                                             mode="constant", constant_values=0)
                    ###change the padding###
                    p2_padded_vec = np.pad(np.array(k[3]), 
                                            pad_width=((0,max_length_p2-len(k[3]))), 
                                            mode="constant", constant_values=0)
                    p2_pad_list.append(p2_padded_vec)

                yield [torch.from_numpy(np.asarray(p1_pad_list)), torch.from_numpy(np.asarray(p2_pad_list)), p1, p2, torch.LongTensor(label)]
            
            else:
                start += batch_size
                p1 = []
                p2 = []
                label = []

                if start > dataset_size - batch_size:
                    if reuse is True:
                        start = 0
                        random.shuffle(order)
                    else:
                        return
                    
                batch_indices = order[start:start + batch_size]
                batch = [matrix[index] for index in batch_indices]
                for i,k in enumerate(batch):
                    p1.append(k[0])
                    p2.append(k[1]) 
                    label.append(k[2])
                    
                yield [[], [], p1, p2, label]
                
    def eval_model_all(self, data_iter, model):
        """
        1. input variable_1, variable_2, model
        2. put (v1 and v2) into model 
        3. return the accuracy: this is for all batch

        """
        correct = 0
        total = 0

        model.eval()

        while True:
            try:
                p1_vec, p2_vec, p1_str, p2_str, label  = next(data_iter)

                p1_vec = Variable(p1_vec)
                p2_vec = Variable(p2_vec)

                outputs = model(p1_vec, p2_vec)
                predicted = (outputs.data > 0.5).long().view(-1)
                total += label.size(0)
                # print('label',label)
                # print(type(label))
                print('outputs.data',outputs.data)
                print(type(outputs.data))
                print('predicted',predicted)
                print(type(predicted))
                print('total',total)
                print(type(total))
                correct += (predicted == label).sum()

            except StopIteration:
                model.train()
                break
            model.train()

        return (100 * correct / total)
        


# In[6]:

# def eval_model_all(data_iter, model):
#     """
#     1. input variable_1, variable_2, model
#     2. put (v1 and v2) into model
#     3. return the accuracy: this is for all batch
    
#     """
#     correct = 0
#     total = 0
    
#     model.eval()
    
#     while True:
#         try:
#             p1_vec, p2_vec, p1_str, p2_str, label  = next(data_iter)
            
#             p1_vec = Variable(p1_vec)
#             p2_vec = Variable(p2_vec)
            
#             outputs = model(p1_vec, p2_vec)
#             predicted = (outputs.data > 0.5).long().view(-1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum()
            
#         except StopIteration:
#             model.train()
#             break
#         model.train()
        
#     return (100 * correct / total)


# #     for data, lengths, labels in data_iter:
# #         data_batch, length_batch, label_batch = Variable(data), Variable(lengths), Variable(labels)
# #         outputs = model(p1_vec, p2_vec)
# #         predicted = (outputs.data > 0.5).long().view(-1)
# #         total += labels.size(0)
# #         correct += (predicted == labels).sum()
# #     model.train()
# #     return (100 * correct / total)


# In[149]:

#### This is demo #####
# emb_size = 50
# pretrain_filename_path = 'glove.6B.50d.txt'
# document_filename = 'quora_duplicate_questions.tsv'
# ignore = True 
# batch_size = 32
# use_pretrain = True


# In[151]:

#### This is demo #####
# pt = pretrain(emb_size, pretrain_filename, document_filename, ignore,batch_size,use_pretrain)


# In[145]:

#### This is demo for using pretrain#####
# data_iter = pt.batch_iter(pt.matrix_validation, pt.batch_size, pt.use_pretrain)
# test = next(data_iter)
# p1_vec = test[0]
# p2_vec = test[1]
# p1_str = test[2]
# p2_str = test[3]
# label = test[4]
      
# # print(p1_vec.shape)
# # print(p2_vec)
# # print(p1_str)
# # print(p2_str)
# # print(label)


# In[146]:

# data_iter = pt.batch_iter(pt.matrix_validation, pt.batch_size, pt.use_pretrain)
# test = next(data_iter)


