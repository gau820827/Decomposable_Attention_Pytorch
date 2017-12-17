"""The file to load the data."""
import numpy as np
import re
import nltk
import random

import torch
from torch.autograd import Variable

from charNgram import Lang, NgramChar


class pretrain():

    '''
    Document file is located(path) in the same file, could be modified if you need to do so
    '''

    def __init__(self, emb_size, pretrain_filename, document_filename,
                 batch_size, use_pretrain, embedding_style):
        sst_home = document_filename
        embedding_style = embedding_style

        self.emb_size = emb_size
        self.pretrain_filename = pretrain_filename
        self.batch_size = batch_size
        self.use_pretrain = use_pretrain

        # The information for word embedding.
        self.corpuslang = Lang('word')
        self.emblang = Lang('emb')
        self.charlang = NgramChar('charn')

        self.train_set = self.load_sst_data(sst_home + 'train.tsv')
        self.validation_set = self.load_sst_data(sst_home + 'dev.tsv')
        self.test_set = self.load_sst_data(sst_home + 'test.tsv')

        # After reading the dataset, we could load the word vector
        if use_pretrain is True:
            self.loaded_embeddings = self.load_wordembeddings(self.emb_size)

            # Indexing the words using embedding dictionary
            self.matrix_train = self.sentence2vec(self.train_set, self.emblang)
        else:
            # Indexing the words using corpus dictionary
            if embedding_style == 'word':
                self.matrix_train = self.sentence2vec(self.train_set, self.corpuslang)
            elif embedding_style == 'char':
                self.matrix_train = self.sentence2vec(self.train_set, self.charlang)
            else:
                raise ValueError('Wrong embedding style, please specify \'word\' or \'char\'')

    def load_wordembeddings(self, emb_size):
        """Load pre-train word embeddings."""
        with open(self.pretrain_filename) as f:

            load = f.readlines()
            loaded_embeddings = np.zeros((len(load), emb_size))

            # Read the embddings
            step = 0
            for line in load:
                s = line.split()

                # Check if the word is in the corpus
                word = ' '.join(s[:-self.emb_size])
                if word not in self.corpuslang.word2index:
                    continue

                # Add to embedding dictionary and get the vector
                self.emblang.addword(word)
                loaded_embeddings[step, :] = np.asarray(s[-emb_size:])
                step += 1

            # Trim the matrix size to the size of used words
            length = self.emblang.n_words - 2  # Not Count <PAD> and <UNK>
            loaded_embeddings = loaded_embeddings[:length, :]
            loaded_embeddings = np.vstack([np.random.uniform(-1, 1, (2, emb_size)),
                                           loaded_embeddings])
            return loaded_embeddings

    def load_sst_data(self, path):
        """Load the quora data and Update the dictionary."""
        data = []

        def updatelang(tokens):
            """A helper to update vocabulary dictionary."""
            for token in tokens:
                self.corpuslang.addword(token)

        with open(path) as f:
            text = f.read().splitlines()
            for i, line in enumerate(text):
                text_1 = re.sub(r'\s*(\(\d)|(\))\s*', '', line)
                text_1 = re.split(r'\t+', text_1)

                # Work Around with different format
                if 'train' in path:
                    label = text_1[3]
                else:
                    label = text_1[0]

                p1_tok = [t.lower() for t in nltk.word_tokenize(text_1[1])]
                p2_tok = [t.lower() for t in nltk.word_tokenize(text_1[2])]

                # For training we should update the dictionary
                if 'train' in path:
                    updatelang(p1_tok)
                    updatelang(p2_tok)

                # Get the tokenized sentences
                data.append([label, p1_tok, p2_tok])

                # Reverse the question pair
                data.append([label, p2_tok, p1_tok])

        return data

    def sentence2vec(self, data_set, lang):
        """This function will get the pairs and translate to indexing."""
        sentence2vec = []

        # Read through the data
        for pair in data_set:
            label, p1, p2 = pair[0], pair[1], pair[2]

            # Translate the token to index
            p1_idx = [lang.getword2index(token) for token in p1]
            p2_idx = [lang.getword2index(token) for token in p2]

            sentence2vec.append([p1, p1_idx, p2, p2_idx, label])

        return sentence2vec

    def batch_iter(self, matrix, batch_size, reuse=True):

        '''
        return a batch: list w and all have been already converted to tensor
        default batch size 32
        w[0,1,2,3,4]
        w[0]: p1_token
        w[1]: p2_idx
        w[2]: p2_token
        w[3]: p2_idx
        w[4]: label
        reuse: True or False, if False, the bacth would runout and throw 'StopIteration'
        '''
        start = -1 * batch_size
        dataset_size = len(matrix)
        order = list(range(dataset_size))
        random.shuffle(order)

        while True:
            # Make some batch
            start += batch_size

            if start > dataset_size - batch_size:

                if reuse is True:
                    start = 0
                    random.shuffle(order)
                else:
                    return

            batch_indices = order[start:start + batch_size]
            batch = [matrix[index] for index in batch_indices]

            # Get the batch data
            p1 = [d[0] for d in batch]
            p1_idx = [d[1] for d in batch]
            p2 = [d[2] for d in batch]
            p2_idx = [d[3] for d in batch]
            label = [int(d[4]) for d in batch]

            # Add paddings
            max_length_p1 = len(max(p1_idx, key=len))
            max_length_p2 = len(max(p2_idx, key=len))

            p1_pad_list = []
            p2_pad_list = []

            for i, k in enumerate(batch):
                p1_padded_vec = np.pad(np.array(k[1]),
                                       pad_width=((0, max_length_p1 - len(k[1]))),
                                       mode="constant", constant_values=0)

                p1_pad_list.append(p1_padded_vec)

                p2_padded_vec = np.pad(np.array(k[3]),
                                       pad_width=((0, max_length_p2 - len(k[3]))),
                                       mode="constant", constant_values=0)

                p2_pad_list.append(p2_padded_vec)

            yield [torch.from_numpy(np.asarray(p1_pad_list)),
                   torch.from_numpy(np.asarray(p2_pad_list)),
                   p1, p2, torch.LongTensor(label)]

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
