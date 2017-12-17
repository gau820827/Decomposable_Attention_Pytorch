"""The file for evaluation."""
import os
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from decomposable_atten import embeddinglayer, DecomposableAttention
from pretrain_v1 import pretrain

use_cuda = torch.cuda.is_available()


def evaluate(embed_model, model, pt, dataset, batch_size):
    """The helper function for evaluation."""

    embed_model.eval()
    model.eval()
    lossf = nn.NLLLoss(size_average=False)

    correct = 0
    total = 0
    total_loss = 0
    print('Start Evaluating!')
    print('Validation Size: {}'.format(len(dataset)))

    threshold = 0.3

    data_iter = iter(pt.batch_iter(dataset, batch_size))

    for i in range(len(dataset) // batch_size):

        # catch the data
        p1_idx, p2_idx, _, _, label = next(data_iter)

        p1_idx = Variable(p1_idx)
        p2_idx = Variable(p2_idx)
        label = Variable(label)

        if use_cuda:
            p1_idx = p1_idx.cuda()
            p2_idx = p2_idx.cuda()
            label = label.cuda()

        # Feed to the network
        p1_emb, p2_emb = embed_model(p1_idx, p2_idx)
        out = model(p1_emb, p2_emb)

        # print(label)
        # print(out)

        loss = lossf(out, label)
        total_loss += loss.data[0]

        prob = torch.exp(out)
        predicted = Variable(torch.LongTensor([1 if l[1].data[0] >= threshold else 0 for l in prob]))

        # _, predicted = torch.max(out, dim=1)
        total += p1_idx.size()[0]

        # print(predicted)

        correct += torch.sum((label == predicted), dim=0).data[0]

        print('Correct Labels: {}/{}'.format(correct, (i + 1) * batch_size))

    print('Valid Loss: {}, Acc: {}'.format(total_loss / float(total),
                                           correct / float(total)))


def load_model(model, model_src, mode='eval'):
    state_dict = torch.load(model_src, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    if mode == 'eval':
        model.eval()
    else:
        model.train()

    return model

# Configuration
embedding_dim = 300
hidden_dim = 200
output_dim = 2

pretrain_emb_size = embedding_dim
pretrain_filename = '../pretrainvec/glove/glove.6B.300d.txt'
document_filename = '../data/'
batch_size = 64
usepretrain = True
embedding_style = 'word'
trainable = True

embed_model_src = './embed_glove_pretrain_trainable_400000'
model_src = './glove_pretrain_trainable_400000'


temp_file = 'trainable_6B_glove.pkl'

# Load the dataset
if temp_file in os.listdir():
    pt = pickle.load(open(temp_file, 'rb'))
    print('Load the temp file...')
else:
    pt = pretrain(pretrain_emb_size, pretrain_filename, document_filename,
                  batch_size, usepretrain, embedding_style)
    pickle.dump(pt, open(temp_file, 'wb'))
    print('Save the temp file...')

# Build the model
if usepretrain is True:
    embed_model = embeddinglayer(embedding_dim,
                                 loaded_embeddings=pt.loaded_embeddings,
                                 vocab_size=pt.emblang.n_words)
    model = DecomposableAttention(embedding_dim, hidden_dim, output_dim)
else:
    embed_model = embeddinglayer(embedding_dim,
                                 vocab_size=pt.corpuslang.n_words)
    model = DecomposableAttention(embedding_dim, hidden_dim, output_dim)

embed_model.double()
model.double()

if use_cuda:
    embed_model.cuda()
    model.cuda()

# Load the model
if trainable is True or usepretrain is False:
    embed_model = load_model(embed_model, embed_model_src)
model = load_model(model, model_src)

evaluate(embed_model, model, pt, pt.matrix_valid, batch_size)
