"""This is the core model of decomposable attention."""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np

from charNgram import NgramChar
from pretrain_v1 import pretrain

from settings import PRETRAIN_FILE, DOCUMENT_FILE
from settings import LR, ITER_TIME, BATCH_SIZE
from settings import GET_LOSS, SAVE_MODEL, OUTPUT_FILE
from settings import EMBEDDING_SIZE, HIDDEN_SIZE, OUTPUT_DIM, DROPOUT
from settings import EMBEDDING_STYLE, TRAIN_EMBED


use_cuda = torch.cuda.is_available()


class embeddinglayer(nn.Module):
    """The layer for embedding sentences"""
    def __init__(self, embedding_dim, loaded_embeddings=None, vocab_size=0, train_embed=TRAIN_EMBED,
                 mode='glove'):
        super(embeddinglayer, self).__init__()
        assert vocab_size != 0, "No embedding words, something must be wrong!"

        # Choose word_embedding model
        self.embedding_model = mode

        print('Load {} vocabularies...'.format(vocab_size))
        print('Size of embedding layer: {}'.format(loaded_embeddings.shape))

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if loaded_embeddings is not None:
            self.embeddings.weight = nn.Parameter(torch.from_numpy(loaded_embeddings).float())
        else:
            self.init_weights()

        # Whether to train the embedding layer or not
        self.train_embed = train_embed
        self.embeddings.weight.requires_grad = train_embed

    def forward(self, x1, x2):
        # Get the embedding
        x1 = self.embeddings(x1)
        x2 = self.embeddings(x2)
        return x1, x2

    def init_weights(self):
        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)


class DecomposableAttention(nn.Module):
    """Core model of decomposable_attention.

    Here, we can choose the mode of word embeddings:
    1. Glove: Use glove word-embeddings
    2. char_n-gram: Use char n-gram encodings

    """
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(DecomposableAttention, self).__init__()

        # Layers of the fucking model
        self.projection = nn.Linear(embedding_dim, hidden_dim)

        self.attend = _Attend(hidden_dim)
        self.compare = _Compare(hidden_dim)
        self.aggregate = _Aggregate(hidden_dim)

        self.final_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x1, x2):
        """ The forward process of the core model.

        If we're not using pre-trained word-embeddings, we assume that
        the inputs are two batch sequences of indexings.

        Else, the inputs should be two batch sequnces of vectors.

        Ex. x1.size = (batch_size * sentence_length * embedding_size)

        """

        # Project the embedding vocabulary
        x1 = self.projection(x1)
        x2 = self.projection(x2)

        # Feed to the model
        alpha, beta = self.attend(x1, x2)
        v1, v2 = self.compare(alpha, beta, x1, x2)
        out = self.aggregate(v1, v2)
        out = self.final_linear(out)
        return F.log_softmax(out)


class _Attend(nn.Module):
    """Attend part for the model."""
    def __init__(self, hidden_dim):
        super(_Attend, self).__init__()
        self.hidden_size = hidden_dim
        self.f = _F(hidden_dim, hidden_dim, hidden_dim)

    def forward(self, x1, x2):

        len1 = x1.size()[1]
        len2 = x2.size()[1]

        # Unnormalized attend weights
        f1 = self.f(x1)
        f2 = self.f(x2)

        # Normalized attend weights
        f1 = f1.view(-1, len1, self.hidden_size)
        # batch_size x len1 x hidden_size
        f2 = f2.view(-1, len2, self.hidden_size)
        # batch_size x len2 x hidden_size

        score1 = torch.bmm(f1, torch.transpose(f2, 1, 2))
        # e_{ij} batch_size x len1 x len2
        prob1 = F.softmax(score1.view(-1, len2)).view(-1, len1, len2)
        # batch_size x len1 x len2

        score2 = torch.transpose(score1.contiguous(), 1, 2)
        score2 = score2.contiguous()
        # e_{ji} batch_size x len2 x len1
        prob2 = F.softmax(score2.view(-1, len1)).view(-1, len2, len1)
        # batch_size x len2 x len1

        beta = torch.bmm(prob1, x2)
        alpha = torch.bmm(prob2, x1)

        return alpha, beta


class _Compare(nn.Module):
    """Compare part for the model"""
    def __init__(self, hidden_dim):
        super(_Compare, self).__init__()
        self.g = _F(hidden_dim * 2, hidden_dim, hidden_dim)

    def forward(self, alpha, beta, x1, x2):
        # Seperately compare the aligned phrases

        v1 = torch.cat((x1, beta), dim=2).view(-1, hidden_dim * 2)
        # batch_size x len1 x (hidden_size x 2)
        v2 = torch.cat((x2, alpha), dim=2).view(-1, hidden_dim * 2)
        # batch_size x len2 x (hidden_size x 2)

        v1 = self.g(v1)
        v2 = self.g(v2)

        v1 = v1.view(-1, x1.size()[1], hidden_dim)
        v2 = v2.view(-1, x2.size()[1], hidden_dim)

        return v1, v2


class _Aggregate(nn.Module):
    """Aggregate part for the model"""
    def __init__(self, hidden_dim):
        super(_Aggregate, self).__init__()
        self.h = _F(hidden_dim * 2, hidden_dim, hidden_dim)

    def forward(self, v1, v2):
        v1 = torch.sum(v1, dim=1)
        v2 = torch.sum(v2, dim=1)

        out = self.h(torch.cat((v1, v2), dim=1))
        return out


class _F(nn.Module):
    """_F for the feed-forward NN"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(_F, self).__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden3 = nn.Linear(hidden_dim, output_dim)
        self.init_weights()

    def forward(self, x):
        out = self.hidden1(x)
        out = F.relu(out)
        out = F.dropout(out, p=DROPOUT)
        out = self.hidden2(out)
        out = F.relu(out)
        out = F.dropout(out, p=DROPOUT)
        out = self.hidden3(out)
        out = F.relu(out)
        return out

    def init_weights(self):
        initrange = 0.1
        lin_layers = [self.hidden1, self.hidden2, self.hidden3]

        for layer in lin_layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.fill_(0)


## TODO: Evaluation Loop

# def evaluate(model, data_iter):
#     """The helper function for evaluation."""
#     model.eval()
#     correct = 0
#     total = 0
#     for i in range(len(data_iter)):
#         vectors, labels = next(data_iter[i])
#         vectors = Variable(torch.stack(vectors).squeeze())
#         labels = torch.stack(labels).squeeze()
#         output = model(vectors)
#         _, predicted = torch.max(output.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum()
#     return correct / float(total)


def train_iter(embed_model, model, data_iter, iter_time):
    """Training loop."""
    if TRAIN_EMBED is True:
        embed_optimizer = optim.Adagrad(embed_model.parameters(), lr=LR)
    optimizer = optim.Adagrad(model.parameters(), lr=LR)
    lossf = nn.NLLLoss()

    print('Start Training!')
    total_loss = 0
    for iteration in range(1, iter_time + 1):
        # catch the data
        p1_vec, p2_vec, p1_str, p2_str, label = next(data_iter)
        p1_vec = Variable(p1_vec)
        p2_vec = Variable(p2_vec)
        label = Variable(label)

        if use_cuda:
            p1_vec = p1_vec.cuda()
            p2_vec = p2_vec.cuda()
            label = label.cuda()

        if TRAIN_EMBED is True:
            embed_model.zero_grad()
        model.zero_grad()

        # Feed to the network
        p1_emb, p2_emb = embed_model(p1_vec, p2_vec)
        out = model(p1_emb, p2_emb)

        loss = lossf(out, label)
        loss.backward()

        if TRAIN_EMBED is True:
            embed_optimizer.step()
        optimizer.step()

        total_loss += loss.data[0]

        if iteration % GET_LOSS == 0:
            print("Iter %i : Avg Loss %f" % (iteration, total_loss / GET_LOSS))
            total_loss = 0

        if iteration % SAVE_MODEL == 0:
            if TRAIN_EMBED is True:
                torch.save(embed_model.state_dict(), "embed_{}_{}".format(OUTPUT_FILE, iteration))
            torch.save(model.state_dict(), "{}_{}".format(OUTPUT_FILE, iteration))
            print("Save the model at iter {}".format(iteration))

        # if step % 100 == 0:
        #     print("Step %i; Loss %f; Train acc: %f; Dev acc %f"
        #           % (step, loss.data[0], evaluate(model, train_eval_iter), evaluate(model, dev_iter)))


def train(embedding_dim, hidden_dim, output_dim, batch_size,
          usepretrain, pretrain_emb_size, pretrain_filename,
          document_filename, iter_time, embedding_style):
    """The core part of training the model.

    First, we will initialize the model based on given parameters.
    Second, we will decide whether to use pre-train word vector or not.
    Third, we load the dataset.

    """
    pt = pretrain(pretrain_emb_size, pretrain_filename, document_filename,
                  batch_size, usepretrain, embedding_style)
    data_iter = pt.batch_iter(pt.matrix_train, pt.batch_size)

    print("Data loaded......")

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

    train_iter(embed_model, model, data_iter, iter_time)


# These are function for testing
def mini_train(pair_a, pair_b, target, model):
    optimizer = optim.Adam(model.parameters())
    lossf = nn.CrossEntropyLoss()

    model.zero_grad()
    out = model(pair_a, pair_b)

    loss = lossf(out, target)
    loss.backward()
    optimizer.step()

    print("Minitest Complete, loss = {}".format(loss.data))


def mini_test():
    """A small function to test the core model."""
    embedding_dim = 50
    hidden_dim = 200
    output_dim = 2

    # minitest uses a batch of 3 pairs with length 10 and 5
    a = np.random.random_sample((2, 7, embedding_dim))
    b = np.random.random_sample((2, 8, embedding_dim))
    a = Variable(torch.DoubleTensor(a))
    b = Variable(torch.DoubleTensor(b))
    target = Variable(torch.LongTensor([1, 0]))

    model = DecomposableAttention(embedding_dim, hidden_dim, output_dim)

    # Set the parameter to double type
    model.double()
    mini_train(a, b, target, model)


def charngram_test():
    """A small function to test the core model with char ngram embedding"""
    embedding_dim = 50
    hidden_dim = 200
    output_dim = 2

    ngram = NgramChar('ngram')

    # minitest uses a batch of 2 pairs
    a = [['I', 'was', 'astonished'], ['I', 'was', 'bor']]
    b = [['She', 'love'], ['He', 'hate']]

    # Encode them
    max_pairs = 0
    index_a = []
    index_b = []
    for sentence in a:
        index_a.append([])
        for token in sentence:
            index_a[-1].append(ngram.getword2index(token))
            max_pairs = max(max_pairs, len(index_a[-1][-1]))

    for sentence in b:
        index_b.append([])
        for token in sentence:
            index_b[-1].append(ngram.getword2index(token))
            max_pairs = max(max_pairs, len(index_b[-1][-1]))

    # Add paddings
    for sentence in index_a:
        for tokens in sentence:
            paddings = [0 for i in range(max_pairs - len(tokens))]
            tokens += paddings

    for sentence in index_b:
        for tokens in sentence:
            paddings = [0 for i in range(max_pairs - len(tokens))]
            tokens += paddings

    a = Variable(torch.LongTensor(index_a))
    b = Variable(torch.LongTensor(index_b))
    target = Variable(torch.LongTensor([1, 0]))

    model = DecomposableAttention(embedding_dim, hidden_dim, output_dim,
                                  mode='charngram', vocab_size=ngram.n_chars)

    mini_train(a, b, target, model)


def showconfig():
    """Display the configuration."""
    print("EMBEDDING_SIZE = {}\nLR = {}\nITER_TIME = {}\nBATCH_SIZE = {}".format(
        EMBEDDING_SIZE, LR, ITER_TIME, BATCH_SIZE))
    print("DROPOUT_RATE = {}".format(DROPOUT))
    print("USE_PRETRAIN = {}\nOUTPUT_FILE = {}".format(PRETRAIN_FILE, OUTPUT_FILE))
    print("EMBEDDING_STYLE = {}\nTRAIN_EMBED = {}".format(EMBEDDING_STYLE, TRAIN_EMBED))


if __name__ == '__main__':

    # Parameters for loading pre-train word
    pretrain_filename = PRETRAIN_FILE
    document_filename = DOCUMENT_FILE

    # Parameter for core model
    embedding_style = EMBEDDING_STYLE
    embedding_dim = EMBEDDING_SIZE
    hidden_dim = HIDDEN_SIZE
    output_dim = OUTPUT_DIM

    # Parameter for data batching
    iter_time = ITER_TIME
    batch_size = BATCH_SIZE

    showconfig()

    usepretrain = False if pretrain_filename is None else True
    pretrain_emb_size = embedding_dim
    test = False

    # Function Test
    if test is True:
        mini_test()
        charngram_test()

    # Main train loop
    train(embedding_dim, hidden_dim, output_dim, batch_size, usepretrain,
          pretrain_emb_size, pretrain_filename, document_filename, iter_time,
          embedding_style)
