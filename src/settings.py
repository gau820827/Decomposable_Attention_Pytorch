"""This is the file for settings."""
PRETRAIN_FILE = '../pretrainvec/glove/glove.6B.300d.txt'
DOCUMENT_FILE = '../data/'

# Parameter for training
LR = 0.01
ITER_TIME = 10000
BATCH_SIZE = 1

# Parameter for display
GET_LOSS = 1
SAVE_MODEL = 1000
OUTPUT_FILE = 'default'

# Parameter for core model
EMBEDDING_SIZE = 300
HIDDEN_SIZE = 200
OUTPUT_DIM = 2
DROPOUT = 0.2
