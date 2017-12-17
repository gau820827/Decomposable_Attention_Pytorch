"""This is the file for settings."""
PRETRAIN_FILE = '../pretrainvec/glove/glove.6B.300d.txt'
DOCUMENT_FILE = '../data/'

# Parameter for training
LR = 0.1
ITER_TIME = 300000
BATCH_SIZE = 64

# Parameter for display
GET_LOSS = 10
SAVE_MODEL = 10000
OUTPUT_FILE = 'no_pretrain'

# Parameter for core model
TRAIN_EMBED = False
EMBEDDING_STYLE = 'word'
EMBEDDING_SIZE = 300
HIDDEN_SIZE = 200
OUTPUT_DIM = 2
DROPOUT = 0.1
