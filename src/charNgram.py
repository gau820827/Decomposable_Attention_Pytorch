"""This is the module for char n-gram encoding."""


class Lang():
    """A class to summarize encoding information.

    This class will build three dicts:
    word2index, word2count, and index2word for
    embedding information. Once a set of data is
    encoded, we can transform it to corrsponding
    indexing use the word2index, and map it back
    using index2word.

    Attributes:
        word2index: A dict mapping word to index.
        word2count: A dict mapping word to counts in the corpus.
        index2word: A dict mapping index to word.

    """

    def __init__(self, name):
        """Init Lang with a name."""
        self.name = name
        self.word2index = {'<PAD>': 0, '<UNK>': 1}
        self.word2count = {'<PAD>': 0, '<UNK>': 1}
        self.index2word = {0: '<PAD>', 1: '<UNK>'}
        self.n_words = 2  # Count <PAD> and <UNK>

    def addword(self, word):
        """Add a word to the dict."""
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def getword2index(self, word):
        """Get the index of the word"""
        try:
            return self.word2index[word]
        except KeyError:
            return self.word2index['<UNK>']


class NgramChar():
    """The class for extracting char n-gram information.

    This class is for extracting char n-gram information,
    specifically speaking, count all pairs of words and
    build the dictionary of those n-gram chars.

    Attributes:
        chars2index: A dict mapping char to index.
        index2chars: A dict mapping index to char.
        ngram: An integer indicates the maximum n-gram length.
        n_chars: An integer indicates the number of chars.

    """

    def __init__(self, name, n=5):
        """Init ngramChar with a name."""
        super(NgramChar, self).__init__()
        self.name = name
        self.chars2index = {'<PAD>': 0, '<UNK>': 1}
        self.index2chars = {0: '<PAD>', 1: '<UNK>'}
        self.ngram = n
        self.n_chars = 2

    def getchars(self, chars):
        """Get the index of chars."""
        if chars not in self.chars2index:
            self.chars2index[chars] = self.n_chars
            self.n_chars += 1
        return self.chars2index[chars]

    def getword2index(self, word):
        """Get the n-gram indexing of a word."""
        chars_vec = []
        for l in range(1, self.ngram + 1):
            if l > len(word):
                break

            for st in range(len(word) - l + 1):
                chars_vec.append(self.getchars(word[st:st + l]))
        return chars_vec


def minitest():
    """A minitest function for n-gram chars."""
    ngram = NgramChar('test')
    tokens = ['abc', 'bcd', 'wordembeddings', 'similarity']
    for token in tokens:
        print(ngram.getword2index(token))
    print("Counts of all chars: {}".format(ngram.n_chars))


if __name__ == '__main__':
    minitest()
