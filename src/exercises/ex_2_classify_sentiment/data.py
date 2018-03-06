from nltk import word_tokenize, FreqDist
import resources as R
import os


def read_file(filename):
    """
    TODO

    Args : 

    Returns :

    """
    with open(filename) as f:
        texts, sentiments = [], []
        sent_indices = []
        for line in f.readlines()[1:]:
            _, sent_index, text, sentiment = line.replace('\n', '').split('\t')
            if sent_index not in sent_indices:
                texts.append(text)
                sentiments.append(sentiment)
                sent_indices.append(sent_index)

    return texts, sentiments

def create_dataset():
    """
    TODO

    Args:

    Returns:

    """
    texts, sentiments = read_file(R.DATA)
    return index_samples(texts, sentiments)

def build_vocabulary(texts, max_vocab_size):
    """
    TODO

    Add 0 for padding

    Args:

    Returns:

    """
    words = word_tokenize(' '.join(texts))
    freq_dist = FreqDist(words)
    return [ R.PAD, R.UNK ] + sorted(set(words), key=lambda w : freq_dist[w], reverse=True)[:max_vocab_size]

def index_samples(texts, sentiments, max_vocab_size=R.MAX_VOCAB_SIZE):
    """
    TODO

    Args:

    Returns:

    """
    vocab = build_vocabulary(texts, max_vocab_size)
    w2i  = { w:i for i,w in enumerate(vocab) }
    return {
            'raw_samples' : [ (t,s) for t,s in zip(texts, sentiments) ],
            'samples'     : [ ([ word2index(w, w2i) for w in word_tokenize(text) ], sentiment) 
                for text, sentiment in zip(texts, sentiments) ],
             'vocab'      : vocab,
             'w2i'        : w2i
             }

def index2word(i, vocab):
    return vocab[i] if i < len(vocab) else R.UNK

def word2index(w, w2i):
    return w2i[w] if w in w2i else w2i[R.UNK]
