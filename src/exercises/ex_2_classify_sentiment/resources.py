import os


# data directory
DATA='../../data/RT-reviews/rt_reviews.tsv'

# sentiment labels
sentiment   = [ 'negative', 'somewhat negative', 'neutral', 'somewhat positive', 'positive' ]
# sentiment tag to index lookup
sentiment2i = { s:i for s,i in enumerate(sentiment) }

# UNK (unknown) token
UNK = '<UNK>'
# PAD (padding) token
PAD = '<PAD>'

# vocabulary ceiling
MAX_VOCAB_SIZE = 3000
