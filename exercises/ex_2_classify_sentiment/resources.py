import os


# data directory
RT_DATA='../../data/RT-reviews/rt_reviews.tsv'
SM_DATA='../../data/socialmedia/training.txt'

# sentiment labels
sentiment   = [ 'negative', 'somewhat negative', 'neutral', 'somewhat positive', 'positive' ]
# sentiment tag to index lookup
sentiment2i = { s:i for s,i in enumerate(sentiment) }

sm_sentiment   = [ '-ve', '+ve' ]
# sentiment tag to index lookup
sm_sentiment2i = { s:i for s,i in enumerate(sentiment) }

# UNK (unknown) token
UNK = '<UNK>'
# PAD (padding) token
PAD = '<PAD>'

# vocabulary ceiling
MAX_VOCAB_SIZE    = 3000
SM_MAX_VOCAB_SIZE = 800
