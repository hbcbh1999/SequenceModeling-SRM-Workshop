from sentiment_classifier import SentimentClassifier
from data import create_dataset, index2word
import resources as R

from random import shuffle, sample

import tensorflow as tf
import numpy as np

from nltk import word_tokenize
from tqdm import tqdm

PAD = 0


def seq_maxlen(seqs):
    """
    Maximum length of max-length sequence 
     in a batch of sequences
    Args:
        seqs : list of sequences
    Returns:
        length of the lengthiest sequence
    """
    return max([len(seq) for seq in seqs])

def pad_seq(seqs, maxlen=0, PAD=PAD, truncate=False):

    # pad sequence with PAD
    #  if seqs is a list of lists
    if type(seqs[0]) == type([]):

        # get maximum length of sequence
        maxlen = maxlen if maxlen else seq_maxlen(seqs)

        def pad_seq_(seq):
            if truncate and len(seq) > maxlen:
                # truncate sequence
                return seq[:maxlen]

            # return padded
            return seq + [PAD]*(maxlen-len(seq))

        seqs = [ pad_seq_(seq) for seq in seqs ]
    
    return seqs

def vectorize_batch(batch):
    return {
        'review'  : np.array(pad_seq([ review for review, label in batch ])),
        'label'   : np.array([ label for review, label in batch ])
    }

def train_model(model, trainset, testset, batch_size=200, max_acc=.90):
    epochs = 100
    iterations = len(trainset)//batch_size

    # fetch default session
    sess = tf.get_default_session()
    
    for j in range(epochs):
        loss = []
        for i in tqdm(range(iterations)):
            # fetch next batch
            batch = vectorize_batch(trainset[i*batch_size : (i+1)*batch_size])
            #print(set(list(batch['label'])))
            _, out = sess.run([ model.trainop,  model.out ],
                    feed_dict = {
                        model.placeholders['review']  : batch['review'],
                        model.placeholders['label' ]  : batch['label' ],
                        }
                    )
            loss.append(out['loss'])

        print('<train> [{}]th epoch : loss : {}'.format(j, np.array(out['loss']).mean()))
        # evaluate and calc accuracy
        accuracy = evaluate(model, testset)
        print('\t<eval > accuracy : {}'.format(accuracy))

        if accuracy >= max_acc :
            print('<train> accuracy > MAX_ACC; Exit training...')
            return

def evaluate(model, testset, batch_size=32):
    iterations = len(testset)//batch_size

    # fetch default session
    sess = tf.get_default_session()

    accuracy = []
    for i in range(iterations):
        # fetch next batch
        batch = vectorize_batch(testset[i*batch_size : (i+1)*batch_size])
        out = sess.run(model.out,
                feed_dict = {
                    model.placeholders['review']  : batch['review' ],
                    model.placeholders['label' ]  : batch['label'  ]
                    }
                )
        accuracy.append(out['accuracy'])

    return np.array(accuracy).mean()

def predict(model, batch, top_k=3):
    sess = tf.get_default_session()
    out = sess.run(model.out,
            feed_dict = {
                model.placeholders['review']  : batch['review' ],
                model.placeholders['label' ]  : batch['label'  ]
                }
            )
    preds = []
    for prob in out['prob']:
        preds.append( sorted([ (i,p) for i,p in enumerate(prob) ],
            key=lambda x : x[1], reverse=True)[:top_k] )

    return [ [ '{} : {}'.format(R.sentiment[i], p) for i,p in pred ]
            for pred in preds ]

'''
def interact(model, validset, lookup, n=3):

    print('\n<interact>\n\n')
    #ui = 'y'
    while input() is not 'q':
        samples = sample(validset, n)
        preds = predict(model, vectorize_batch(samples))
        for i, (review, label) in enumerate(samples):
            print(' '.join([ index2word(i, lookup) for i in review ]), 'is', R.sentiment[label])
            for pred in preds[i]:
                print('\t', pred)
     
'''

if __name__ == '__main__':

    dataset = create_dataset()

    samples = dataset['samples']
    shuffle(samples)
    trainlen = int(len(samples)*0.80)
    testlen  = int(len(samples)*0.10)
    validlen = testlen
    # split
    len_sorted = lambda l : sorted(l, key=lambda x : len(x[0]))
    trainset = len_sorted(samples[:trainlen])
    testset  = len_sorted(samples[trainlen:trainlen + testlen])
    validset = len_sorted(samples[trainlen + testlen : ])

    vocab = dataset['vocab']

    model = SentimentClassifier(
            wdim = 100, 
            hdim = 100,
            vocab_size = len(vocab),
            num_labels = len(R.sentiment)
            )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_model(model, trainset, testset, batch_size=32, max_acc=0.80)
        #interact(model, validset, vocab)
