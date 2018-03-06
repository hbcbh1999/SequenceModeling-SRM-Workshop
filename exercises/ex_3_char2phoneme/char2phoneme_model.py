import tensorflow as tf
import numpy as np


class Char2Phoneme():

    def __init__(self, emb_dim, char_vocab_size, phoneme_vocab_size, seqlen):

        tf.reset_default_graph()

        # define placeholders
        chars    = tf.placeholder(tf.int32, [None, seqlen], 'chars')
        phonemes = tf.placeholder(tf.int32, [None, seqlen], 'phonemes')

        # expose placeholders
        self.placeholders = { 'chars' : chars, 'phonemes' : phonemes }

        # infer dimensions of batch
        batch_size_, seq_len_ = tf.unstack(tf.shape(chars))

        # actual length of sequences considering padding
        seqlens = tf.count_nonzero(chars, axis=-1)

        # Character and Phoneme Embedding Matrices
        chE = tf.get_variable('chE', [char_vocab_size, emb_dim], tf.float32, 
                            initializer=tf.random_uniform_initializer(-0.01, 0.01)
                           )
        phE = tf.get_variable('phE', [1 + phoneme_vocab_size, emb_dim], tf.float32, 
                            initializer=tf.random_uniform_initializer(-0.01, 0.01)
                           ) # +1 corresponds to <START> token to signal "start generating"

        # <START> token
        PH_START = tf.tile([phE[-1]], [batch_size_, 1])

        # lookup character embedding
        chars_emb = tf.nn.embedding_lookup(chE, tf.transpose(chars))
        # break into iterable list
        #  batch_major to time_major
        chars_emb_list = chars_emb #tf.transpose(chars_emb, [1, 0, 2]))

        # encoder
        encoder_outputs = []
        with tf.variable_scope('encoder') as scope:
            enc_cell  = tf.nn.rnn_cell.LSTMCell(emb_dim)
            enc_state = enc_cell.zero_state(batch_size_, tf.float32)
            for i in range(seqlen):
                output, enc_state = enc_cell(chars_emb_list[i], enc_state)
                # accumulate outputs at each step
                encoder_outputs.append(output)

        # output projection parameters
        Wo = tf.get_variable('Wo', 
            shape=[emb_dim, phoneme_vocab_size], 
            dtype=tf.float32, 
            initializer=tf.random_uniform_initializer(-0.01, 0.01))

        bo = tf.get_variable('bo', 
            shape=[phoneme_vocab_size], 
            dtype=tf.float32, 
            initializer=tf.random_uniform_initializer(-0.01, 0.01))

        llogits = []
        with tf.variable_scope('decoder') as scope:
            dec_cell  = tf.nn.rnn_cell.LSTMCell(emb_dim, name='decoder_cell')
            dec_state = enc_state
            input_ = PH_START # start generation
            for i in range(seqlen):
                output, dec_state = dec_cell(input_, dec_state)
                logits = tf.matmul(output, Wo) + bo # tf.linear
                llogits.append(logits)
                prediction = tf.argmax(tf.nn.softmax(logits), axis=-1)
                input_ = tf.nn.embedding_lookup(phE, prediction)

        # stack list of logits
        #  convert to time_major
        logits = tf.transpose(tf.stack(llogits), [1, 0, 2])
        # probability distribution across vocabulary
        probs  = tf.nn.softmax(logits)
        # predictions
        preds  = tf.argmax(probs, axis=-1)

        # Cross Entropy
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, 
                labels=phonemes
                )
        # reduce to scalar
        loss = tf.reduce_mean(ce)

        # Accuracy
        accuracy = tf.reduce_mean(
                        tf.cast(
                            tf.equal(tf.cast(preds, tf.int32), phonemes),
                            tf.float32
                            )
                        )

        self.out = { 
                'loss'     : loss,
                'prob'     : probs,
                'pred'     : preds,
                'logits'   : logits,
                'accuracy' : accuracy
                }

        # training operation
        self.trainop = tf.train.AdamOptimizer().minimize(loss)


def rand_exec(model):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(model.out,
                feed_dict = {
                    model.placeholders['chars']     : np.random.randint(0, 10, [8, 16]),
                    model.placeholders['phonemes']  : np.random.randint(0, 10, [8, 16])
                    }
                )
                    

if __name__ == '__main__':

    model = Char2Phoneme(120, 100, 100, 16)
    out = rand_exec(model)

    print(out['pred'])
