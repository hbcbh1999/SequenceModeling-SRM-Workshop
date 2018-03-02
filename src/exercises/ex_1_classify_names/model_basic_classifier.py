import tensorflow as tf
import numpy as np


class BasicClassifier():

    def __init__(self, wdim, hdim, vocab_size, num_labels):

        tf.reset_default_graph()

        # define placeholders
        name  = tf.placeholder(tf.int32, [None, None], name='name')
        labels = tf.placeholder(tf.int32, [None, ], name='label'  )

        # expose placeholders
        self.placeholders = { 'name' : name, 'label' : labels }

        # infer dimensions of batch
        batch_size_, seq_len_ = tf.unstack(tf.shape(name))

        # actual length of sequences considering padding
        seqlens = tf.count_nonzero(name, axis=-1)

        # word embedding
        wemb = tf.get_variable(shape=[vocab_size, wdim], dtype=tf.float32,
                        initializer=tf.random_uniform_initializer(-0.01, 0.01),
                        name='word_embedding')

        with tf.variable_scope('encoder') as scope:
            _ , (fsf, fsb) = tf.nn.bidirectional_dynamic_rnn( 
                    tf.nn.rnn_cell.LSTMCell(hdim), 
                    tf.nn.rnn_cell.LSTMCell(hdim), 
                    inputs=tf.nn.embedding_lookup(wemb, name), 
                    sequence_length=seqlens, 
                    dtype=tf.float32)

        # output projection parameters
        Wo = tf.get_variable('Wo', 
                shape=[hdim*2, num_labels], 
                dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-0.01, 0.01))

        bo = tf.get_variable('bo', 
                shape=[num_labels,], 
                dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-0.01, 0.01))

        logits = tf.matmul(tf.concat([fsf.c, fsb.c], axis=-1), Wo) + bo

        probs  = tf.nn.softmax(logits)
        preds  = tf.argmax(probs, axis=-1)

        # Cross Entropy
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(ce)

        # Accuracy
        accuracy = tf.reduce_mean(
                        tf.cast(
                            tf.equal(tf.cast(preds, tf.int32), labels),
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
                    model.placeholders['name' ]  : np.random.randint(0, 10, [8, 10]),
                    model.placeholders['label']  : np.random.randint(0, 10, [8, ]  )
                    }
                )
                    

if __name__ == '__main__':

    model = BasicClassifier(10, 10, 10, 10)
    out = rand_exec(model)

    print(out['loss'], out['accuracy'])
