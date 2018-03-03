import tensorflow as tf
import numpy as np


class NameGenerator():

    def __init__(self, wdim, hdim, vocab_size, num_labels, max_seq_len):

        tf.reset_default_graph()

        label = tf.placeholder(tf.int32, [None, ], 'label')
        name  = tf.placeholder(tf.int32, [None, max_seq_len], 'name')

        # expose placeholders
        self.placeholders = { 'label' : label, 'name' : name }

        batch_size_ = tf.shape(name)[0]
        # actual length of sequences considering padding                                 
        seqlens = tf.count_nonzero(name, axis=-1)                                        

        # word embedding
        lemb = tf.get_variable(shape=[num_labels, wdim], dtype=tf.float32,               
                        initializer=tf.random_uniform_initializer(-0.01, 0.01),          
                        name='word_embedding')

        wemb = tf.get_variable(shape=[vocab_size, wdim], dtype=tf.float32,               
                        initializer=tf.random_uniform_initializer(-0.01, 0.01),          
                        name='label_embedding')

        wemb = tf.concat([ tf.zeros([1, wdim]), wemb ], axis=0)

        Wo = tf.get_variable(shape=[hdim, vocab_size], dtype=tf.float32,               
                        initializer=tf.random_uniform_initializer(-0.01, 0.01),          
                        name='output_projection_w')

        bo = tf.get_variable(shape=[vocab_size], dtype=tf.float32,               
                        initializer=tf.random_uniform_initializer(-0.01, 0.01),          
                        name='output_bias')

        #with tf.variable_scope ('decoder') as dec_scope:
        cell = tf.nn.rnn_cell.LSTMCell(hdim)
        zero_state = cell.zero_state(batch_size_, tf.float32)
        # initial state
        state = tf.nn.rnn_cell.LSTMStateTuple(zero_state.c, tf.nn.embedding_lookup(lemb, label))
        ip = tf.nn.embedding_lookup(wemb, tf.transpose(name, [1, 0])[0])

        outputs = []
        llogits  = []
        predictions = []
        with tf.variable_scope('decoder') as scope:
            for i in range(max_seq_len):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output, state = cell(ip, state)
                outputs.append(output)
                # predict character; embed
                logits = tf.matmul(output, Wo) + bo
                llogits.append(logits)
                prediction = tf.argmax(tf.nn.softmax(logits), axis=-1)
                predictions.append(prediction)
                # input to next step
                ip = tf.nn.embedding_lookup(wemb, prediction)

        probs  = tf.nn.softmax(tf.stack(llogits))
        preds  = tf.stack(predictions)#tf.argmax(probs, axis=-1)

        # Cross Entropy
        #  mask padding
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=tf.reshape(tf.stack(llogits), [batch_size_, max_seq_len, -1]),
                labels=name) * tf.cast(name > 0, tf.float32)
        loss = tf.reduce_mean(ce)

        # Accuracy
        accuracy = tf.reduce_mean(
                        tf.cast(
                            tf.equal(tf.transpose(tf.cast(preds, tf.int32)), name),
                            tf.float32
                            ) * tf.cast(name > 0, tf.float32)
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
                    model.placeholders['name' ]  : np.random.randint(0, 10, [8, 20]),
                    model.placeholders['label']  : np.random.randint(0, 10, [8, ]  )
                    }
                )
                    

if __name__ == '__main__':

    model = NameGenerator(10, 10, 100, 12, 20)
    out = rand_exec(model)

    print(out['loss'], out['accuracy'])
