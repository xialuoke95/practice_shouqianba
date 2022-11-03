import tensorflow as tf
import math

class UserItemFM():
    def __init__(self, n_user, n_item, batch_size, dim, use_bias=False):
        init_val = 1.0 / math.sqrt(dim)

        self.user_bias = tf.Variable(
            tf.random_uniform([n_user], -0.01, 0.01))
        self.item_bias = tf.Variable(
            tf.random_uniform([n_item], -0.01, 0.01))

        self.user_embeddings = tf.Variable(
            tf.random_uniform([n_user, dim],
                              -1.0 * init_val, init_val),
            name = 'user_embed')
        self.item_embeddings = tf.Variable(
            tf.random_uniform([n_item, dim],
                              -1.0 * init_val, init_val),
            name = 'item_embed')
        
        self.user_inputs = tf.placeholder(tf.int32,
                                          shape=[batch_size],
                                          name='user')
        self.item_inputs = tf.placeholder(tf.int32,
                                          shape=[batch_size],
                                          name='item')
        self.label_inputs = tf.placeholder(tf.float32,
                                           shape=[batch_size],
                                           name='label')

        user_embed = tf.nn.embedding_lookup(
            self.user_embeddings, self.user_inputs)
        item_embed = tf.nn.embedding_lookup(
            self.item_embeddings, self.item_inputs)

        bu = tf.nn.embedding_lookup(self.user_bias, self.user_inputs)
        bi = tf.nn.embedding_lookup(self.item_bias, self.item_inputs)
        ui = tf.multiply(user_embed, item_embed)
        self.output = tf.reduce_sum(ui, 1)
        self.output = self.output + bu + bi
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels = self.label_inputs, logits = self.output)

with tf.Session(config = config) as sess:
    sess.run(init)
    user_batch = []
    item_batch = []
    author_batch = []
    label_batch = []
    preds = []

    for line in file('sorted_small_dataset.tsv'):
        n += 1
        tks = line.strip().split('\t')
        user, item, author, label, tm = [int(x) for x in tks]
        user_batch.append(user)
        item_batch.append(item)
        author_batch.append(author)
        label_batch.append(label)
        if len(user_batch) == batch_size:
            _, l, y= sess.run([optimizer, net.loss, net.output],
                            feed_dict={net.user_inputs: user_batch,
                                       net.item_inputs: item_batch,
                                       net.author_inputs: author_batch,
                                       net.label_inputs: label_batch})
            preds = preds + [(y[i], label_batch[i]) for i in range(batch_size)]
            user_batch = []
            item_batch = []
            author_batch = []
            label_batch = []

        if len(preds) > 100000:
            print n, auc(preds)
            preds = []
        