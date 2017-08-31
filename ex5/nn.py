import tensorflow as tf

n_nodes_hl1 = 15
n_nodes_hl2 = 10

batch_size = 5
n_epochs = 50


def build_model(x, n_features, n_classes):

    hidden_1_layer = {
        'w': tf.Variable(tf.random_normal([n_features, n_nodes_hl1])),
        'b': tf.Variable(tf.random_normal([n_nodes_hl1]))
    }

    hidden_2_layer = {
        'w': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
        'b': tf.Variable(tf.random_normal([n_nodes_hl2]))
    }

    output_layer = {
        'w': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
        'b': tf.Variable(tf.random_normal([n_classes]))
    }

    l1 = tf.add(
        tf.matmul(x, hidden_1_layer['w']),
        hidden_1_layer['b']
    )
    l1 = tf.nn.relu(l1)

    l2 = tf.add(
        tf.matmul(l1, hidden_2_layer['w']),
        hidden_2_layer['b']
    )
    l2 = tf.nn.relu(l2)

    output = tf.add(
        tf.matmul(l2, output_layer['w']),
        output_layer['b']
    )

    return output


def build_cost(prediction, y):

    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=prediction,
            labels=y
        )
    )
    return cost

def create_place_holders(n_features, n_classes):
    x = tf.placeholder('float', [None, n_features])
    y = tf.placeholder('float')

    return x, y

def batch_gen(data):
    current_idx = 0
    while current_idx < len(data[0]):
        vectors = data[0][current_idx:current_idx+batch_size]
        labels = data[1][current_idx:current_idx+batch_size]
        yield vectors, labels
        current_idx += batch_size


def train(data, n_feature, n_classes):
    x, y = create_place_holders(n_feature, n_classes)
    prediction = build_model(x, n_feature, n_classes)
    cost = build_cost(prediction, y)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in xrange(n_epochs):
            epoch_loss = 0
            batches = batch_gen(data)
            for i, batch in enumerate(batches):
                ex = batch[0]
                ey = batch[1]
                _, c = sess.run(
                    [optimizer, cost],
                    feed_dict={
                        x: ex,
                        y: ey
                    }
                )
                epoch_loss += c
            print('Epoch {} out of {} completed with loss {}'.format(epoch, n_epochs, epoch_loss))


        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(correct, 'float'))
        calculated_acc = acc.eval(
            {x: data[0], y: data[1]}
        )

        print 'accuracy = {}'.format(calculated_acc)
