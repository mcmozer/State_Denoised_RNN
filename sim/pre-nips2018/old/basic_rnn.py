# This code implements a basic RNN for parity computation.  The architecture
# consists of an input layer, hidden layer, and output layer. The hidden layer is
# fully recurrent.  The hidden units can be standard tanh neurons, LSTM neurons, or
# GRU neurons.

from __future__ import print_function
import itertools
import tensorflow as tf
import numpy as np

# Training Parameters
training_epochs = 10000
n_replications = 100
learning_rate = 0.002
batch_size = 16
display_epoch = 200
n_train = 100
n_test = 100

# Architecture Parameters
n_input = 1 
seq_len = 5 # sequence + the answer
n_hidden = 5 # hidden layer num of features
n_classes = 1
arch = 'GRU' # 'LSTM', 'GRU', or 'tanh'
tf.set_random_seed(1)
np.random.seed(1)

# tf Graph input
X = tf.placeholder("float", [None, seq_len, n_input])
Y = tf.placeholder("float", [None, n_classes])

def generate_parity_sequences(N, count):
    """
    Generate :count: sequences of length :N:.
    If odd # of 1's -> output 1
    else -> output 0
    """
    parity = lambda x: 1 if (x % 2 == 1) else 0
    if (count >= 2**N):
        sequences = np.asarray([seq for seq in itertools.product([0,1],repeat=N)])
    else:
        sequences = np.random.choice([0, 1], size=[count, N], replace=True)
    counts = np.count_nonzero(sequences == 1, axis=1)
    # parity each sequence, expand dimensions by 1 to match sequences shape
    y = np.expand_dims(np.array([parity(x) for x in counts]), axis=1)

    # In case if you wanted to have the answer just appended at the end of the sequence:
    #     # append the answer at the end of each sequence
    #     seq_plus_y = np.concatenate([sequences, y], axis=1)
    #     print(sequences.shape, y.shape, seq_plus_y.shape)
    #     return seq_plus_y
    return np.expand_dims(sequences, axis=2), y


X_train, Y_train = generate_parity_sequences(seq_len, n_train)
# BUG: need to generate distinct test and training sequences
X_test, Y_test = generate_parity_sequences(seq_len, n_test)

def mozer_get_variable(vname, mat_dim):
    if (len(mat_dim) == 1): # bias
        val = 0.1 * tf.random_normal(mat_dim)
        var = tf.get_variable(vname, initializer=val)

    else:
        #var = tf.get_variable(vname, shape=mat_dim, 
        #                    initializer=tf.contrib.layers.xavier_initializer())

        #val = tf.random_normal(mat_dim)
        #var = tf.get_variable(vname, initializer=val)

        val = tf.random_normal(mat_dim)
        val = 2 * val / tf.reduce_sum(tf.abs(val),axis=0, keep_dims=True)
        var = tf.get_variable(vname, initializer=val)
    return var

######### BEGIN GRU ###############################################################

def GRU_params_init():
    W = {'out': mozer_get_variable("W_out", [n_hidden, n_classes]),
         'in_stack': mozer_get_variable("W_in_stack", [n_input, 3*n_hidden]),
         'rec_stack': mozer_get_variable("W_rec_stack", [n_hidden,3*n_hidden])}

    b = {'out': mozer_get_variable("b_out", [n_classes]),
         'stack': mozer_get_variable("b_stack", [3 * n_hidden])
        }

    params = {
        'W': W,
        'b': b
    }
    return params

def GRU(X, params):
    with tf.variable_scope("GRU"):
        W = params['W']
        b = params['b']
        block_size = [-1, n_hidden]

        def _step(h_prev, input_vars):
            x_in = input_vars

            preact = tf.matmul(x_in, W['in_stack'][:,:n_hidden*2]) + \
                     tf.matmul(h_prev, W['rec_stack'][:,:n_hidden*2]) + \
                     b['stack'][:n_hidden*2]
            z = tf.sigmoid(tf.slice(preact, [0, 0 * n_hidden], block_size))
            r = tf.sigmoid(tf.slice(preact, [0, 1 * n_hidden], block_size))
            # new potential candidate for memory vector
            c_cand = tf.tanh( tf.matmul(x_in, W['in_stack'][:,n_hidden*2:]) + \
                              tf.matmul(h_prev * r, W['rec_stack'][:,n_hidden*2:]) + \
                              b['stack'][n_hidden*2:])
            h = z * h_prev + (1.0 - z) * c_cand

            return h

        # X:                       (batch_size, seq_len, n_hidden) 
        # expected shape for scan: (seq_len, batch_size, n_hidden) 
        batch_size = tf.shape(X)[0]
        h = tf.scan(_step, elems=tf.transpose(X, [1, 0, 2]),
                           initializer=tf.zeros([batch_size, n_hidden], tf.float32),  # h
                           name='GRU/scan')
        # output activated prediction (sigmoid since we want a classification barrier at 0.5)
        return tf.nn.sigmoid(tf.matmul(h[-1], W['out']) + b['out'])

######### END GRU #################################################################

######### BEGIN LSTM ##############################################################

def LSTM_params_init():
    with tf.variable_scope("LSTM"):
        W = {'out': tf.Variable(tf.random_normal([n_hidden,n_classes]),
                                 name='W_out'),
             'in_stack': tf.Variable(tf.random_normal([n_input,4 * n_hidden]),
                                      name = 'W_in_stack'),
             'rec_stack': tf.Variable(tf.random_normal([n_hidden,4 * n_hidden]),
                                       name='W_rec_stack')
             }
        b = {'out': tf.Variable(tf.random_normal([n_classes]),
                        name='b_out'),
            'stack': tf.Variable(tf.random_normal([4*n_hidden]),
                        name='b_stack')
            }


    params = {
        'W': W,
        'b': b
    }
    return params

def LSTM(X, params):
    W = params['W']
    b = params['b']
    block_size = [-1, n_hidden]

    def _step(accumulated_vars, input_vars):
        h_prev, c_prev, = accumulated_vars
        x_in = input_vars
        # m - multiply for all four vectors at once and then slice it
        # gates: i - input, f - forget, o - output

        preact = tf.matmul(x_in, W['in_stack']) + \
                 tf.matmul(h_prev, W['rec_stack']) + \
                 b['stack']
        i = tf.sigmoid(tf.slice(preact, [0, 0 * n_hidden], block_size))
        f = tf.sigmoid(tf.slice(preact, [0, 1 * n_hidden], block_size))
        o = tf.sigmoid(tf.slice(preact, [0, 2 * n_hidden], block_size))
        # new potential candidate for memory vector
        c_cand = tf.tanh(tf.slice(preact, [0, 3 * n_hidden], block_size))

        # update memory by forgetting existing memory & adding new candidate memory
        c = f * c_prev + i * c_cand

        # update hidden vector state
        h = o * tf.tanh(c)

        return [h, c]

    # X:                       (batch_size, seq_len, n_hidden) 
    # expected shape for scan: (seq_len, batch_size, n_hidden) 
    batch_size = tf.shape(X)[0]
    outputs = tf.scan(_step,
                      elems=tf.transpose(X, [1, 0, 2]),
                      initializer=[tf.zeros([batch_size, n_hidden], tf.float32),  # h
                                   tf.zeros([batch_size, n_hidden], tf.float32)],  # c
                      name='LSTM/scan')
    h = outputs[0]  # only get h vector from [h, c] pair
    # output activated prediction (sigmoid since we want a classification barrier at 0.5)
    return tf.nn.sigmoid(tf.matmul(h[-1], W['out']) + b['out'])

######### END LSTM ##############################################################

######### BEGIN TANH RNN ########################################################

def RNN_tanh_params_init():

    W = {
        'in' : mozer_get_variable("W_in",[n_input, n_hidden]),
        'rec' : mozer_get_variable("W_rec", [n_hidden, n_hidden]),
        'out': mozer_get_variable("W_out", [n_hidden, n_classes])
    }
    b = {
        'rec': mozer_get_variable("b_rec", [n_hidden]),
        'out': mozer_get_variable("b_out", [n_classes])
    }

    params = {
        'W': W,
        'b': b
    }
    return params


def RNN_tanh(X, params):
    W = params['W']
    b = params['b']
    def _step_tanh(accumulated_vars, input_vars):
        h = accumulated_vars
        x = input_vars
        # update the hidden state
        h = tf.tanh(tf.matmul(h, W['rec']) + tf.matmul(x, W['in']) + b['rec'])
        return h

    # X:                       (batch_size, seq_len, n_hidden) 
    # expected shape for scan: (seq_len, batch_size, n_hidden) 
    batch_size = tf.shape(X)[0]
    outputs = tf.scan(_step_tanh,
                      elems=tf.transpose(X, [1, 0, 2]),
                      initializer=tf.zeros([batch_size, n_hidden], tf.float32),
                      name='RNN/scan')

    # output activated prediction (sigmoid since we want a classification barrier at 0.5)
    return tf.nn.sigmoid(tf.matmul(outputs[-1], W['out']) + b['out'])

######### END TANH RNN ###########################################################

if (arch == 'tanh'):
    Y_ = RNN_tanh(X, RNN_tanh_params_init())
elif (arch == 'LSTM'):
    Y_ = LSTM(X, LSTM_params_init())
elif (arch == 'GRU'):
    Y_ = GRU(X, GRU_params_init())
else:
    print("ERROR: undefined architecture")
    exit()

# Define loss and optimizer
# Since only one output unit,
loss_op = tf.reduce_mean(tf.pow(Y_ - Y, 2))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.round(Y_), Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


def get_batches(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    batches = []
    while len(idx) > 0:
       cur_idx = idx[:min(num, len(idx))]
       data_shuffle = [data[i] for i in cur_idx]
       labels_shuffle = [labels[i] for i in cur_idx]
       batches.append((np.asarray(data_shuffle), np.asarray(labels_shuffle)))
       idx = idx[num:]
    return batches


# Start training
with tf.Session() as sess:
    saved_acc = []
    saved_epoch = []
    for replication in range(n_replications):
        sess.run(init) # Run the initializer

        for epoch in range(1, training_epochs + 2):
            if (epoch-1) % display_epoch == 0:
                loss, acc = sess.run([loss_op, accuracy], 
                                             feed_dict={X: X_train, Y: Y_train})
                print("epoch " + str(epoch-1) + ", Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))
                if (acc == 1.0):
                   break
            batches = get_batches(batch_size, X_train, Y_train)
            for (batch_x, batch_y) in batches:
                #(batch_x, batch_y) = batch
                # Run optimization
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        print("Optimization Finished!")
        saved_acc.append(acc)
        if (acc == 1.0):
            saved_epoch.append(epoch)

        # test performance
        if (0):
            accs = []
            batches = get_batches(batch_size, X_test, Y_test)
            for batch in batches:
                (batch_x, batch_y) = batch
                acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
                print(acc)
                accs.append(acc)
            print("Testing Accuracy:", accuracy)
    print('mean accuracy', np.mean(saved_acc))
    print('indiv runs ',saved_acc)
    print('mean epoch', np.mean(saved_epoch))
    print('indiv epochs ',saved_epoch)
    print('arch',arch,'seq_len',seq_len, 'training_epochs',training_epochs)
