# This version of the code includes a second objective function to optimize noisy
# reconstruction via the attractor net.
# The two objective functions are integrated with weight LAMBDA and all parameters
# are optimized simultaneously to achieve the combined objective.

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
N_INPUT = 1 
SEQ_LEN = 5 # sequence + the answer
N_HIDDEN = 5 # hidden layer num of features
N_CLASSES = 1
ARCH = 'tanh' # 'LSTM', 'GRU', or 'tanh'
NOISE_LEVEL = .25
N_ATTRACTOR_STEPS = 3
LAMBDA = .08 # learning rate for attractor loss

tf.set_random_seed(1)
np.random.seed(1)

# tf Graph input
X = tf.placeholder("float", [None, SEQ_LEN, N_INPUT])
Y = tf.placeholder("float", [None, N_CLASSES])

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


X_train, Y_train = generate_parity_sequences(SEQ_LEN, n_train)
# BUG: need to generate distinct test and training sequences
X_test, Y_test = generate_parity_sequences(SEQ_LEN, n_test)

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
    W = {'out': mozer_get_variable("W_out", [N_HIDDEN, N_CLASSES]),
         'in_stack': mozer_get_variable("W_in_stack", [N_INPUT, 3*N_HIDDEN]),
         'rec_stack': mozer_get_variable("W_rec_stack", [N_HIDDEN,3*N_HIDDEN]),
         'attractor': tf.get_variable("W_attractor", initializer=tf.zeros([N_HIDDEN,N_HIDDEN]))
        }

    b = {'out': mozer_get_variable("b_out", [N_CLASSES]),
         'stack': mozer_get_variable("b_stack", [3 * N_HIDDEN]),
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
        block_size = [-1, N_HIDDEN]

        W_attractor_constr = 0.5 * (W['attractor'] + tf.transpose(W['attractor'])) * \
                                  (1.0 - tf.eye(N_HIDDEN))
    
        def _step(h_prev, x):

            preact = tf.matmul(x, W['in_stack'][:,:N_HIDDEN*2]) + \
                     tf.matmul(h_prev, W['rec_stack'][:,:N_HIDDEN*2]) + \
                     b['stack'][:N_HIDDEN*2]
            z = tf.sigmoid(tf.slice(preact, [0, 0 * N_HIDDEN], block_size))
            r = tf.sigmoid(tf.slice(preact, [0, 1 * N_HIDDEN], block_size))
            # new potential candidate for memory vector
            c_cand = tf.tanh( tf.matmul(x, W['in_stack'][:,N_HIDDEN*2:]) + \
                              tf.matmul(h_prev * r, W['rec_stack'][:,N_HIDDEN*2:]) + \
                              b['stack'][N_HIDDEN*2:])
            h = z * h_prev + (1.0 - z) * c_cand

            ##################################
            # run attractor net
            ##################################
            h_noisy_net = tf.atanh(tf.minimum(.99999, tf.maximum(-.99999, h))) + \
                                           NOISE_LEVEL * tf.random_normal([N_HIDDEN])

            h_cleaned = tf.tanh(h_noisy_net)
            for i in range(0,N_ATTRACTOR_STEPS):
                h_cleaned = tf.tanh(tf.matmul(h_cleaned, W_attractor_constr) + h_noisy_net) 
            return h_cleaned

        # X:                       (batch_size, SEQ_LEN, N_HIDDEN) 
        # expected shape for scan: (SEQ_LEN, batch_size, N_HIDDEN) 
        batch_size = tf.shape(X)[0]
        h = tf.scan(_step, elems=tf.transpose(X, [1, 0, 2]),
                           initializer=tf.zeros([batch_size, N_HIDDEN], tf.float32),  # h
                           name='GRU/scan')
        # output activated prediction (sigmoid since we want a classification barrier at 0.5)
        return tf.nn.sigmoid(tf.matmul(h[-1], W['out']) + b['out'])

######### END GRU #################################################################

######### BEGIN LSTM ##############################################################

def LSTM_params_init():
    with tf.variable_scope("LSTM"):
        W = {'out': tf.Variable(tf.random_normal([N_HIDDEN,N_CLASSES]),
                                 name='W_out'),
             'in_stack': tf.Variable(tf.random_normal([N_INPUT,4 * N_HIDDEN]),
                                      name = 'W_in_stack'),
             'rec_stack': tf.Variable(tf.random_normal([N_HIDDEN,4 * N_HIDDEN]),
                                       name='W_rec_stack')
             }
        b = {'out': tf.Variable(tf.random_normal([N_CLASSES]),
                        name='b_out'),
            'stack': tf.Variable(tf.random_normal([4*N_HIDDEN]),
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
    block_size = [-1, N_HIDDEN]

    def _step(accumulated_vars, input_vars):
        h_prev, c_prev, = accumulated_vars
        x_in = input_vars
        # m - multiply for all four vectors at once and then slice it
        # gates: i - input, f - forget, o - output

        preact = tf.matmul(x_in, W['in_stack']) + \
                 tf.matmul(h_prev, W['rec_stack']) + \
                 b['stack']
        i = tf.sigmoid(tf.slice(preact, [0, 0 * N_HIDDEN], block_size))
        f = tf.sigmoid(tf.slice(preact, [0, 1 * N_HIDDEN], block_size))
        o = tf.sigmoid(tf.slice(preact, [0, 2 * N_HIDDEN], block_size))
        # new potential candidate for memory vector
        c_cand = tf.tanh(tf.slice(preact, [0, 3 * N_HIDDEN], block_size))

        # update memory by forgetting existing memory & adding new candidate memory
        c = f * c_prev + i * c_cand

        # update hidden vector state
        h = o * tf.tanh(c)

        return [h, c]

    # X:                       (batch_size, SEQ_LEN, N_HIDDEN) 
    # expected shape for scan: (SEQ_LEN, batch_size, N_HIDDEN) 
    batch_size = tf.shape(X)[0]
    outputs = tf.scan(_step,
                      elems=tf.transpose(X, [1, 0, 2]),
                      initializer=[tf.zeros([batch_size, N_HIDDEN], tf.float32),  # h
                                   tf.zeros([batch_size, N_HIDDEN], tf.float32)],  # c
                      name='LSTM/scan')
    h = outputs[0]  # only get h vector from [h, c] pair
    # output activated prediction (sigmoid since we want a classification barrier at 0.5)
    return tf.nn.sigmoid(tf.matmul(h[-1], W['out']) + b['out'])

######### END LSTM ##############################################################

######### BEGIN TANH RNN ########################################################

def RNN_tanh_params_init():

    W = {'in' : mozer_get_variable("W_in",[N_INPUT, N_HIDDEN]),
         'rec' : mozer_get_variable("W_rec", [N_HIDDEN, N_HIDDEN]),
         'out': mozer_get_variable("W_out", [N_HIDDEN, N_CLASSES]),
         'attractor': tf.get_variable("W_attractor", initializer=tf.zeros([N_HIDDEN,N_HIDDEN]))
        }
    b = {'rec': mozer_get_variable("b_rec", [N_HIDDEN]),
         'out': mozer_get_variable("b_out", [N_CLASSES])
        }

    params = {
        'W': W,
        'b': b
    }
    return params


def RNN_tanh(X, params):
    W = params['W']
    b = params['b']
    W_attractor_constr = 0.5 * (W['attractor'] + tf.transpose(W['attractor'])) * \
                                  (1.0 - tf.eye(N_HIDDEN))
    
    def _step(accumulated_vars, input_vars):
        h_prev, _ = accumulated_vars
        x = input_vars

        # update the hidden state but don't apply the squashing function
        h_net = tf.matmul(h_prev, W['rec']) + tf.matmul(x, W['in']) + b['rec']

        ##################################
        # run attractor net
        ##################################
        h_noisy_net = h_net + NOISE_LEVEL * tf.random_normal([N_HIDDEN])

        h_cleaned = tf.tanh(h_noisy_net)
        for i in range(0,N_ATTRACTOR_STEPS):
            h_cleaned = tf.tanh(tf.matmul(h_cleaned, W_attractor_constr) + h_noisy_net) 

        objective2 = tf.pow(h_cleaned - tf.tanh(h_net),2)
        return [h_cleaned, objective2]

    # X:                       (batch_size, SEQ_LEN, N_HIDDEN) 
    # expected shape for scan: (SEQ_LEN, batch_size, N_HIDDEN) 
    batch_size = tf.shape(X)[0]
    [outputs, objective2] = tf.scan(_step,
                      elems=tf.transpose(X, [1, 0, 2]),
                      initializer=[tf.zeros([batch_size, N_HIDDEN], tf.float32), # h
                                   tf.zeros([batch_size, N_HIDDEN],tf.float32)], # objective fn 2
                      name='RNN/scan')

    # output activated prediction (sigmoid since we want a classification barrier at 0.5)
    return [tf.nn.sigmoid(tf.matmul(outputs[-1], W['out']) + b['out']), objective2]

######### END TANH RNN ##########################################################

if (ARCH == 'tanh'):
    [Y_, objective2] = RNN_tanh(X, RNN_tanh_params_init())
    loss_op = tf.reduce_mean(tf.pow(Y_ - Y, 2)) + LAMBDA * tf.reduce_mean(objective2)
elif (ARCH == 'LSTM'):
    Y_ = LSTM(X, LSTM_params_init())
    loss_op = tf.reduce_mean(tf.pow(Y_ - Y, 2))
elif (ARCH == 'GRU'):
    Y_ = GRU(X, GRU_params_init())
    loss_op = tf.reduce_mean(tf.pow(Y_ - Y, 2))
else:
    print("ERROR: undefined architecture")
    exit()

# Define loss and optimizer
# Since only one output unit,
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
    print('********************************************************************')
    print('arch',ARCH,'SEQ_LEN',SEQ_LEN, 'training_epochs',training_epochs,
          'noise_level',NOISE_LEVEL,'n_attr_steps',N_ATTRACTOR_STEPS, 'lambda', LAMBDA)
    print('mean accuracy', np.mean(saved_acc))
    print('indiv runs ',saved_acc)
    print('mean epoch', np.mean(saved_epoch))
    print('indiv epochs ',saved_epoch)

