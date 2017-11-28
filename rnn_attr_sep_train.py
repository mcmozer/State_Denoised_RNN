# This version of the code trains the attractor connections with a separate
# objective function than the objective function used to train all other weights
# in the network (on the prediction task).

from __future__ import print_function
import itertools
import tensorflow as tf
import numpy as np

# Architecture Parameters
N_INPUT = 1           # number of input units
SEQ_LEN = 5           # number of bits in input sequence   
N_HIDDEN = 5          # number of hidden units 
N_CLASSES = 1         # number of output units
ARCH = 'tanh'         # hidden layer type: 'GRU' or 'tanh'
NOISE_LEVEL = .25     # noise in training attractor net
N_ATTRACTOR_STEPS = 5 # number of time steps in attractor dynamics
ATTR_WEIGHT_CONSTRAINTS = False # DEBUG TRUE
                      # True: make attractor weights symmetric and have zero diag
                      # False: unconstrained
ATTR_OUTPUT_TRANSFORM = True # DEBUG False
                      # True: add extra linear transform to output of attractor net
ATTR_WEIGHTS_TRAINED_ON_PREDICTION = False
                      # True: train attractor weights on attractor net _and_ prediction

# Training Parameters
training_epochs = 10000
n_replications = 100
batch_size = 16
display_epoch = 200
n_train = pow(2,SEQ_LEN) # train on all seqs
n_test = pow(2,SEQ_LEN)
LRATE_PREDICTION = 0.008
LRATE_ATTRACTOR = 0.008 
LOSS_SWITCH_FREQ = 0 # how often (in epochs) to switch between attractor and prediction loss

tf.set_random_seed(100)
np.random.seed(100)

# tf Graph input
X = tf.placeholder("float", [None, SEQ_LEN, N_INPUT])
Y = tf.placeholder("float", [None, N_CLASSES])
attractor_tgt_net = tf.placeholder("float", [None, N_HIDDEN])

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
         'aout': tf.get_variable("W_aout", initializer=tf.eye(N_HIDDEN)),
         'attractor': tf.get_variable("W_attractor", initializer=tf.zeros([N_HIDDEN,N_HIDDEN]))
        }

    b = {'out': mozer_get_variable("b_out", [N_CLASSES]),
         'stack': mozer_get_variable("b_stack", [3 * N_HIDDEN]),
         'aout': tf.get_variable("b_aout", initializer=tf.zeros([N_HIDDEN])),
         'attractor': tf.get_variable("b_attractor", initializer=tf.zeros([N_HIDDEN]))
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
        if ATTR_WEIGHT_CONSTRAINTS:
            W_attractor_constrained = 0.5 * (W['attractor'] + \
                                  tf.transpose(W['attractor'])) * \
                                               (1.0 - tf.eye(N_HIDDEN))
        else:
            W_attractor_constrained = W['attractor']

        block_size = [-1, N_HIDDEN]

        def _step(accumulated_vars, input_vars):
            h_prev, _, = accumulated_vars
            x = input_vars

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

            h_net = tf.atanh(tf.minimum(.99999, tf.maximum(-.99999, h))) 

            h_cleaned = tf.zeros(tf.shape(h_prev))
            for _ in range(N_ATTRACTOR_STEPS):
                h_cleaned = tf.matmul(tf.tanh(h_cleaned), W_attractor_constrained) \
                                                                + h_net + b['attractor']
            # add the linear transform at the end that seems to help
            if (ATTR_OUTPUT_TRANSFORM):
                h_cleaned = tf.matmul(tf.tanh(h_cleaned), W['aout']) + b['aout']
            else:
                h_cleaned = tf.tanh(h_cleaned)
            return [h_cleaned, h_net]

        # X:                       (batch_size, SEQ_LEN, N_HIDDEN) 
        # expected shape for scan: (SEQ_LEN, batch_size, N_HIDDEN) 
        batch_size = tf.shape(X)[0]
        [h_clean_seq, h_net_seq] = tf.scan(_step, 
                  elems=tf.transpose(X, [1, 0, 2]),
                  initializer=[tf.zeros([batch_size, N_HIDDEN], tf.float32),  # h_clean
                               tf.zeros([batch_size, N_HIDDEN], tf.float32)],  # h_net
                  name='GRU/scan')

        out = tf.nn.sigmoid(tf.matmul(h_clean_seq[-1], W['out']) + b['out'])
        return [out, h_net_seq]

######### END GRU #################################################################

######### BEGIN TANH RNN ########################################################

def RNN_tanh_params_init():

    W = {'in' : mozer_get_variable("W_in",[N_INPUT, N_HIDDEN]),
         'rec' : mozer_get_variable("W_rec", [N_HIDDEN, N_HIDDEN]),
         'out': mozer_get_variable("W_out", [N_HIDDEN, N_CLASSES]),
         'aout': tf.get_variable("W_aout", initializer=tf.eye(N_HIDDEN)),
         'attractor': tf.get_variable("W_attractor", initializer=tf.zeros([N_HIDDEN,N_HIDDEN]))
        }
    b = {'rec': mozer_get_variable("b_rec", [N_HIDDEN]),
         'out': mozer_get_variable("b_out", [N_CLASSES]),
         'aout': tf.get_variable("b_aout", initializer=tf.zeros([N_HIDDEN])),
         'attractor': tf.get_variable("b_attractor", initializer=tf.zeros([N_HIDDEN]))
        }

    params = {
        'W': W,
        'b': b
    }
    return params


def RNN_tanh(X, params):
    W = params['W']
    b = params['b']
    if ATTR_WEIGHT_CONSTRAINTS:
        W_attractor_constrained = 0.5 * (W['attractor'] + \
                              tf.transpose(W['attractor'])) * \
                                           (1.0 - tf.eye(N_HIDDEN))
    else:
        W_attractor_constrained = W['attractor']
    
    def _step(accumulated_vars, input_vars):
        h_prev, _, = accumulated_vars
        x = input_vars

        # update the hidden state but don't apply the squashing function
        h_net = tf.matmul(h_prev, W['rec']) + tf.matmul(x, W['in']) + b['rec']

        ##################################
        # run attractor net
        ##################################

        h_cleaned = tf.zeros(tf.shape(h_prev))
        for _ in range(N_ATTRACTOR_STEPS):
            h_cleaned = tf.matmul(tf.tanh(h_cleaned), W_attractor_constrained) \
                                                            + h_net + b['attractor']
        # add the linear transform at the end that seems to help
        if (ATTR_OUTPUT_TRANSFORM):
            h_cleaned = tf.matmul(tf.tanh(h_cleaned), W['aout']) + b['aout']
        else:
            h_cleaned = tf.tanh(h_cleaned)
        return [h_cleaned, h_net]

    # X:                       (batch_size, SEQ_LEN, N_INPUT) 
    # expected shape for scan: (SEQ_LEN, batch_size, N_INPUT) 
    batch_size = tf.shape(X)[0]
    [h_clean_seq, h_net_seq] = tf.scan(_step,
                  elems=tf.transpose(X, [1, 0, 2]),
                  initializer=[tf.zeros([batch_size, N_HIDDEN], tf.float32),  # h_clean
                               tf.zeros([batch_size, N_HIDDEN], tf.float32)], # h_net
                  name='RNN/scan')

    out = tf.nn.sigmoid(tf.matmul(h_clean_seq[-1], W['out']) + b['out'])

    return [out, h_net_seq]
    # out:                     (batch_size)
    # h_net_seq                (SEQ_LEN, batch_size, N_HIDDEN)

######### END TANH RNN ##########################################################

######### BEGIN ATTRACTOR NET LOSS FUNCTION #####################################

def attractor_net_loss_function(attractor_tgt_net, params):
    # attractor_tgt_net has dimensions #examples X #hidden
    #                   where the target value is tanh(attractor_tgt_net)

    W = params['W']
    b = params['b']
    if ATTR_WEIGHT_CONSTRAINTS:
        W_attractor_constrained = 0.5 * (W['attractor'] + \
                              tf.transpose(W['attractor'])) * \
                                           (1.0 - tf.eye(N_HIDDEN))
    else:
        W_attractor_constrained = W['attractor']

    # clean-up for attractor net training
    input_bias = attractor_tgt_net + NOISE_LEVEL \
                                     * tf.random_normal(tf.shape(attractor_tgt_net))
    a_cleaned = tf.zeros(tf.shape(attractor_tgt_net))
    for _ in range(N_ATTRACTOR_STEPS):
        a_cleaned = tf.matmul(tf.tanh(a_cleaned), W_attractor_constrained) \
                                                     + input_bias + b['attractor']
    if (ATTR_OUTPUT_TRANSFORM):
        a_cleaned = tf.matmul(tf.tanh(a_cleaned), W['aout']) + b['aout']
    else:
        a_cleaned = tf.tanh(a_cleaned)

    # loss is % reduction in noise level
    attr_tgt = tf.tanh(attractor_tgt_net)
    attr_loss = tf.reduce_mean(tf.pow(attr_tgt - a_cleaned,2)) / \
                tf.reduce_mean(tf.pow(attr_tgt - tf.tanh(input_bias),2))

    return attr_loss

######### END ATTRACTOR NET LOSS FUNCTION #######################################

# Define architecture graph
if (ARCH == 'tanh'):
    params = RNN_tanh_params_init()
    [Y_, h_net_seq] = RNN_tanh(X, params)
elif (ARCH == 'GRU'):
    params = GRU_params_init()
    [Y_, h_net_seq] = GRU(X, params)
else:
    print("ERROR: undefined architecture")
    exit()

# Define loss graphs
pred_loss_op = tf.reduce_mean(tf.pow(Y_ - Y, 2) / .25)
attr_loss_op = attractor_net_loss_function(attractor_tgt_net, params)

# get list of all parameters except attractor weights
params_attractor = [ params['W']['attractor'], params['b']['attractor'],
                    params['W']['aout'], params['b']['aout'] ]
params_without_attractor = [params['W'][key] for key in params['W']] + \
                                   [params['b'][key] for key in params['b']]
if (not ATTR_WEIGHTS_TRAINED_ON_PREDICTION):
    for p in params_attractor:
        params_without_attractor.remove(p)

# Define optimizer for prediction task
optimizer_pred = tf.train.AdamOptimizer(learning_rate=LRATE_PREDICTION)
pred_train_op = optimizer_pred.minimize(pred_loss_op, 
                                   var_list=params_without_attractor)
# Define optimizer for attractor net task
optimizer_attr = tf.train.AdamOptimizer(learning_rate=LRATE_ATTRACTOR)
attr_train_op = optimizer_attr.minimize(attr_loss_op, var_list=params_attractor)
# Evaluate model accuracy
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

        train_prediction_loss = True
        for epoch in range(1, training_epochs + 2):
            if (epoch-1) % display_epoch == 0:
                ploss, acc, hid_vals = sess.run([pred_loss_op, accuracy, h_net_seq], 
                                             feed_dict={X: X_train, Y: Y_train})
                aloss = sess.run(attr_loss_op,feed_dict={attractor_tgt_net: \
                                                           hid_vals.reshape(-1,N_HIDDEN)})
                print("epoch " + str(epoch-1) + ", Loss Pred " + \
                          "{:.4f}".format(ploss) + ", Loss Att " + \
                          "{:.4f}".format(aloss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))
                if (acc == 1.0):
                   break
            if epoch > 1 and LOSS_SWITCH_FREQ > 0 and (epoch-1) % LOSS_SWITCH_FREQ == 0:
               train_prediction_loss = not train_prediction_loss
            batches = get_batches(batch_size, X_train, Y_train)
            for (batch_x, batch_y) in batches:
                if (LOSS_SWITCH_FREQ == 0 or train_prediction_loss):
                    # Optimize all parameters except for attractor weights
                    _, hid_vals = sess.run([pred_train_op, h_net_seq], 
                                           feed_dict={X: batch_x, Y: batch_y})
                else:
                    hid_vals = sess.run(h_net_seq, feed_dict={X: batch_x, Y: batch_y})
                    sess.run(attr_train_op, feed_dict={attractor_tgt_net: 
                                                   hid_vals.reshape(-1,N_HIDDEN)})

                # Optimize attractor weights
                if (LOSS_SWITCH_FREQ == 0):
                    sess.run(attr_train_op, feed_dict={attractor_tgt_net: 
                                                   hid_vals.reshape(-1,N_HIDDEN)})

        print("Optimization Finished!")
        saved_acc.append(acc)
        if (acc == 1.0):
            saved_epoch.append(epoch)

        for p in params_attractor:
            print (p.name, ' ', p.eval())
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
          'noise_level',NOISE_LEVEL,'n_attr_steps',N_ATTRACTOR_STEPS)
    print ('lrate prediction','LRATE_PREDICTION',' lrate attractor ',LRATE_ATTRACTOR)
    print('mean accuracy', np.mean(saved_acc))
    print('indiv runs ',saved_acc)
    print('mean epoch', np.mean(saved_epoch))
    print('indiv epochs ',saved_epoch)

