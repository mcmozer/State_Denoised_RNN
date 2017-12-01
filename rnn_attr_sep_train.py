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
ARCH = 'GRU'         # hidden layer type: 'GRU' or 'tanh'
NOISE_LEVEL = .25     # noise in training attractor net (std deviation)
N_ATTRACTOR_STEPS = 5 # number of time steps in attractor dynamics
                      # REMEMBER: 1 step = no attractor net
ATTR_WEIGHT_CONSTRAINTS = True
                      # True: make attractor weights symmetric and have zero diag
                      # False: unconstrained
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


################ GLOBAL VARIABLES #######################################################

# prediction input
X = tf.placeholder("float", [None, SEQ_LEN, N_INPUT])
# prediction output
Y = tf.placeholder("float", [None, N_CLASSES])
# attr net target
attractor_tgt_net = tf.placeholder("float", [None, N_HIDDEN])

# attr net weights
attr_net = {
     'W': tf.get_variable("attractor_W", initializer=.01*tf.random_normal([N_HIDDEN,N_HIDDEN])),
     'b': tf.get_variable("attractor_b", initializer=.01*tf.random_normal([N_HIDDEN])),
     'scale': tf.get_variable("attractor_scale", initializer=tf.ones([1]))
     }

if ATTR_WEIGHT_CONSTRAINTS: # symmetric + nonnegative diagonal weight matrix
    attr_net['Wconstr'] = (attr_net['W'] + tf.transpose(attr_net['W'])) \
                                              * .5 * (1.0 - tf.eye(N_HIDDEN)) \
                               + tf.diag(tf.abs(tf.diag_part(attr_net['W'])))
else:
    attr_net['Wconstr'] = attr_net['W'] 

################ generate_parity_sequences ##############################################

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

################ get_batches ############################################################

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

################ mozer_get_variable #####################################################

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


############### RUN_ATTRACTOR_NET #################################################

def run_attractor_net(input_bias):

    if (N_ATTRACTOR_STEPS > 0):
        a_clean = tf.zeros(tf.shape(input_bias))
        for i in range(N_ATTRACTOR_STEPS):
            a_clean = tf.matmul(tf.tanh(a_clean), attr_net['Wconstr']) \
                                        + attr_net['scale'] * input_bias + attr_net['b']
        a_clean = tf.tanh(a_clean)
    else:
        a_clean = tf.tanh(input_bias)
    return a_clean


############### ATTRACTOR NET LOSS FUNCTION #####################################

def attractor_net_loss_function(attractor_tgt_net, params):
    # attractor_tgt_net has dimensions #examples X #hidden
    #                   where the target value is tanh(attractor_tgt_net)

    # clean-up for attractor net training
    input_bias = attractor_tgt_net + NOISE_LEVEL \
                                     * tf.random_normal(tf.shape(attractor_tgt_net))
    a_cleaned = run_attractor_net(input_bias)

    # loss is % reduction in noise level
    attr_tgt = tf.tanh(attractor_tgt_net)
    attr_loss = tf.reduce_mean(tf.pow(attr_tgt - a_cleaned,2)) / \
                tf.reduce_mean(tf.pow(attr_tgt - tf.tanh(input_bias),2))

    return attr_loss


############### GRU ###############################################################

def GRU_params_init():
    W = {'out': mozer_get_variable("W_out", [N_HIDDEN, N_CLASSES]),
         'in_stack': mozer_get_variable("W_in_stack", [N_INPUT, 3*N_HIDDEN]),
         'rec_stack': mozer_get_variable("W_rec_stack", [N_HIDDEN,3*N_HIDDEN]),
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

            # insert attractor net
            h_net = tf.atanh(tf.minimum(.99999, tf.maximum(-.99999, h))) 
            h_cleaned = run_attractor_net(h_net)

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
        }
    b = {'rec': mozer_get_variable("b_rec", [N_HIDDEN]),
         'out': mozer_get_variable("b_out", [N_CLASSES]),
        }

    params = {
        'W': W,
        'b': b
    }
    return params


def RNN_tanh(X, params):
    W = params['W']
    b = params['b']
    
    def _step(accumulated_vars, input_vars):
        h_prev, _, = accumulated_vars
        x = input_vars

        # update the hidden state but don't apply the squashing function
        h_net = tf.matmul(h_prev, W['rec']) + tf.matmul(x, W['in']) + b['rec']

        # insert attractor net
        h_cleaned = run_attractor_net(h_net)

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


######### MAIN CODE #############################################################
tf.set_random_seed(100)
np.random.seed(100)

X_train, Y_train = generate_parity_sequences(SEQ_LEN, n_train)
# BUG: need to generate distinct test and training sequences
X_test, Y_test = generate_parity_sequences(SEQ_LEN, n_test)

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

# separate out parameters to be optimized
prediction_parameters = params['W'].values() + params['b'].values()
attr_net_parameters = attr_net.values()
attr_net_parameters.remove(attr_net['Wconstr']) # not a real parameter

if (ATTR_WEIGHTS_TRAINED_ON_PREDICTION):
    prediction_parameters += attr_net_parameters

# Define optimizer for prediction task
optimizer_pred = tf.train.AdamOptimizer(learning_rate=LRATE_PREDICTION)
pred_train_op = optimizer_pred.minimize(pred_loss_op, var_list=prediction_parameters)
# Define optimizer for attractor net task
optimizer_attr = tf.train.AdamOptimizer(learning_rate=LRATE_ATTRACTOR)
attr_train_op = optimizer_attr.minimize(attr_loss_op, var_list=attr_net_parameters)
# Evaluate model accuracy
correct_pred = tf.equal(tf.round(Y_), Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


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
                if (LOSS_SWITCH_FREQ == 0 or not train_prediction_loss):
                    if (N_ATTRACTOR_STEPS > 0):
                        # Optimize attractor weights
                        hid_vals = sess.run(h_net_seq, feed_dict={X: batch_x, Y: batch_y})
                        sess.run(attr_train_op, feed_dict={attractor_tgt_net: 
                                                       hid_vals.reshape(-1,N_HIDDEN)})

        print("Optimization Finished!")
        saved_acc.append(acc)
        if (acc == 1.0):
            saved_epoch.append(epoch)

        for p in attr_net.values():
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

