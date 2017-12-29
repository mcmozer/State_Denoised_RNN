#!/usr/local/bin/python

# This version of the code trains the attractor connections with a separate
# objective function than the objective function used to train all other weights
# in the network (on the prediction task).

from __future__ import print_function
import itertools
import tensorflow as tf
import numpy as np
import sys
import argparse
import fsm

parser = argparse.ArgumentParser()
parser.add_argument('-arch',type=str,default='GRU',
                    help='hidden layer type, GRU or tanh')
parser.add_argument('-task',type=str,default='parity',
                    help='task (parity, majority, reber, kazakov)')
parser.add_argument('-lrate_prediction',type=float,default=0.008,
                    help='prediction task learning rate')
parser.add_argument('-lrate_attractor',type=float,default=0.008,
                    help='attractor task learning rate')
parser.add_argument('-noise_level',type=float,default=0.25,
                    help='attractor input noise (+ = Gauss std dev, - = % removed')
parser.add_argument('-n_attractor_steps',type=int,default=5,
                    help='number of attractor steps (0=no attractor net)')
parser.add_argument('-seq_len',type=int,default=5,
                    help='input sequence length')
parser.add_argument('-n_hidden',type=int,default=5,
                    help='number of recurrent hidden units')
parser.add_argument('-display_epoch',type=int,default=200,
                    help='frequency of displaying training results')
parser.add_argument('-training_epochs',type=int,default=10000,
                    help='number of training epochs')
parser.add_argument('-n_replications',type=int,default=100,
                    help='number of replications')
parser.add_argument('-train_attr_weights_on_prediction',
                    dest='train_attr_weights_on_prediction', action='store_true')
parser.add_argument('-no-train_attr_weights_on_prediction',
                    dest='train_attr_weights_on_prediction', action='store_false')
parser.set_defaults(train_attr_weights_on_prediction=False)
parser.add_argument('-report_best_train_performance',
                    dest='report_best_train_performance', action='store_true')
parser.add_argument('-no-report_best_train_performance',
                    dest='report_best_train_performance', action='store_false')
parser.set_defaults(report_best_train_performance=False)

# NOT YET IMPLEMENTED
#parser.add_argument('-attractor_train_delay',type=int,default=100,
#                    help='number of epochs to wait before training attractor weights')

args=parser.parse_args()
print(args)

# Architecture Parameters
SEQ_LEN = args.seq_len# number of bits in input sequence   
N_HIDDEN = args.n_hidden          
                      # number of hidden units 
ARCH = args.arch      # hidden layer type: 'GRU' or 'tanh'
NOISE_LEVEL = args.noise_level
                      # noise in training attractor net 
                      # if >=0, Gaussian with std dev NOISE_LEVEL 
                      # if < 0, Bernoulli dropout proportion -NOISE_LEVEL 
INPUT_NOISE_LEVEL = .1

N_ATTRACTOR_STEPS = args.n_attractor_steps 
                      # number of time steps in attractor dynamics
                      # if = 0, then no attractor net
ATTR_WEIGHT_CONSTRAINTS = True
                      # True: make attractor weights symmetric and have zero diag
                      # False: unconstrained
TRAIN_ATTR_WEIGHTS_ON_PREDICTION = args.train_attr_weights_on_prediction
                      # True: train attractor weights on attractor net _and_ prediction
REPORT_BEST_TRAIN_PERFORMANCE = args.report_best_train_performance
                      # True: save the train/test perf on the epoch for which train perf was best

TASK = args.task      # task (parity, majority, reber, kazakov)
if (TASK=='parity'):
    N_INPUT = 1           # number of input units
    N_CLASSES = 1         # number of output units
    N_TRAIN = pow(2,SEQ_LEN) # train on all seqs
    N_TEST = pow(2,SEQ_LEN)
elif (TASK=='majority'):
    N_INPUT = 1           # number of input units
    N_CLASSES = 1         # number of output units
    N_TRAIN = 64 
    N_TEST = 4096-64
elif (TASK=='reber'):
    N_INPUT = 7 # B E P S T V X
    N_CLASSES = 1
    N_TRAIN = 200
    N_TEST = 2000
elif (TASK=='kazakov'):
    N_INPUT = 5
    N_CLASSES = 1
    N_TRAIN = 400
    N_TEST = 2000
else:
    print('Invalid task: ',TASK)
    quit()


# Training Parameters

TRAINING_EPOCHS = args.training_epochs
N_REPLICATIONS = args.n_replications
BATCH_SIZE = 16
DISPLAY_EPOCH = args.display_epoch
LRATE_PREDICTION = args.lrate_prediction
LRATE_ATTRACTOR = args.lrate_attractor
LOSS_SWITCH_FREQ = 0 # how often (in epochs) to switch between attractor and prediction loss


################ GLOBAL VARIABLES #######################################################

# prediction input
X = tf.placeholder("float", [None, SEQ_LEN, N_INPUT])
# prediction output
Y = tf.placeholder("float", [None, N_CLASSES])
# attr net target
attractor_tgt_net = tf.placeholder("float", [None, N_HIDDEN])

# attr net weights
# NOTE: i tried setting attractor_W = attractor_b = 0 and attractor_scale=1.0
# which is the default "no attractor" model, but that doesn't learn as well as
# randomizing initial weights
attr_net = {
     'W': tf.get_variable("attractor_W", initializer=.01*tf.random_normal([N_HIDDEN,N_HIDDEN])),
     'b': tf.get_variable("attractor_b", initializer=.01*tf.random_normal([N_HIDDEN])),
     'scale': tf.get_variable("attractor_scale", initializer=.01*tf.ones([1]))
     }

if ATTR_WEIGHT_CONSTRAINTS: # symmetric + nonnegative diagonal weight matrix
    Wdiag = tf.matrix_band_part(attr_net['W'],0,0) # diagonal
    Wlowdiag = tf.matrix_band_part(attr_net['W'], -1, 0) - Wdiag # lower diagonal
    attr_net['Wconstr'] = Wlowdiag + tf.transpose(Wlowdiag) + tf.abs(Wdiag)
    #attr_net['Wconstr'] = .5 * (attr_net['W'] + tf.transpose(attr_net['W'])) * \
    #                      (1.0-tf.eye(N_HIDDEN)) + tf.abs(tf.matrix_band_part(attr_net['W'],0,0))

else:
    attr_net['Wconstr'] = attr_net['W'] 

################ generate_examples ######################################################

def generate_examples():

    if (TASK == 'parity'):
        X_train, Y_train = generate_parity_majority_sequences(SEQ_LEN, N_TRAIN)
        X_test, Y_test = generate_parity_majority_sequences(SEQ_LEN, N_TEST)
        if (INPUT_NOISE_LEVEL > 0.):
           X_test, Y_test = add_input_noise(INPUT_NOISE_LEVEL,X_test,Y_test,2)
    # for majority, split all sequences into training and test sets
    elif (TASK == 'majority'):
        X_train, Y_train = generate_parity_majority_sequences(SEQ_LEN, N_TRAIN+N_TEST)
        pix = np.random.permutation(N_TRAIN+N_TEST)
        X_train = X_train[pix[:N_TRAIN],:]
        Y_train = Y_train[pix[:N_TRAIN],:]
        X_test = X_train[pix[N_TRAIN:],:]
        Y_test = Y_train[pix[N_TRAIN:],:]
        if (INPUT_NOISE_LEVEL > 0.):
           X_test, Y_test = add_input_noise(INPUT_NOISE_LEVEL,X_test,Y_test,1)
    elif (TASK == 'reber'):
        _, Y_train, X_train, _ = fsm.generate_grammar_dataset(1, SEQ_LEN, N_TRAIN)
        _, Y_test, X_test, _ = fsm.generate_grammar_dataset(1, SEQ_LEN, N_TEST)
    elif (TASK == 'kazakov'):
        _, Y_train, X_train, _ = fsm.generate_grammar_dataset(2, SEQ_LEN, N_TRAIN)
        _, Y_test, X_test, _ = fsm.generate_grammar_dataset(2, SEQ_LEN, N_TEST)

    return [X_train, Y_train, X_test, Y_test]

################ add_input_noise ########################################################
# incorporate input noise into the test patterns

def add_input_noise(noise_level, X, Y, n_repeat):
# X: # examples X # sequence elements X #inputs
    X = np.repeat(X, n_repeat, axis=0)
    Y = np.repeat(Y, n_repeat, axis=0)
    X = X + (np.random.random(X.shape)*2.0-1.0) * noise_level
    return X,Y

################ generate_parity_majority_sequences #####################################

def generate_parity_majority_sequences(N, count):
    """
    Generate :count: sequences of length :N:.
    If odd # of 1's -> output 1
    else -> output 0
    """
    parity = lambda x: 1 if (x % 2 == 1) else 0
    majority = lambda x: 1 if x > N/2 else 0
    if (count >= 2**N):
        sequences = np.asarray([seq for seq in itertools.product([0,1],repeat=N)])
    else:
        sequences = np.random.choice([0, 1], size=[count, N], replace=True)
    counts = np.count_nonzero(sequences == 1, axis=1)
    # parity each sequence, expand dimensions by 1 to match sequences shape
    if (TASK == 'parity'):
        y = np.expand_dims(np.array([parity(x) for x in counts]), axis=1)
    else: # majority
        y = np.expand_dims(np.array([majority(x) for x in counts]), axis=1)

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
    if (NOISE_LEVEL >= 0.0): # Gaussian mean-zero noise
        input_bias = attractor_tgt_net + NOISE_LEVEL \
                                 * tf.random_normal(tf.shape(attractor_tgt_net))
    else: # Bernoulli dropout
        input_bias = attractor_tgt_net * \
                tf.cast((tf.random_uniform(tf.shape(attractor_tgt_net)) \
                                                  >= -NOISE_LEVEL),tf.float32)

    a_cleaned = run_attractor_net(input_bias)

    # loss is % reduction in noise level
    attr_tgt = tf.tanh(attractor_tgt_net)
    attr_loss = tf.reduce_mean(tf.pow(attr_tgt - a_cleaned,2)) /\
                tf.reduce_mean(tf.pow(attr_tgt - tf.tanh(input_bias),2))

    return attr_loss, input_bias


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
np.random.seed(100)
tf.set_random_seed(100)

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
attr_loss_op, input_bias = \
           attractor_net_loss_function(attractor_tgt_net, params)

# separate out parameters to be optimized
prediction_parameters = params['W'].values() + params['b'].values()
attr_net_parameters = attr_net.values()
attr_net_parameters.remove(attr_net['Wconstr']) # not a real parameter

if (TRAIN_ATTR_WEIGHTS_ON_PREDICTION):
    prediction_parameters += attr_net_parameters

# Define optimizer for prediction task
optimizer_pred = tf.train.AdamOptimizer(learning_rate=LRATE_PREDICTION)
pred_train_op = optimizer_pred.minimize(pred_loss_op, var_list=prediction_parameters)
# Define optimizer for attractor net task
if (N_ATTRACTOR_STEPS > 0):
    optimizer_attr = tf.train.AdamOptimizer(learning_rate=LRATE_ATTRACTOR)
    attr_train_op = optimizer_attr.minimize(attr_loss_op, var_list=attr_net_parameters)
# Evaluate model accuracy
correct_pred = tf.equal(tf.round(Y_), Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    saved_train_acc = []
    saved_test_acc = []
    saved_epoch = []
    # Start training
    for replication in range(N_REPLICATIONS):
        print("********** replication ", replication," **********")
        sess.run(init) # Run the initializer
        if (0):
           writer = tf.summary.FileWriter("./tf.log",sess.graph)
           writer.close()

        [X_train, Y_train, X_test, Y_test] = generate_examples()

        train_prediction_loss = True
        best_train_acc = -1000.
        best_test_acc = 0
        for epoch in range(1, TRAINING_EPOCHS + 2):
            if (epoch-1) % DISPLAY_EPOCH == 0:
                ploss, train_acc, hid_vals = sess.run([pred_loss_op, accuracy, h_net_seq],
                                             feed_dict={X: X_train, Y: Y_train})
                aloss = sess.run(attr_loss_op,feed_dict={attractor_tgt_net: \
                                                           hid_vals.reshape(-1,N_HIDDEN)})
                #print(hid_vals.reshape(-1,N_HIDDEN)[:,:])
                test_acc = sess.run(accuracy, feed_dict={X: X_test, Y: Y_test})
                print("epoch " + str(epoch-1) + ", Loss Pred " + \
                          "{:.4f}".format(ploss) + ", Loss Att " + \
                          "{:.4f}".format(aloss) + ", Train Acc= " + \
                          "{:.3f}".format(train_acc) + ", Test Acc= " + \
                          "{:.3f}".format(test_acc))
                if (train_acc > best_train_acc):
                   best_train_acc = train_acc
                   best_test_acc = test_acc
                if (train_acc == 1.0):
                   break
            if epoch > 1 and LOSS_SWITCH_FREQ > 0 and (epoch-1) % LOSS_SWITCH_FREQ == 0:
               train_prediction_loss = not train_prediction_loss
            batches = get_batches(BATCH_SIZE, X_train, Y_train)
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
        if (REPORT_BEST_TRAIN_PERFORMANCE):
            saved_train_acc.append(best_train_acc)
            saved_test_acc.append(best_test_acc)
        else:
            saved_train_acc.append(train_acc)
            saved_test_acc.append(test_acc)
        if (train_acc == 1.0):
            saved_epoch.append(epoch)

        # print weights
        #for p in attr_net.values():
        #    print (p.name, ' ', p.eval())
    print('********************************************************************')
    print(args)
    print('********************************************************************')
    print('mean train accuracy', np.mean(saved_train_acc))
    print('indiv runs ',saved_train_acc)
    print('mean epoch', np.mean(saved_epoch))
    print('indiv epochs ',saved_epoch)
    print('mean test accuracy', np.mean(saved_test_acc))
    print('indiv runs ',saved_test_acc)

