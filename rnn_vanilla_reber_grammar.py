import tensorflow as tf
import numpy as np
from tf_helpers import batch_shuffle, early_stop_tracker, generate_xor_sequences
from tf_models import   RNN_vanilla, \
                        LSTM, \
                        LSTM_tensorflow, \
                        LSTM_raw_params_init,\
                        RNN_params_init, \
                        LSTM_attractor

from finite_state_machine import generate_grammar_dataset, embed_one_hot
"""
TODOs:
CODE:
- do stopping critereon for training:
- proper batches
* access to CU's GPU cluster
* xavier initializations
* setup the experiment for sequence length and training on average of runs:
    for length:
        for trials:
            run tf simulation and record accuracy once stopped
* Attractor Nets:
    Code:
    1) :A: - for how many attractor unrolls to perform
    2) The attractor cell (return losses from each iteration of the cell)
    3) ensure symmetric weights (just perform the transform at each sequence start)

THEORY
* is it like a convergence criterion for Markov Chains
* read literature on attractor nets
* continuous time discrete update. either neurons are cont-s or discrete.
    discrete, discrete value - hopfield 82
    continuous time, continuous activation value - hopfield 84
    discrete time, continous activation avalue - theorem? (constraint on symmetry or constraint that's stronger)
* some case when it ends up in corners (every unit is either a zero or one)
* temporal difference learning? TD(1), TD(0), TD(\lambda)

* We are essentially bulding a deeper network - (#attractor_unrolls * #recurrent steps) deep, so truncated backprop?
* Hebian learning?(1) (Hinton & Ba: fast weight, slow weights paper. 3.1. - layer normalization. )
* batch normalization is the most important thing you could do for training. **Layer normalization** - does it relate? 
* does layer normalization screw up attractor dynamics?  
"""


# Training Parameters
n_examples = 1000
n_trials = 1
timesteps = 15 # # number of elements in the sequences generated
epochs_num = 400


# Experiments Parameters:
model_type = "LSTM_raw" # CHOICES ARE: vanilla, LSTM_raw, LSTM_tensorflow, LSTM_attractor
timesteps_arr = [5]

# Network Parameters
ops = {'hid': 1,
           'in': 1,
           'out': 1,
            'batch_size':n_examples, #since the sequences are 1-dimensional it's easier to just run them all at once
            # TODO: would the network just learn to cycle (a simple hack to recreate identity)
            'n_attractor_iterations': 3,
            'attractor_lambda': 0.1
       }



def run_simulation(timesteps, model_type, n_examples, ops, print_freq=100):
    # # dataset:
    # X_train, Y_train = generate_xor_sequences(timesteps, n_examples)
    # X_test, Y_test = generate_xor_sequences(timesteps, n_examples)

    # x_train_real is the original sequences, X_train is 1-hot representation, timesteps is NOT const (varies due to random nature of
    # sequence generation from the grammar
    x_train_real, Y_train, x_test_real, Y_test, X_train, X_test, timesteps, unique_chars = generate_grammar_dataset(n_examples)
    ops['in'] = len(unique_chars) # change the input dimension of the network

    # model training:
    tf.reset_default_graph()

    # input-output
    X = tf.placeholder("float", [None, timesteps, ops['in']])
    Y = tf.placeholder("float", [None, ops['out']])

    # choose the RNN model
    if model_type == "vanilla":
        Y_ = RNN_vanilla(X, RNN_params_init(ops), ops)
    elif model_type == "LSTM_raw":
        Y_ = LSTM(X, LSTM_raw_params_init(ops), ops)
    elif model_type == "LSTM_tensorflow":
        Y_ = LSTM_tensorflow(X, LSTM_raw_params_init(ops), ops)
    elif model_type == "LSTM_attractor":
        Y_, att_loss = LSTM_attractor(X, LSTM_raw_params_init(ops), ops)

    # Define loss and optimizer
    # Since only one output unit,
    loss_op = tf.losses.mean_squared_error(Y_, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)

    if model_type == "LSTM_attractor":
        att_loss_normalized = tf.reduce_sum(att_loss, axis=[0,1], keep_dims=False)/timesteps
        att_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             "LSTM_att")
        att_train_op = optimizer.minimize(att_loss_normalized, var_list = att_vars)
        lstm_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             "LSTM")
        lstm_train_op = optimizer.minimize(loss_op, var_list=lstm_vars)
    else:
        train_op = optimizer.minimize(loss_op)

    # Evaluate model
    correct_pred = tf.equal(tf.round(Y_), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    writer = tf.summary.FileWriter(logs_path, graph=sess.graph)

    # Optimization stage
    loss_hist = []  # used for tracking early stopping criterion
    acc_hist = []
    patience_cnt = 0
    for epoch in range(epochs_num):

        # shuffle and sample training set:
        losses = []
        accs = []
        batches_x, batches_y = batch_shuffle(ops['batch_size'], X_train, Y_train)
        for i in range(len(batches_x)):
            # Run optimization
            if model_type == "LSTM_attractor":
                _, _, loss, acc, att_loss = sess.run([lstm_train_op, att_train_op, loss_op, accuracy, att_loss_normalized], feed_dict={X: batches_x[i], Y: batches_y[i]})
                print "att {}, lstm {}".format(att_loss, loss)
            else:
                _, loss, acc = sess.run([train_op, loss_op, accuracy], feed_dict={X: batches_x[i], Y: batches_y[i]})

            accs.append(acc)
            losses.append(loss)

        # accumulate values over the whole epoch:
        loss_hist.append(np.mean(losses))
        acc_hist.append(np.mean(accs))
        if epoch % print_freq == 0:
            print("[len:{}, model:{}] ".format(timesteps, model_type) + \
                  "Epoch " + str(epoch) + ", Loss= " + \
                  "{:.6f}".format(loss_hist[epoch]) + ", Accuracy= " + \
                  "{:.3f}".format(acc_hist[epoch]))

        # early stopping:
        patience_cnt = early_stop_tracker(loss_hist, epoch, patience_cnt, min_delta=1e-6)
        if (epoch > 500) and (patience_cnt > 500 or acc_hist[-1] == 1.0):
            print "stopped early"
            break

    # TODO: save model, save errors etc
    print("Optimization Finished!")

    # test performance
    test_accs = []
    batches_x, batches_y = batch_shuffle(ops['batch_size'], X_test, Y_test)
    for i in range(len(batches_x)):
        acc = sess.run(accuracy, feed_dict={X: batches_x[i], Y: batches_y[i]})
        print(acc)
        test_accs.append(acc)
    print("Testing Accuracy:", np.mean(test_accs))

    return np.mean(test_accs)



logs_path = '/Users/denis/Dropbox/School/2017fall/ta_neural_net/hw6/tf_board/run1'
accs = []
std_devs = []
full_list = []
for timesteps in timesteps_arr:
    print timesteps
    sample_accs = []
    for trial in range(n_trials):
        acc = run_simulation(timesteps=timesteps,
                   model_type=model_type,
                   n_examples=n_examples,
                   ops=ops,
                   print_freq=4)
        sample_accs.append(acc)
    accs.append(np.mean(sample_accs))
    std_devs.append(np.std(sample_accs))
    full_list.append(sample_accs)





