# This is a test of a simple attractor RNN

from __future__ import print_function
import itertools
import tensorflow as tf
import numpy as np

# Training Parameters
training_epochs = 200
n_replications = 10
learning_rate = 0.002
batch_size = 16
display_epoch = 10
N_INSTANCES_PER_ATTRACTOR = 50 # half for train, half for test
N_ATTRACTORS = 20

# Architecture Parameters
N_FEATURES = 20
NOISE_LEVEL = 0.5
N_ATTRACTOR_STEPS = 10

def atanh(x):
    return np.log((1.0+x)/(1.0-x))/2.0

################### generate_attrac_patterns ############################################

def generate_attrac_patterns(n_attractors, n_instances_per_attractor, n_elements):
    tgt = np.random.random([n_attractors, n_elements])*2.0 - 1.0

    x = np.zeros([n_attractors*n_instances_per_attractor, n_elements])
    y = np.zeros([n_attractors*n_instances_per_attractor, n_elements])
    for i in range(n_attractors*n_instances_per_attractor):
        x[i,:] = np.tanh(atanh(tgt[i%n_attractors]) + np.random.randn(n_elements)*NOISE_LEVEL)
        y[i,:] = tgt[i%n_attractors]

    return x,y



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

######### BEGIN TANH RNN ########################################################

N_INPUT = N_OUTPUT = N_HIDDEN = N_FEATURES

def attractor_net_params_init():

    W = {'hid': tf.get_variable("W_hid", initializer=tf.zeros([N_HIDDEN,N_HIDDEN])),
         'out': tf.get_variable("W_out", initializer=tf.zeros([N_HIDDEN,N_OUTPUT])),
        }
    b = {'hid': tf.get_variable("b_hid", initializer=tf.zeros([N_HIDDEN])),
         'out': tf.get_variable("b_out", initializer=tf.zeros([N_OUTPUT])),
        }

    params = {
        'W': W,
        'b': b
    }
    return params

def attractor_net(X, params):
    W = params['W']
    b = params['b']

    # MIKE: removing symmetry and zero diag constraint seems to be a help. don't know why
    #W_hid_constr = (W['hid'] + tf.transpose(W['hid'])) * (1.0 - tf.eye(N_HIDDEN))
    W_hid_constr = W['hid'] 
    #W_hid_constr = (W['hid'] ) * (1.0 - tf.eye(N_HIDDEN))
    #W_hid_constr = (W['hid'] + tf.transpose(W['hid'])) 

    # code is set up in this arrangement to match attr net embedded in RNN
    
    b_input = tf.atanh(X)
    h = tf.zeros(tf.shape(X))  # NOTE: resetting at 0, versus h = X
    for _ in range(N_ATTRACTOR_STEPS):
        # MIKE: including bias seems to be a big help. don't understand why
        h = tf.matmul(tf.tanh(h), W_hid_constr) + b_input + b['hid']

    # MIKE: this seems to be a big help. don't understand why
    #h = tf.tanh(tf.matmul(h, W['out']) + b['out'])
    h = tf.tanh(h)
    #h = tf.matmul(h, W['out']) + b['out']
    return h

######### END TANH RNN ##########################################################


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

#################### MAIN CODE ##########################################################

tf.set_random_seed(222)
np.random.seed(222)

# tf Graph input
X = tf.placeholder("float", [None, N_FEATURES])
Y = tf.placeholder("float", [None, N_FEATURES])

params = attractor_net_params_init()
Y_ = attractor_net(X, params)

# compute relative distance:   |Y_-Y| / |X-Y|
distXY = tf.sqrt(tf.reduce_sum(tf.square(X-Y), axis=1))
distYY_ = tf.sqrt(tf.reduce_sum(tf.square(Y_-Y), axis=1))
rel_dist = tf.reduce_mean(distYY_ / distXY)

# Define loss and optimizer
# Since only one output unit,
loss_op = tf.reduce_mean(tf.pow(Y_ - Y, 2))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

X_all, Y_all = generate_attrac_patterns(N_ATTRACTORS, N_INSTANCES_PER_ATTRACTOR*2, 
                                        N_FEATURES)
X_train = X_all[:N_ATTRACTORS*N_INSTANCES_PER_ATTRACTOR,:]
X_test = X_all[N_ATTRACTORS*N_INSTANCES_PER_ATTRACTOR:,:]
Y_train = Y_all[:N_ATTRACTORS*N_INSTANCES_PER_ATTRACTOR,:]
Y_test = Y_all[N_ATTRACTORS*N_INSTANCES_PER_ATTRACTOR:,:]

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    saved_test_rel_dist=[]
    for replication in range(n_replications):
        sess.run(init) # Run the initializer

        for epoch in range(1, training_epochs + 2):
            if (epoch-1) % display_epoch == 0:
                [train_loss, train_rel_dist] = \
                      sess.run([loss_op, rel_dist], feed_dict={X: X_train, Y: Y_train})
                [test_loss, test_rel_dist] = \
                      sess.run([loss_op, rel_dist], feed_dict={X: X_test, Y: Y_test})
                print("epoch "+str(epoch-1) + "  loss/reldist Train " + \
                          "{:.4f}".format(train_loss) + " " + \
                          "{:.4f}".format(train_rel_dist) + " Test " + \
                          "{:.4f}".format(test_loss) + " " +\
                          "{:.4f}".format(test_rel_dist))

            batches = get_batches(batch_size, X_train, Y_train)
            for (batch_x, batch_y) in batches:
                # Run optimization
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        saved_test_rel_dist.append(test_rel_dist)

        #W = params['W']
        #W_hid_constr = (W['hid'] + tf.transpose(W['hid'])) * (1.0 - tf.eye(N_HIDDEN))
        #W_hid_constr = W['hid'] 
        #print(sess.run(params['b']['hid']))

    print('Mean relative distance on test: ',np.mean(saved_test_rel_dist))
    

