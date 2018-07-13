# This is a test of a simple attractor RNN
# with temporal difference learning.

from __future__ import print_function
import itertools
import tensorflow as tf
import numpy as np

# need to set tf seed before defining any tf variables
tf.set_random_seed(222)
np.random.seed(222)

# Training Parameters
training_epochs = 500
n_replications = 1
learning_rate = 0.002
batch_size = 16
display_epoch = 10
N_ATTRACTORS = 20  # number of attractors in a space that has dimensionality N_FEATURES
N_INSTANCES_PER_ATTRACTOR = 50*20/N_ATTRACTORS # half for train, half for test

# Architecture Parameters
N_FEATURES = 20 # dimensions for attractor input space
N_HIDDEN = 20 # dimensions for attractor dynamics
NOISE_LEVEL = .25 # if positive, gaussian noise; if negative % turned off bits
N_ATTRACTOR_STEPS = 5
#### WARNING #### TD VERSION NO GOOD FOR VERSION WITH HIDDEN ATTRACTOR STATE
LAMBDA = -1.0 # temporal difference training
             # to turn off TD, use LAMBDA < 0; otherwise use [0,1] value
             # 0: predict next step; 1: predict final step

ALLOW_DIRECT_COPY = False # include coefficient to copy input to output in
                          # order to have attractor dynamics learn delta

def atanh(x):
    return np.log((1.0+x)/(1.0-x))/2.0

################### generate_attrac_patterns ##################################

def generate_attrac_patterns(n_attractors, n_instances_per_attractor, n_elements):
    tgt = np.random.random([n_attractors, n_elements])*2.0 - 1.0

    x = np.zeros([n_attractors*n_instances_per_attractor, n_elements])
    y = np.zeros([n_attractors*n_instances_per_attractor, n_elements])
    for i in range(n_attractors*n_instances_per_attractor):
        if (NOISE_LEVEL >= 0.0):
            x[i,:] = np.tanh(atanh(tgt[i%n_attractors]) + \
                     np.random.randn(n_elements)*NOISE_LEVEL)
        else:
            x[i,:] = np.tanh(atanh(tgt[i%n_attractors]) * \
                     (np.random.random(n_elements)>= -NOISE_LEVEL))
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

######### attractor_net_ params_init ##########################################

N_INPUT = N_FEATURES

def attractor_net_params_init():

    W = {'hid': tf.get_variable("W_hid",initializer=.01*tf.random_normal([N_HIDDEN,N_HIDDEN])),
         'in':  tf.get_variable("W_in", initializer=.01*tf.random_normal([N_INPUT,N_HIDDEN])),
         'out': tf.get_variable("W_out",initializer=.01*tf.random_normal([N_HIDDEN,N_INPUT])),
         'copy': tf.get_variable("w_copy", initializer=.01*tf.random_normal([1])),
        }
    b = {'in':  tf.get_variable("b_in", initializer=.01*tf.random_normal([N_HIDDEN])),
         'out': tf.get_variable("b_out", initializer=.01*tf.random_normal([N_INPUT])),
        }

    params = {
        'W': W,
        'b': b
    }
    return params

######### attractor_net #######################################################

def attractor_net(X, params, Y):
    W = params['W']
    b = params['b']

    # DEBUG: removing symmetry and zero diag constraint seems to be a help. don't know why
    #W_hid_constr = W['hid'] 
    #W_hid_constr = .5 * (W['hid'] + tf.transpose(W['hid'])) * (1.0 - tf.eye(N_HIDDEN))\
    #                      + tf.diag(tf.abs(tf.diag_part(W['hid'])))
    #Wlowdiag = tf.matrix_band_part(W['hid'], -1, 0) # lower diagonal matrix
    #W_hid_constr = tf.matmul(Wlowdiag, tf.transpose(Wlowdiag))
    #W_hid_constr = tf.matmul(W['hid'], tf.transpose(W['hid']))

    Wdiag = tf.matrix_band_part(W['hid'],0,0) # diagonal
    Wlowdiag = tf.matrix_band_part(W['hid'], -1, 0) - Wdiag # lower diagonal matrix
    W_hid_constr = Wlowdiag + tf.transpose(Wlowdiag) + tf.abs(Wdiag)

    # code is set up in this arrangement to match attr net embedded in RNN
    
    Xatanh = tf.atanh(X)
    hplus = b['in'] + tf.matmul(Xatanh, W['in'])
    h = tf.expand_dims(tf.zeros_like(hplus),0)
    if (LAMBDA >= 0.0):
        eligibility = tf.zeros_like(hplus)
        total_loss = 0.0
    for i in range(1,N_ATTRACTOR_STEPS+1):
        h = tf.concat([h, tf.expand_dims(
                              tf.matmul(tf.tanh(h[i-1,:,:]), W_hid_constr) + hplus,0)], 0)
        if (LAMBDA >= 0.0):
            if (i > 0):
               total_loss = total_loss + tf.reduce_mean(
                           tf.stop_gradient(h[i-1,:,:] - h[i,:,:])*eligibility)
            eligibility = h[i,:,:] + LAMBDA * eligibility

    h_final_net = b['out'] + tf.matmul(h[N_ATTRACTOR_STEPS,:,:], W['out']) 
    if (ALLOW_DIRECT_COPY):
        h_final_net += Xatanh * tf.sigmoid(W['copy'])
    h_final = tf.tanh(h_final_net)

    if (LAMBDA >= 0.0): # temporal difference version
        total_loss = total_loss + tf.reduce_mean(
                tf.stop_gradient(h_final - Y)*eligibility)
    else: # standard 'just the final output' version
        total_loss = tf.reduce_mean(tf.square(h_final-Y))

    return h_final, W_hid_constr, total_loss

######### get_batches #########################################################


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

#################### MAIN CODE ################################################

# tf Graph input
X = tf.placeholder("float", [None, N_FEATURES])
Y = tf.placeholder("float", [None, N_FEATURES])

params = attractor_net_params_init()
Y_, W_hid_constr, attractor_loss = attractor_net(X, params, Y)

# compute relative distance:   |Y_-Y| / |X-Y|
distXY = tf.reduce_sum(tf.square(X-Y))
distYY_ = tf.reduce_sum(tf.square(Y_-Y))

#dummydist = tf.reduce_mean(tf.square(h-Y),axis=[1,2])
#dummyh = tf.squeeze(h[:,0,:])

# loss function and optimizer
final_loss = distYY_ / distXY
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(attractor_loss)

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
    saved_test_final_loss=[]
    for replication in range(n_replications):
        sess.run(init) # Run the initializer

        for epoch in range(1, training_epochs + 2):
            if (epoch-1) % display_epoch == 0:
                train_loss, train_final_loss = sess.run([attractor_loss, final_loss], feed_dict={X: X_train, Y: Y_train})
                test_loss, test_final_loss = sess.run([attractor_loss, final_loss], feed_dict={X: X_test, Y: Y_test})
                print("epoch "+str(epoch-1) + "  loss Train " + 
                          "{:.4f}".format(train_loss) + " scaled " + 
                          "{:.4f}".format(train_final_loss) + " Test " + 
                          "{:.4f}".format(test_loss) + " scaled " + 
                          "{:.4f}".format(test_final_loss))

                #print(params['W']['copy'].eval())

            batches = get_batches(batch_size, X_train, Y_train)
            for (batch_x, batch_y) in batches:
                # Run optimization
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        saved_test_final_loss.append(test_final_loss)
        #print(W_hid_constr.eval())
        #print(params['W']['hid'].eval())
        #print('hidden biases:\n', params['b']['hid'].eval())
        #print('input scale: ',params['b']['scale'].eval())
        #dh, dd = sess.run([dummyh, dummydist], feed_dict={X: batch_x, Y:batch_y})
        #print(dh)
        #print('squared distance per iteration:\n', dd)

    print('Mean relative distance on test: ',np.mean(saved_test_final_loss))
    

