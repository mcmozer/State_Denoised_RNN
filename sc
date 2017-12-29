#/bin/csh

################################################################################
# replicating parity
################################################################################

# no attractor net
#python rnn_attr.py -task parity -seq_len 5 -display_epoch 100 -n_hidden 5 -training_epochs 5000 -n_replications 100 -lrate_attractor .008 -lrate_prediction .008 -n_attractor_steps 0 -arch tanh -report_best_train_performance
#echo PARITY NO ATTRACTOR

# with attractor trained on Gaussian noise
#python rnn_attr_td1.py -task parity -seq_len 5 -display_epoch 100 -n_hidden 5 -training_epochs 5000 -n_replications 100 -lrate_attractor .008 -lrate_prediction .008 -n_attractor_steps 5 -arch tanh -noise_level 0.250 -report_best_train_performance
#echo PARITY GAUSSIAN NOISE

# with attractor trained on Bernoulli dropout
python rnn_attr_td1.py -task parity -seq_len 5 -display_epoch 100 -n_hidden 5 -training_epochs 5000 -n_replications 100 -lrate_attractor .008 -lrate_prediction .008 -n_attractor_steps 5 -arch tanh -noise_level -0.20 -report_best_train_performance
echo PARITY BERNOULLI NOISE



################################################################################

#python rnn_attr.py -task majority -seq_len 12 -display_epoch 100 -training_epochs 2500 -n_replications 100 -lrate_attractor .002 -lrate_prediction .002 -n_attractor_steps 5 -arch tanh

#python rnn_attr.py -task majority -seq_len 12 -display_epoch 100 -training_epochs 2500 -n_replications 100 -lrate_attractor .002 -lrate_prediction .002 -n_attractor_steps 0 -arch tanh

