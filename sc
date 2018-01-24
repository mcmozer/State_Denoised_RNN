#/bin/csh

################################################################################
# replicating parity
################################################################################

# no attractor net
#python rnn_attr.py -task parity -seq_len 5 -display_epoch 100 -n_hidden 5 -training_epochs 5000 -n_replications 100 -lrate_attractor .008 -lrate_prediction .008 -n_attractor_steps 0 -arch tanh -report_best_train_performance
#echo PARITY NO ATTRACTOR

# with attractor trained on Gaussian noise
#python rnn_attr.py -task parity -seq_len 5 -display_epoch 100 -n_hidden 5 -training_epochs 5000 -n_replications 100 -lrate_attractor .008 -lrate_prediction .008 -n_attractor_steps 5 -arch tanh -noise_level 0.250 -report_best_train_performance
#echo PARITY GAUSSIAN NOISE

### with attractor trained on Bernoulli dropout
###python rnn_attr.py -task parity -seq_len 5 -display_epoch 100 -n_hidden 5 -training_epochs 5000 -n_replications 100 -lrate_attractor .008 -lrate_prediction .008 -n_attractor_steps 5 -arch tanh -noise_level -0.20 -report_best_train_performance
###echo PARITY BERNOULLI NOISE

# with attractor trained via grad descent
# This version starts with attractor_scale=1.0 in order to succeed at all
#python rnn_attr2.py -task parity -seq_len 5 -display_epoch 100 -n_hidden 5 -training_epochs 5000 -n_replications 100 -lrate_attractor .000 -lrate_prediction .008 -n_attractor_steps 5 -arch tanh -noise_level 0.250 -report_best_train_performance -train_attr_weights_on_prediction 
#echo PARITY TRAIN ATTR WEIGHTS ON PREDICTION TASK

################################################################################

#python rnn_attr.py -task majority -seq_len 12 -display_epoch 100 -training_epochs 2500 -n_replications 100 -lrate_attractor .002 -lrate_prediction .002 -n_attractor_steps 5 -arch tanh

#python rnn_attr.py -task majority -seq_len 12 -display_epoch 100 -training_epochs 2500 -n_replications 100 -lrate_attractor .002 -lrate_prediction .002 -n_attractor_steps 0 -arch tanh

################################################################################

#python rnn_attr.py -arch GRU -display_epoch 50 -lrate_attractor 0.002 -lrate_prediction 0.002 -n_attractor_steps 0 -n_hidden 10 -n_replications 100 -noise_level=0.25 -report_best_train_performance -seq_len 20 -task kazakov -no-train_attr_weights_on_prediction -training_epochs 2000
#echo KAZAKOV NO ATTRACTOR WITH GRU

#python rnn_attr.py -arch GRU -display_epoch 50 -lrate_attractor 0.002 -lrate_prediction 0.002 -n_attractor_steps 5 -n_hidden 10 -n_replications 100 -noise_level=0.25 -report_best_train_performance -seq_len 20 -task kazakov -no-train_attr_weights_on_prediction -training_epochs 4000 -loss_switch_frequency 5
#echo KAZAKOV ATTR NET loss switch every 5 epoch 4000 epochs due to switch
