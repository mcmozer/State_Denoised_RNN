#/bin/csh

#python rnn_attr_sep_train.py -task majority -seq_len 12 -display_epoch 100 -training_epochs 2500 -n_replications 100 -lrate_attractor .002 -lrate_prediction .002 -n_attractor_steps 5 -arch tanh

python rnn_attr_sep_train.py -task majority -seq_len 12 -display_epoch 100 -training_epochs 2500 -n_replications 100 -lrate_attractor .002 -lrate_prediction .002 -n_attractor_steps 0 -arch tanh
