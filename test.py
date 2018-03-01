import tensorflow as tf
import numpy as np
import random


# MULTIPLE RUNS DO REPLICATE IF I PLACE RANDOM_SEED here
#tf.set_random_seed(1234)

a = tf.get_variable("a",initializer=tf.random_normal([3]))
b = tf.get_variable("b",initializer=tf.random_uniform([3]))

# MULTIPLE RUNS DO NOT REPLICATE IF I PLACE RANDOM_SEED HERE
tf.set_random_seed(1234)

# Repeatedly running this block with the same graph will generate the same
# sequences of 'a' and 'b'.

init = tf.global_variables_initializer()
with tf.Session() as sess1:
    for i in range(0,2):
        sess1.run(init)
        print('a: ',sess1.run(a))  
        print('b: ',sess1.run(b)) 
