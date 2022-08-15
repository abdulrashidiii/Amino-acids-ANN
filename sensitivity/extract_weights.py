import tensorflow as tf
import numpy as np
from scipy.optimize import fmin
from datetime import datetime
#from getextrema3d import conf

t0 = datetime.now()

# FULLY CONNECTED HIDDEN LAYER
def fc_layer(input_data, input_dim, output_dim, name='fc'):
  with tf.name_scope(name):
    w = tf.Variable(tf.random.normal(shape=[output_dim,input_dim],
      mean=0.0,
      stddev=1.0/np.sqrt(input_dim)),
      name='W')
    b = tf.Variable(tf.zeros([output_dim, 1]),
        name='b')
    layer = tf.add(tf.matmul(w, input_data),b)
    act = tf.nn.sigmoid(layer)
    tf.compat.v1.summary.histogram('weights', w)
    tf.compat.v1.summary.histogram('biases', b)
    tf.compat.v1.summary.histogram('activations', act)
    return act

# REGRESSION OUTPUT LAYER
def output_layer(input_data, input_dim, output_dim, name='output'):
  with tf.name_scope(name):
    w = tf.Variable(tf.random.normal(shape=[output_dim,input_dim],
      mean=0.5,
      stddev=1.0/np.sqrt(input_dim)),
      name='W',
      constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
    #b = tf.Variable(tf.zeros([output_dim, 1]),
    #    name='b')
    output = tf.matmul(w, input_data)

    tf.compat.v1.summary.histogram('weights', w)
    #tf.compat.v1.summary.histogram('biases', b)
    tf.compat.v1.summary.histogram('outputs', output)

    return output

# FIX FLOAT ERRORS
def FixZeros(values):
  for i in range(len(values)):
    if np.allclose(values[i],0):
      values[i] = 0

# PARAMETERS OF NEURAL NETWORK
n_inputs = 4
n_hidden1 = 51
n_outputs = 1


# INITIAL TENSORFLOW NODES
xs = tf.compat.v1.placeholder('float32', name='X')
ys = tf.compat.v1.placeholder('float32', name='y')

# ONE HIDDEN LAYER
fc1 = fc_layer(xs, n_inputs, n_hidden1, name='FC1')
output = output_layer(fc1, n_hidden1, n_outputs, name='Output')

# SAVER CLASS
saver = tf.compat.v1.train.Saver()

# EXTRACT WEIGHTS
with tf.compat.v1.Session() as sess:
  saver.restore(sess, '../models/model_final_layer1_%d_group_4.ckpt' % n_hidden1)
  for i in tf.compat.v1.trainable_variables():
    fname = i.name.split('/')
    np.savetxt('%s-%s.csv' % (fname[0], fname[1][0]), i.eval(), delimiter=',')

print('Runtime:', datetime.now() - t0)
