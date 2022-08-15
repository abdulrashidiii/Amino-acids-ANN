"""

Features
- restores trained neural network model
- imports data on minima from fine grid
- peforms optimization at each minima point
- reports newly found minima points

"""

import tensorflow as tf
import numpy as np
from datetime import datetime
from getextrema3d import conf

t0 = datetime.now()

############### FULLY CONNECTED HIDDEN LAYER ########################

def fc_layer(input_data, input_dim, output_dim, name='fc'):
  with tf.name_scope(name):
    w = tf.Variable(tf.random_normal(shape=[output_dim,input_dim],
      mean=0.0,
      stddev=1.0/np.sqrt(input_dim)),
      name='W')
    b = tf.Variable(tf.zeros([output_dim, 1]),
        name='b')
    layer = tf.add(tf.matmul(w, input_data), b)
    act = tf.nn.sigmoid(layer)
    tf.summary.histogram('weights', w)
    tf.summary.histogram('biases', b)
    tf.summary.histogram('activations', act)
    return act

######################################################################

#################### REGRESSION OUTPUT LAYER #########################

def output_layer(input_data, input_dim, output_dim, name='output'):
  with tf.name_scope(name):
    w = tf.Variable(tf.random_normal(shape=[output_dim,input_dim],
      mean=0.0,
      stddev=1.0/np.sqrt(input_dim)),
      name='W')
    b = tf.Variable(tf.zeros([output_dim, 1]),
        name='b')
    output = tf.add(tf.matmul(w, input_data), b)

    tf.summary.histogram('weights', w)
    tf.summary.histogram('biases', b)
    tf.summary.histogram('outputs', output)

    return output

#####################################################################

######### FIX FLOAT ERRORS ############

def FixZeros(values):
  for i in range(len(values)):
    if np.allclose(values[i],0):
      values[i] = 0

#######################################

##### CONTROL PARAMETERS OF NEURAL NETWORK ######

tf.reset_default_graph()
n_inputs = 4
n_hidden1 = 51
n_outputs = 1

#################################################

######## INITIAL TENSORFLOW NODES ###############

xs = tf.placeholder('float32', name='X')
ys = tf.placeholder('float32', name='y')

#################################################

#################### ONE HIDDEN LAYER ###########################

fc1 = fc_layer(xs, n_inputs, n_hidden1, name='FC1')
output = output_layer(fc1, n_hidden1, n_outputs, name='Output')

#################################################################

######### SAVER CLASS #######

saver = tf.train.Saver()

#############################

########### FUNCTION OF TRAINED NEURAL NETWORK #################

def trained_model(angles):
  """
  Takes in a phi and a psi value
  returns the predicted energy in kcal/mol
  """

  phi, psi = angles
  del angles

  with tf.Session() as sess:
    saver.restore(sess, '../models/model_final_layer1_%d.ckpt' % n_hidden1)
  
    phi_sin = np.sin(np.radians(phi))
    psi_sin = np.sin(np.radians(psi))
    phi_cos = np.cos(np.radians(phi))
    psi_cos = np.cos(np.radians(psi))
    x = np.array([phi_sin, psi_sin, phi_cos, psi_cos])
    for i in range(len(x)):
      if np.allclose(x[i], 0):
        x[i] = 0
    x = x.reshape(-1,1)

    act = sess.run(fc1, feed_dict={xs:x}).ravel()

  return act

################################################################

x = np.arange(-180,181,15)
x = np.meshgrid(x,x)
angles = list(zip(x[1].reshape(-1), x[0].reshape(-1)))

activations = []

for i in angles:
  fc1_act = trained_model(i)
  activations.append(fc1_act)

activations = np.array(activations)

np.savetxt('layer1_activations_coarse_grid.csv', activations, delimiter=',')

print('Runtime:', datetime.now() - t0)
