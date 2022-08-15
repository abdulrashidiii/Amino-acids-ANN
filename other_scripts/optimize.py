"""

Features
- restores trained neural network model
- imports data on minima from fine grid
- peforms optimization at each minima point
- reports newly found minima points

"""

import tensorflow as tf
import numpy as np
from scipy.optimize import fmin
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

    pred = sess.run(output, feed_dict={xs:x}).ravel()

  return pred[0]

######## CONVERT ANGLES TO RANGE (-180, 180) #######

def convert(angle):
  if angle < -180:
    angle = angle + 360
  elif angle > 180:
    angle = angle - 360
  return angle

vconvert = np.vectorize(convert)

####################################################

############ IMPORT DATA ON MINIMA FROM FINE GRID ######################

guess_minima = np.loadtxt('minima.dat', skiprows=3, usecols=[2,4])
guess_energy = np.loadtxt('minima.dat', skiprows=3, usecols=[6])

########################################################################

############# DO OPTIMIZATION AT EACH MINIMA AND WRITE TO OUTPUT FILE ########################

with open('new_minima.dat', 'w') as f:
  with open('better_minima.dat', 'w') as g:
    g.write('Number of minima found: %d\n' % len(guess_minima))
    g.write('Conformation     phi         psi      RelEnergy\n')
    g.write('-' * 48 + '\n')

    opt_angles = []
    opt_energies = []

    # iterate over each minima found and do optimization at that point
    for i,j in zip(guess_minima, guess_energy):
      xopt, fopt, _, _, _ = fmin(trained_model, x0=i, disp=False, full_output=True)
      xopt = vconvert(xopt)
      opt_angles.append(xopt)
      opt_energies.append(fopt)

    opt_energies = np.array(opt_energies)
    opt_energies = opt_energies - opt_energies.min()
  
    for i,j,k,l in zip(guess_minima, guess_energy, opt_angles, opt_energies):
      f.write('Old: %s   %10.5f,   %10.5f,   %10.5f\n' % (conf(i[0], i[1]), i[0], i[1], j))
      f.write('New: %s   %10.5f,   %10.5f,   %10.5f\n\n' % (conf(k[0], k[1]), k[0], k[1], l))
      g.write('%s\t\t\t:\t%8.3f\t\t,\t%8.3f\t:%10.5f\n' % (conf(k[0], k[1]), k[0], k[1], l))

##############################################################################################

print('Runtime:', datetime.now() - t0)
