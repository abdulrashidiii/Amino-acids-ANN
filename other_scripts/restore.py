import tensorflow as tf
import numpy as np
from datetime import datetime

t0 = datetime.now()

figures_path = 'figures/'
models_path = 'models/'

def fc_layer(input_data, input_dim, output_dim, name='fc'):
  with tf.name_scope(name):
    w = tf.Variable(tf.random_normal(shape=[output_dim,input_dim],
      mean=0.0,
      stddev=1.0/np.sqrt(input_dim)),
      name='W')
    b = tf.Variable(tf.zeros([output_dim,1]),
        name='b')
    layer = tf.add(tf.matmul(w, input_data), b)
    act = tf.nn.sigmoid(layer)
    tf.summary.histogram('weights', w)
    tf.summary.histogram('biases', b)
    tf.summary.histogram('activations', act)
    return act

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

def FixZeros(values):
  for i in range(len(values)):
    if np.allclose(values[i],0):
      values[i] = 0

def generate_grid(x, fname):
  # full ramachandran contour plot
  x = np.meshgrid(x,x)
  one = np.sin(np.radians(x[1].reshape(-1)))
  FixZeros(one)
  two = np.sin(np.radians(x[0].reshape(-1)))
  FixZeros(two)
  three = np.cos(np.radians(x[1].reshape(-1)))
  FixZeros(three)
  four = np.cos(np.radians(x[0].reshape(-1)))
  FixZeros(four)
  input_vec = np.array(list(zip(one,two,three,four))).T

  print('Prediction')
  pred = sess.run(output, feed_dict={xs:input_vec}).ravel()
  pred = pred - pred.min()

  grid = np.array(list(zip(x[1].reshape(-1), x[0].reshape(-1), pred)))
  print('Printing to output file')
  np.savetxt(fname, grid, delimiter=',')

n_inputs = 4
n_hidden1 = 51
n_outputs = 1

xs = tf.placeholder('float32', name='X')
ys = tf.placeholder('float32', name='y')

fc1 = fc_layer(xs, n_inputs, n_hidden1, name='FC1')
output = output_layer(fc1, n_hidden1, n_outputs, name='Output')

saver = tf.train.Saver()

with tf.Session() as sess:
  saver.restore(sess, models_path + 'model_final_layer1_%d.ckpt' % n_hidden1)
  print('Model Restored.')

  generate_grid(np.arange(-180,181,1), 'prediction_fine_grid.dat')
  generate_grid(np.arange(-180,181,15), 'prediction_coarse_grid.dat')

print('Runtime:', datetime.now() - t0)
