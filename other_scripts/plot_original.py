from sklearn.model_selection import train_test_split
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

# ALL AMINO ACIDS
#aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
#    'N', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y']
#
#data = np.empty((0,4))
#for i in range(len(aas)):
#  aa_data = np.loadtxt(aas[i]+'.csv', delimiter=',')
#  third = np.ones(len(aa_data)) * i
#  data = np.append(data, list(zip(third, aa_data[:,0], aa_data[:,1], aa_data[:,2])), axis=0)
#
#print(data.shape)

# fix floating point errors
def FixZeros(values):
  for i in range(len(values)):
    if np.allclose(values[i], 0):
      values[i] = 0

# importing data on Glycine only
data = pd.read_csv('../../../../data/G.csv', header=None)
data.columns = ['Phi', 'Psi', 'Energy']
data['Energy'] = (data['Energy'] - data['Energy'].min()) * 2625.499

data['SinPhi'] = np.sin(np.radians(data['Phi']))
FixZeros(data['SinPhi'].values)
data['SinPsi'] = np.sin(np.radians(data['Psi']))
FixZeros(data['SinPsi'].values)
data['CosPhi'] = np.cos(np.radians(data['Phi']))
FixZeros(data['CosPhi'].values)
data['CosPsi'] = np.cos(np.radians(data['Psi']))
FixZeros(data['CosPsi'].values)

data = data.drop(['Phi', 'Psi'], axis=1)

# creating a shuffled copy of the data
df_train = data.sample(frac=1).reset_index(drop=True)
#df_train = data.copy()

# generate training set
X_train = df_train.drop(['Energy'], axis=1).as_matrix()
y_train = df_train['Energy'].values.reshape(-1,1)

# generate testing set
X_test = data.drop(['Energy'], axis=1).as_matrix()
y_test = data['Energy'].values.reshape(-1,1)

# contour plot function
def plot(x, data, fname):
  # Plot contour of original data
  #plt.figure(figsize=(11,8.5))
  plt.figure()
  
  # colorbar ticks
  cb_min = np.floor(np.min(data))
  cb_max = np.ceil(np.max(data))
  cbar_ticks = np.arange(0,cb_max+0.5,20)
  
  # data smoothing
  train_contour = gaussian_filter(data.reshape(len(x),len(x)).T, 1)
  
  # colormap used
  colormap = 'coolwarm'
  
  # filled contour
  plt.contourf(x, 
      x, 
      train_contour, 
      cmap=colormap, 
      extent=[x[0],x[-1],x[0],x[-1]], 
      levels=np.arange(cb_min,cb_max+0.5,5))

  # colorbar
#  cb = plt.colorbar(ticks = cbar_ticks)
  #cb.set_label(r'$$', fontsize=18)
  
  # contour lines
  C = plt.contour(x, 
      x, 
      train_contour, 
      extent=[x[0],x[-1],x[0],x[-1]], 
      levels=np.arange(cb_min,cb_max+0.5,5))
  
  #  # contour line labels
  #  plt.clabel(C, inline=1, fontsize=12)
  
  #plt.gca().invert_xaxis()
  
#  cfaxes = plt.gca()
#  plt.sca(cb.ax)
#  plt.clim(vmin=0, vmax=cb_max)
#  plt.yticks(size='18')
  
#  plt.sca(cfaxes)
  plt.xticks([], size='22')
  plt.yticks([], size='22')

#  plt.xlabel(r'$\phi$', fontsize=30)
#  plt.ylabel(r'$\psi$', fontsize=30)
  plt.xlim(-180,180)
  plt.ylim(-180,180)

  plt.savefig(fname, bbox_inches='tight', transparent=True, pad_inches=0, dpi=300)
#  plt.show()
  plt.close()

plot(np.arange(-180,181,15), y_test, './original.png')

#def transform_input(input_data, name='transform input'):
#  with tf.name_scope(name):
#  radian = tf.multiply(input_data,
#      tf.convert_to_tensor(np.pi/180., dtype='float64'))
#  sin = tf.sin(radian)
#  cos = tf.cos(radian)
#
## fully connected layer
#def fc_layer(input_data, input_dim, output_dim, name='fc'):
#  """
#  weight initialization from normal distribution
#  sigmoid activation function
#  """
#  with tf.name_scope(name):
#    w = tf.Variable(tf.random_normal(shape=[input_dim,output_dim],
#      mean=0.0,
#      stddev=1.0/np.sqrt(input_dim)),
#      name='W')
#    b = tf.Variable(tf.zeros([output_dim]),
#        name='b')
#    layer = tf.add(tf.matmul(input_data,w), b)
#    act = tf.nn.sigmoid(layer)
#    tf.summary.histogram('weights', w)
#    tf.summary.histogram('biases', b)
#    tf.summary.histogram('activations', act)
#    return act
#
## output layer for regression
#def output_layer(input_data, input_dim, output_dim, name='output'):
#  with tf.name_scope(name):
#    w = tf.Variable(tf.random_normal(shape=[input_dim,output_dim],
#      mean=0.0,
#      stddev=1.0/np.sqrt(input_dim)),
#      name='W')
#    b = tf.Variable(tf.zeros([output_dim]),
#        name='b')
#    output = tf.add(tf.matmul(input_data,w),   b)
#
#    tf.summary.histogram('weights', w)
#    tf.summary.histogram('biases', b)
#    tf.summary.histogram('outputs', output)
#
#    return output
#
#def neural_net_model(X_data, input_dim, n_hidden1):
#
#  """
#  One hidden layer
#  Sigmoid activation function
#  Normal distribution initialization
#  """
#
#  with tf.name_scope('neural_net'):
#    W_1 = tf.Variable(tf.random_normal(shape=[input_dim,n_hidden1],
#      mean=0.0,
#      stddev=1.0/np.sqrt(input_dim)), 
#      name='W_1')
#    b_1 = tf.Variable(tf.zeros([n_hidden1]), 
#        name='b_1')
#    layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
#    layer_1 = tf.nn.sigmoid(layer_1)
#
#    W_O = tf.Variable(tf.random_normal(shape=[n_hidden1,1],
#      mean=0.0,
#      stddev=1.0/np.sqrt(input_dim)), 
#      name='W_O')
#    b_O = tf.Variable(tf.zeros([1]), 
#        name='b_O')
#    output = tf.add(tf.matmul(layer_1,W_O), b_O)
#
#  return output, W_O

## contour plot of Gaussian data
#plot(np.arange(-180,181,15), y_test, 'original.png')

##for first in [30,40,51,60,70]:
#for first in [51]:
#  with open('nn_layer1_%d.log' % first, 'w') as f:
#    tf.reset_default_graph()
#
#    # control parameters
#    n_inputs = 4
#    n_hidden1 = first
#    n_outputs = 1
#    n_epochs = 2000
#
#    # placeholders
#    xs = tf.placeholder('float32', name='X')
#    ys = tf.placeholder('float32', name='y')
#
#    # neural network model
#    fc1 = fc_layer(xs, n_inputs, n_hidden1, name='FC1')
#    output = output_layer(fc1, n_hidden1, n_outputs, name='Output')
#    
#    # MSE cost function
#    with tf.name_scope('Cost'):
#      cost = tf.reduce_mean(tf.square(output-ys))
#
#    tf.summary.scalar('Cost', cost)
#    
#    # SGD Adam optimizer
#    with tf.name_scope('Train'):
#      train = tf.train.AdamOptimizer(0.001).minimize(cost)
#    
#    c_t = []
#    c_test = []
#    
#    # create a node to initialize all variables
#    init = tf.global_variables_initializer()
#    
#    # save our trained model
#    saver = tf.train.Saver()
#    
#    # Execution Phase
#    with tf.Session() as sess:
#      init.run()
#    
#      y_t = y_train
#
#      merged_summary = tf.summary.merge_all()
#      writer = tf.summary.FileWriter('./tensorboard/%d' % n_hidden1)
#      writer.add_graph(sess.graph)
#    
#      for i in range(n_epochs):
#        for j in range(len(X_train)):
#          sess.run([cost,train], 
#              feed_dict={xs:X_train[j,:].reshape(1,4), ys:y_train[j]})
#
#        if i % 5 == 0:
#          s = sess.run(merged_summary, feed_dict={xs:X_train, ys:y_train})
#          writer.add_summary(s,i)
#    
#        pred = sess.run(output, feed_dict={xs:X_train})
#    
#        c_t.append(sess.run(cost, feed_dict={xs:X_train, ys:y_train}))
#        c_test.append(sess.run(cost, feed_dict={xs:X_test, ys:y_test}))
#        f.write('Epoch : %d Training Cost : %f\n' % (i, c_t[i]))
#        f.write('Epoch : %d Testing Cost : %f\n' % (i, c_test[i]))
#    
#      pred = sess.run(output, feed_dict={xs:X_test})
#    
#      for i in range(y_test.shape[0]):
#        #print('Original :', y_test[i], 'Predicted :', pred[i])
#        f.write('Original : %f Predicted : %f\n' % (y_test[i], pred[i]))
#    
#      # Figures
#
#      # predicted contour plot
#      plot(np.arange(-180,181,15), pred, 'prediction_layer1_%d.png' % n_hidden1)
#
#      # delta contour plot
#      plot(np.arange(-180,181,15), np.absolute(y_test - pred), 'delta_layer1_%d.png' % n_hidden1)
#
#      # cost vs epoch line plot
#      plt.plot(c_test, c='k')
#      plt.xlabel('Epoch')
#      plt.ylabel('Cost')
#      plt.savefig('cost_layer1_%d.png' % n_hidden1)
#      plt.close()
#
#      # full ramachandran contour plot -360,360
#      x = np.arange(-360,361,15)
#      x = np.meshgrid(x,x)
#      one = np.sin(np.radians(x[1].reshape(-1)))
#      FixZeros(one)
#      two = np.sin(np.radians(x[0].reshape(-1)))
#      FixZeros(two)
#      three = np.cos(np.radians(x[1].reshape(-1)))
#      FixZeros(three)
#      four = np.cos(np.radians(x[0].reshape(-1)))
#      FixZeros(four)
#      x = np.array(list(zip(one, two, three, four)))
#
#      pred = sess.run(output, feed_dict={xs:x})
#      plot(np.arange(-360,361,15), pred, 'prediction_layer1_%d_full.png' % n_hidden1)
#    
#      # save model
#      save_path = saver.save(sess, 
#          './model_final_layer1_%d.ckpt' % n_hidden1)
