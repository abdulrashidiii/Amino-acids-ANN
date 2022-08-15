"""

Features:

- importing and preprocessing of gaussian data
- generation of contour plot
- implementation of fully connected hidden layer and regression output layer
- training of implemented neural network model
- report of RMSE and R^2 of trained neural network model
- saving of trained neural network model
- generation of grid points from a trained neural network model
- finding minima points from a fine grid
- perform optimization on the trained model starting from the minima points

ANN Model
- Sigmoid activation function on hidden layer
- Random normal distribution in weight initialization
- Zero bias initialization
- No activation function on ouput node
- MSE cost function
- SGD Adam Optimizer
- To keep response variable nonnegative, values are shifted according to minimum value
- Minimum value becomes 0

"""

from datetime import datetime
t0 = datetime.now()
print('Start time:', t0)

from sklearn.model_selection import train_test_split
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import fmin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from extrema import e3d

################ PATHS ##################

figures_path = 'figures/'
models_path = 'models/'
minima_path = 'minima/'

#########################################

################################# FIND EXTREMA POINTS ###################################

def getextrema(x,y,z):
  n = int(np.sqrt(len(z)))
  G = z.reshape(n,n)
  # add gridpoints at the edge
  zmax, imax, zmin, imin = e3d(G)

  zmax = np.array(zmax).astype('float64').ravel().tolist()
  zmin = np.array(zmin).astype('float64').ravel().tolist()
  imax = np.array(imax).astype('int').ravel().tolist()
  imin = np.array(imin).astype('int').ravel().tolist()

  extmin = []
  for i in range(len(imin)):
    if ((-180.0 <= x[imin[i]] <= 180.0) and (-180.0 <= y[imin[i]] <= 180.0)):
      extmin.append([x[imin[i]], y[imin[i]],zmin[i]])

  extmin=np.array(extmin)

  extmax = []
  for i in range(len(imax)):
    if ((-180.0 <= x[imax[i]] <= 180.0) and (-180.0 <= y[imax[i]] <= 180.0)):
      extmax.append([x[imax[i]], y[imax[i]],zmax[i]])

  extmax = np.array(extmax)

  return extmin,extmax

#########################################################################################

######### CONFORMATIONS ##########

def conf(x,y):
  if -180 <= x < -120:
    if -180 <= y < -120:
      z = 'beta-L   '
    elif -120 <= y < 0:
      z = 'delta-D  '
    elif 0 <= y < 120:
      z = 'delta-L  '
    elif 120 <= y <= 180:
      z = 'beta-L   '
  elif -120 <= x < 0:
    if -180 <= y < -120:
      z = 'epsilon-L'
    elif -120 <= y < 0:
      z = 'alpha-L  '
    elif 0 <= y < 120:
      z = 'gamma-L  '
    elif 120 <= y <= 180:
      z = 'epsilon-L'
  elif 0 <= x < 120:
    if -180 <= y < -120:
      z = 'epsilon-D'
    elif -120 <= y < 0:
      z = 'gamma-D  '
    elif 0 <= y < 120:
      z = 'alpha-D  '
    elif 120 <= y <= 180:
      z = 'epsilon-D'
  elif 120 <= x <= 180:
    if -180 <= y < -120:
      z = 'beta-L   '
    elif -120 <= y < 0:
      z = 'delta-D  '
    elif 0 <= y < 120:
      z = 'delta-L  '
    elif 120 <= y <= 180:
      z = 'beta-L   '

  return z

##################################

######### FIX FLOAT ERRORS ##############

def FixZeros(values):
  for i in range(len(values)):
    if np.allclose(values[i], 0):
      values[i] = 0

#########################################

############ R-SQUARED #####################

def r_squared(data, prediction):
  sst = np.square(data - data.mean()).sum()
  ssr = np.square(data - prediction).sum()

  return 1 - (ssr / sst)

############################################

#################### IMPORT PREPROCESSED GAUSSIAN DATA #################################

print('Reading Data...')
data = pd.read_csv('../data_preprocessed.csv')
X = data.drop(['Energy', 'Group'], axis=1).values.T
Y = data['Energy'].values.ravel()

########################## CONTOUR PLOT FUNCTION ####################################

def plot(x, data, color, fname):                                                    
  plt.figure(figsize=(11,8.5))                                                      
  #plt.figure()                                                                     

  # colorbar ticks
  cb_min = np.floor(np.min(data))
  cb_max = np.ceil(np.max(data))
  cbar_levels = (((cb_max // 10) + 1) * 10) / 5
  cbar_ticks = np.arange(0,cb_max,cbar_levels)

  # data smoothing
  train_contour = gaussian_filter(data.reshape(len(x),len(x)).T, 1)

  # colormap used
  colormap = color

  # filled contour
  plt.contourf(x,
      x,
      train_contour,
      cmap=colormap,
      extent=[x[0],x[-1],x[0],x[-1]],
      levels=np.arange(cb_min,cb_max+0.5,cbar_levels/5))

  # colorbar
  cb = plt.colorbar(ticks = cbar_ticks)
  #cb.set_label(r'$$', fontsize=18)
  
  # contour lines
  C = plt.contour(x, 
      x, 
      train_contour, 
      extent=[x[0],x[-1],x[0],x[-1]], 
      levels=np.arange(cb_min,cb_max+0.5,cbar_levels/5))
  
  #  # contour line labels
  #  plt.clabel(C, inline=1, fontsize=12)
  
  cfaxes = plt.gca()
  plt.sca(cb.ax)
  plt.clim(vmin=0, vmax=cb_max)
  plt.yticks(size='18')
  
  plt.sca(cfaxes)
  plt.xticks(size='22')
  plt.yticks(size='22')

  plt.xlabel(r'$\phi$', fontsize=30)
  plt.ylabel(r'$\psi$', fontsize=30)

  plt.savefig(fname, bbox_inches='tight', transparent=True, pad_inches=0)
  plt.close()

##########################################################################

################# FULLY CONNECTED HIDDEN LAYER ###########################

def fc_layer(input_data, input_dim, output_dim, name='fc'):
  """
  weight initialization from normal distribution
  sigmoid activation function
  """
  with tf.name_scope(name):
    w = tf.Variable(tf.random.normal(shape=[output_dim,input_dim],
      mean=0.0,
      stddev=1.0/np.sqrt(input_dim)),
      name='W')
    b = tf.Variable(tf.zeros([output_dim,1]),
        name='b')
    layer = tf.add(tf.matmul(w, input_data), b)
    act = tf.nn.sigmoid(layer)
    tf.compat.v1.summary.histogram('weights', w)
    tf.compat.v1.summary.histogram('biases', b)
    tf.compat.v1.summary.histogram('activations', act)
    return act

#############################################################################

####################### REGRESSION OUTPUT LAYER #############################

def output_layer(input_data, input_dim, output_dim, name='output'):
  with tf.name_scope(name):
    w = tf.Variable(tf.random.normal(shape=[output_dim,input_dim],
      mean=1.0,
      stddev=1.0/np.sqrt(input_dim)),
      name='W')
    b = tf.Variable(tf.zeros([output_dim, 1]),
        name='b')
    output = tf.add(tf.matmul(w, input_data), b)

    tf.compat.v1.summary.histogram('weights', w)
    tf.compat.v1.summary.histogram('biases', b)
    tf.compat.v1.summary.histogram('outputs', output)

    return output

##############################################################################

################### GENERATE GRID FROM TRAINED MODEL #########################

def generate_grid(x, fname):
  # full ramachandran contour plot -360,360
  x = np.meshgrid(x,x)
  one = np.sin(np.radians(x[1].reshape(-1)))
  FixZeros(one)
  two = np.sin(np.radians(x[0].reshape(-1)))
  FixZeros(two)
  three = np.cos(np.radians(x[1].reshape(-1)))
  FixZeros(three)
  four = np.cos(np.radians(x[0].reshape(-1)))
  FixZeros(four)
  input_vec = np.array(list(zip(one, two, three, four))).T

  pred = sess.run(output, feed_dict={xs:input_vec}).ravel()
  pred = pred - pred.min()

  grid = np.array(list(zip(x[1].reshape(-1), x[0].reshape(-1), pred)))
  np.savetxt(fname, grid, delimiter=',')

  return pred

###############################################################################

################## GAUSSIAN DATA CONTOUR PLOT #################################

#plot(np.arange(-180,181,15), y_test, 'coolwarm', figures_path + 'original.png')

###############################################################################

######################### TRAIN NEURAL NETWORK ################################

for first in [60]:
  with open(models_path + 'nn_layer1_%d.log' % first, 'w') as f:
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
  
    # control parameters
    n_inputs = 4
    n_hidden1 = first
    n_outputs = 1
    n_epochs = 2000
  
    # compat.v1.placeholders
    xs = tf.compat.v1.placeholder('float32', name='X')
    ys = tf.compat.v1.placeholder('float32', name='y')
  
    # neural network model
    fc1 = fc_layer(xs, n_inputs, n_hidden1, name='FC1')
    output = output_layer(fc1, n_hidden1, n_outputs, name='Output')
    
    # MSE cost function
    with tf.name_scope('Cost'):
      cost = tf.reduce_mean(tf.square(output-ys))
  
    tf.compat.v1.summary.scalar('Cost', cost)
    
    # SGD Adam optimizer
    with tf.name_scope('Train'):
      train = tf.compat.v1.train.AdamOptimizer(0.001).minimize(cost)
    
    c_train = []
    
    # create a node to initialize all variables
    init = tf.compat.v1.global_variables_initializer()
    
    # save our trained model
    saver = tf.compat.v1.train.Saver()
    
    # Execution Phase
    print('Training...')
    with tf.compat.v1.Session() as sess:
      init.run()
  
      merged_summary = tf.compat.v1.summary.merge_all()
      writer = tf.compat.v1.summary.FileWriter('./tensorboard/%d' % n_hidden1)
      writer.add_graph(sess.graph)
  
      # training of model
      for i in range(1, n_epochs+1):
        for j in range(len(Y)):
          sess.run([cost,train], 
              feed_dict={xs:X[:,j].reshape(n_inputs,1), ys:Y[j]})
  
        if i % 10 == 0:
          s = sess.run(merged_summary, feed_dict={xs:X, ys:Y})
          writer.add_summary(s,i)
    
          pred = sess.run(output, feed_dict={xs:X})
    
          c_train.append(sess.run(cost, feed_dict={xs:X, ys:Y}))
          f.write('Epoch : %d Training Cost : %f\n' % (i, np.sqrt(c_train[-1])))
  
      # generation of predictions
      pred = sess.run(output, feed_dict={xs:X}).ravel()
  
      ## shift values for minimum to become zero
      pred = pred - pred.min()
  
      # FINAL RMSE VALUE
      print('Calculating overall RMSE...')
      train_rmse = np.sqrt(c_train[-1])
      rmse = np.sqrt(np.mean(np.square(pred - Y)))
      f.write('Training RMSE: %10.5f\n' % train_rmse)
      f.write('Overall RMSE: %10.5f\n' % rmse)
  
      # FINAL R-SQUARED VALUE
      print('Calculating overall R^2...')
      train_r2 = r_squared(Y, sess.run(output, feed_dict={xs:X}).ravel())
      r2 = r_squared(Y, pred)
      f.write('Training R^2: %8.3f\n' % train_r2)
      f.write('Overall R^2: %8.3f\n' % r2)
  
      # save predictions to data file
      np.savetxt(models_path + 'prediction_coarse_grid_layer1_%d.csv' % (n_hidden1), pred, delimiter=',')
    
      # Figures
  
      # predicted contour plot
      plot(np.arange(-180,181,15), pred, 'jet', figures_path + 'prediction_layer1_%d.png' % (n_hidden1))
  
      # delta contour plot
      plot(np.arange(-180,181,15), np.absolute(Y.ravel() - pred), 'jet', figures_path + 'delta_layer1_%d.png' % (n_hidden1))
  
      # cost vs epoch line plot
      c_train = np.array(c_train)
      c_train = np.sqrt(c_train)
      plt.plot(np.arange(len(c_train)) * 10, c_train, c='k', linestyle='--', label='Training')
      plt.xlabel('Epoch')
      plt.ylabel('Cost')
      plt.legend()
  
      a = plt.axes([.3,.4,.4,.3])
      plt.plot(np.arange(len(c_train)) * 10, c_train, c='k', linestyle='--')
      plt.ylim(np.floor(c_train.min()),np.ceil(c_train.min())+10)
      plt.savefig(figures_path + 'cost_layer1_%d.png' % (n_hidden1))
      plt.close()
  
      # generate finer grid points
      pred = generate_grid(np.arange(-180,181,1), models_path + 'prediction_fine_grid_layer1_%d.csv' % (n_hidden1))
  
      # find minima in finer grid
      x,y,z = np.loadtxt(models_path + 'prediction_fine_grid_layer1_%d.csv' % (n_hidden1), delimiter=',', unpack=True).astype('float64')
      extmin,extmax = getextrema(x,y,z)
  
      # write minima to file
      print('Writing minima to file...')
      with open(minima_path + 'minima_layer1_%d.dat' % (n_hidden1), 'w') as f2:
        f2.write('Number of minima found: ' + str(len(extmin)) + '\n')
        f2.write('Conformation     phi         psi      RelEnergy\n')
        f2.write('-' * 48 + '\n')
        for i in range(len(extmin)):
          f2.write("%s : %8.3f , %8.3f :%10.5f\n" % (conf(extmin[i,0],extmin[i,1]),extmin[i,0],extmin[i,1],extmin[i,2]))
  
  
      ## predicted contour plot with finer grid points
      #plot(np.arange(-360,361,15), pred, 'jet', figures_path + 'prediction_layer1_%d_full.png' % n_hidden1)
    
      # save model
      print('Saving trained model...')
      save_path = saver.save(sess, 
          models_path + 'model_final_layer1_%d.ckpt' % (n_hidden1))

#############################################################################################################

############### PART 2 ###############################

######## CONVERT ANGLES TO RANGE (-180,180) ##########

def convert(angle):
  if angle < -180:
    angle = angle + 360
  elif angle > 180:
    angle = angle - 360
  return angle

vconvert = np.vectorize(convert)

for first in [60]:
  #for first in [51]:
  ######################################################
  
  ####### CONTROL PARAMETERS OF NEURAL NETWORK #########
  
  tf.compat.v1.reset_default_graph()
  n_inputs = 4
  n_hidden1 = first
  n_outputs = 1
  
  ######################################################
  
  ################## INITIAL TENSORFLOW NODES ##########
  
  xs = tf.compat.v1.placeholder('float32', name='X')
  ys = tf.compat.v1.placeholder('float32', name='y')
  
  ######################################################
  
  ######### ONE HIDDEN LAYER ###########################
  
  fc1 = fc_layer(xs, n_inputs, n_hidden1, name='FC1')
  output = output_layer(fc1, n_hidden1, n_outputs, name='Output')
  
  ######################################################
  
  ########### SAVER CLASS ##############################
  
  print('Retrieving trained model...')
  saver = tf.compat.v1.train.Saver()
  
  ######################################################
  
  ######## FUNCTION FOR TRAINED NEURAL NETWORK #########
  
  def trained_model(angles):
    phi, psi = angles
    del angles
  
    with tf.compat.v1.Session() as sess:
      saver.restore(sess, models_path + 'model_final_layer1_%d.ckpt' % (n_hidden1))
  
      phi_sin = np.sin(np.radians(phi))
      psi_sin = np.sin(np.radians(psi))
      phi_cos = np.cos(np.radians(phi))
      psi_cos = np.cos(np.radians(psi))
      x = np.array([phi_sin, psi_sin, phi_cos, psi_cos])
      for i in range(len(x)):
        if np.allclose(x[i],0):
          x[i] = 0
      x = x.reshape(-1,1)
  
      pred = sess.run(output, feed_dict={xs:x}).ravel()
  
    return pred[0]
  
  ######################################################
  
  ##################### IMPORT DATA ON MINIMA FROM FINE GRID #########################
  
  guess_minima = np.loadtxt(minima_path + 'minima_layer1_%d.dat' % (n_hidden1), skiprows=3, usecols=[2,4])
  guess_energy = np.loadtxt(minima_path + 'minima_layer1_%d.dat' % (n_hidden1), skiprows=3, usecols=[6])
  
  ####################################################################################
  
  print('Performing optimization on the trained model...')
  ############## DO OPTIMIZATION AT EACH MINIMA AND WRITE TO OUTPUT FILE #######################
  with open(minima_path + 'new_minima_layer1_%d.dat' % (n_hidden1), 'w') as f:
    with open(minima_path + 'better_minima_layer1_%d.dat' % (n_hidden1), 'w') as g:
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
        g.write('%s : %8.3f , %8.3f :%10.5f\n' % (conf(k[0], k[1]), k[0], k[1], l))
  
  ##############################################################################################

print('Runtime:', datetime.now() - t0)
