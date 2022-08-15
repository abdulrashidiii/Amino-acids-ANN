import numpy as np

r2 = np.loadtxt('../models/nn_layer1_51.log', comments=['#', 'Epoch', 'Training', 'Testing RMSE', 'Overall'], usecols=[2])
print('R^2:', r2.mean(), '+/-', r2.std())

rmse = np.loadtxt('../models/nn_layer1_51.log', comments=['#', 'Epoch', 'Training', 'Testing R^2', 'Overall'], usecols=[2])
print('RMSE:', rmse.mean(), '+/-', rmse.std())
