import numpy as np
import matplotlib.pyplot as plt

nodes = np.array([10,30,51,70,90])
ave_rmse = np.empty((0,0))
ave_r2 = np.empty((0,0))

for i in nodes:
  rmse = np.loadtxt(
      '../models/nn_layer1_%d.log' % i, 
      comments=['#', 'Epoch', 'Training', 'Overall', 'Testing R^2'], 
      usecols=[2])

  r2 = np.loadtxt(
      '../models/nn_layer1_%d.log' % i, 
      comments=['#', 'Epoch', 'Training', 'Overall', 'Testing RMSE'], 
      usecols=[2])

  ave_rmse = np.append(ave_rmse, rmse.mean())
  ave_r2 = np.append(ave_r2, r2.mean())

plt.figure()
plt.scatter(nodes, ave_rmse, c='k', s=20)
plt.xticks(nodes)
plt.yticks(np.arange(0,np.ceil(ave_rmse.max())+1))
plt.grid()
plt.xlabel('Number of Nodes', fontsize=15)
plt.ylabel('RMSE', fontsize=15)
plt.savefig('../figures/nodes_vs_rmse.png', transparent=True)

plt.figure()
plt.scatter(nodes, ave_r2, c='k', s=20)
plt.xticks(nodes)
plt.ylim(0.9,1.0)
plt.yticks(np.arange(0.9,np.ceil(ave_r2.max())+0.01,0.01))
plt.grid()
plt.xlabel('Number of Nodes', fontsize=15)
plt.ylabel(r'$R^{2}$', fontsize=15)
plt.savefig('../figures/nodes_vs_r2.png', transparent=True)
