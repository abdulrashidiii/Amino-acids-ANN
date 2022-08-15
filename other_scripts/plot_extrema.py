from scipy.ndimage.filters import gaussian_filter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sys import argv

t0 = datetime.now()

dof = int(argv[1])


# contour plot function
def plot(x, data, minima_fname, fname):
  # Plot contour of original data
  plt.figure(figsize=(11,8.5))
  #plt.figure()
  
  # colorbar ticks
  cb_min = np.floor(np.min(data))
  cb_max = np.ceil(np.max(data))
  cbar_levels = (((cb_max // 10) + 1) * 10) / 5
  cbar_ticks = np.append(np.arange(0,cb_max,cbar_levels), cb_max)
  
  # data smoothing
  train_contour = gaussian_filter(data.reshape(len(x),len(x)).T, 1)
  
  # colormap used
  colormap = 'jet'
  
  # filled contour
  plt.contourf(x, 
      x, 
      train_contour, 
      cmap=colormap, 
      extent=[x[0],x[-1],x[0],x[-1]],
      levels=np.append(np.arange(0,cb_max,cbar_levels/5), cb_max))

  # colorbar
  cb = plt.colorbar(ticks = cbar_ticks)
  cb.set_label(r'$\Delta E\ (kJ\cdot mol^{-1})$', fontsize=18)
  
  # contour lines
  C = plt.contour(x, 
      x, 
      train_contour, 
      extent=[x[0],x[-1],x[0],x[-1]], 
      levels=np.append(np.arange(0,cb_max,cbar_levels/5),cb_max))
  
  #  # contour line labels
  #  plt.clabel(C, inline=1, fontsize=12)
  
  #plt.gca().invert_xaxis()

  phi_minima = np.loadtxt(minima_fname, skiprows=3, usecols=[2])
  psi_minima = np.loadtxt(minima_fname, skiprows=3, usecols=[4])

  plt.scatter(phi_minima[:1], psi_minima[:1], c='#f44747', s=40)
  plt.scatter(phi_minima[1:], psi_minima[1:], c='#ffff00', s=40)
  plt.axvline(0, linestyle=':', c='white')
  plt.axvline(-120, linestyle=':', c='white')
  plt.axvline(120, linestyle=':', c='white')
  plt.axhline(0, linestyle=':', c='white')
  plt.axhline(-120, linestyle=':', c='white')
  plt.axhline(120, linestyle=':', c='white')
  plt.annotate(r'$\beta_{L}$', xy=(-160,150), color='white', fontsize=25)
  plt.annotate(r'$\beta_{L}$', xy=(-160,-160), color='white', fontsize=25)
  plt.annotate(r'$\beta_{L}$', xy=(150,-160), color='white', fontsize=25)
  plt.annotate(r'$\beta_{L}$', xy=(150,150), color='white', fontsize=25)
  plt.annotate(r'$\delta_{L}$', xy=(-160,60), color='white', fontsize=25)
  plt.annotate(r'$\delta_{L}$', xy=(150,60), color='white', fontsize=25)
  plt.annotate(r'$\delta_{D}$', xy=(-160,-60), color='white', fontsize=25)
  plt.annotate(r'$\delta_{D}$', xy=(150,-60), color='white', fontsize=25)
  plt.annotate(r'$\epsilon_{L}$', xy=(-60,150), color='white', fontsize=25)
  plt.annotate(r'$\epsilon_{L}$', xy=(-60,-160), color='white', fontsize=25)
  plt.annotate(r'$\epsilon_{D}$', xy=(50,150), color='white', fontsize=25)
  plt.annotate(r'$\epsilon_{D}$', xy=(50,-160), color='white', fontsize=25)
  plt.annotate(r'$\gamma_{L}$', xy=(-60,60), color='white', fontsize=25)
  plt.annotate(r'$\gamma_{D}$', xy=(50,-60), color='white', fontsize=25)
  plt.annotate(r'$\alpha_{L}$', xy=(-60,-60), color='white', fontsize=25)
  plt.annotate(r'$\alpha_{D}$', xy=(50,60), color='white', fontsize=25)
  
  
  cfaxes = plt.gca()
  plt.sca(cb.ax)
  plt.clim(vmin=0, vmax=cb_max)
  plt.yticks(size='20')
  
  plt.sca(cfaxes)
  plt.xticks([-180,-120,-60,0,60,120,180], size='20')
  plt.yticks([-120,-60,0,60,120,180], size='20')

  plt.xlabel(r'$\phi$', fontsize=20)
  plt.ylabel(r'$\psi$', fontsize=20)
  plt.xlim(-180,180)
  plt.ylim(-180,180)

  plt.savefig(fname, bbox_inches='tight', transparent=True, pad_inches=0)
#  plt.show()
  plt.close()

data = np.loadtxt('../models/prediction_fine_grid_layer1_%d.csv' % dof, usecols=[2], delimiter=',')
data = data.reshape(361,361)
plot(np.arange(-180,181), data, './final_minima.dat', 'surface_with_new_minima.png')
#plot(np.arange(-180,181,15), data, './minimize/better_minima.dat', 'prediction_layer1_51_with_new_minima.png')

print('Runtime:', datetime.now() - t0)
