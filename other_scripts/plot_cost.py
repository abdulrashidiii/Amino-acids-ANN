import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('presentation')

plt.rc('axes', linewidth=2)

#plt.figure(figsize=(11,8.5))
#
#cost_min = []
#
#cost = []
#for first in range(10,200,20):
#  with open('./nn_layer1_%d.log' % first, 'r') as f:
#    lines = f.readlines()
#    cost.append(float(lines[3999][28:-1]))
#
#cost = np.array(cost, dtype='float')
#
#plt.plot(range(10,200,20), cost, c='k')
#
#plt.grid()
#plt.xlabel('Number of Neurons')
#plt.xticks(list(range(10,200,20)))
#plt.ylim(0,1)
#plt.ylabel('Cost (MSE)')
#plt.title('Cost after 2000 epochs')
#plt.savefig('cost_all2.png', bbox_inches='tight')
#plt.close()
#
#for first in range(10,200,20):
#  cost = np.loadtxt('./nn_layer1_%d.log' % first, usecols=[6], comments='O')[::2]
#  plt.figure()
#  plt.plot(range(2000), cost, c='k')
#  plt.xlabel('Epoch')
#  plt.ylabel('Cost')
#  plt.ylim(0,10)
#  plt.savefig('./cost_layer1_%d_better.png' % first, bbox_inches='tight')
#  plt.close()
#
#for first in range(10,200,20):
#  cost = np.loadtxt('./nn_layer1_%d.log' % first, usecols=[2,5], comments='E')
#  plt.figure(figsize=(11,8.5))
#  plt.scatter(np.arange(625), cost.T[0], s=1, c='k', marker='.', label='Gaussian')
#  plt.plot(cost.T[0], linestyle='-', c='k', linewidth=0.5)
#  plt.scatter(np.arange(625), cost.T[1], s=1, c='k', marker='^', label='ANN')
#  plt.plot(cost.T[1], linestyle='--', c='k', linewidth=0.5)
##  plt.xlabel('')
#  plt.xticks(np.arange(0,626,25))
#  plt.ylabel(r'Energy $\frac{kJ}{mol}$')
#  plt.ylim(0,100)
#  plt.legend()
##  plt.grid()
#  plt.savefig('./lineplot_layer%d.png' % first, bbox_inches='tight', dpi=1000)
#  plt.close()
#  
#  plt.figure(figsize=(11,8.5))
#  plt.plot(cost.T[0], linestyle='-', c='k', linewidth=0.5, label='Gaussian')
#  plt.plot(cost.T[1], linestyle='--', c='k', linewidth=0.5, label='ANN')
##  plt.xlabel('')
#  plt.xticks(np.arange(0,626,25))
#  plt.ylabel(r'Energy $\frac{kJ}{mol}$')
#  plt.ylim(0,100)
#  plt.legend()
##  plt.grid()
#  plt.savefig('./lineplot_layer%d_nopoints.png' % first, bbox_inches='tight', dpi=1000)
#  plt.close()

def arrowed_spines(fig, ax):

  xmin, xmax = ax.get_xlim() 
  ymin, ymax = ax.get_ylim()

  ## removing the default axis on all sides:
  #for side in ['bottom','right','top','left']:
  #  ax.spines[side].set_visible(False)

  ## removing the axis ticks
  #plt.xticks([]) # labels 
  #plt.yticks([])
  #ax.xaxis.set_ticks_position('none') # tick markers
  #ax.yaxis.set_ticks_position('none')

  # get width and height of axes object to compute 
  # matching arrowhead length and width
  dps = fig.dpi_scale_trans.inverted()
  bbox = ax.get_window_extent().transformed(dps)
  width, height = bbox.width, bbox.height

  # manual arrowhead width and length
  hw = 1./20.*(ymax-ymin) 
  hl = 1./20.*(xmax-xmin)
  lw = 1. # axis line width
  ohg = 0.3 # arrow overhang

  # compute matching arrowhead length and width
  yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
  yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

  # draw x and y axis
  ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw, 
           head_width=hw, head_length=hl, overhang = ohg, 
           length_includes_head= True, clip_on = False) 

  ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw, 
           head_width=yhw, head_length=yhl, overhang = ohg, 
           length_includes_head= True, clip_on = False)

                                            

plt.figure(figsize=(11,8.5))
#plt.axes(facecolor='#f2f2f2')
for first, marker in zip([30,40,51,60,70],['.','^','*','x','D']):
  cost = np.loadtxt('./nn_layer1_%d.log' % first,
      usecols=[6],
      comments='O')[::2]
  plt.plot(range(0,2000), cost, linewidth=1.1)
  plt.scatter(range(0,2000,40), cost[::40], marker=marker, s=50, label='%d neurons' % first, alpha=0.9)

plt.xticks(np.arange(0,2001,500))
#plt.yticks(np.arange(0,21,5))
#plt.ylim(-2)
plt.gca().set_yscale('log')
#plt.gca().set_xscale('log')
#plt.gca().spines['top'].set_color('none')
#plt.gca().spines['right'].set_color('none')
#fig = plt.gcf()
#ax = plt.gca()
#arrowed_spines(fig,ax)
plt.xlabel('Number of Epochs', fontsize=25)
plt.ylabel('Cost (MSE)', fontsize=25)
plt.xticks(size='18')
plt.yticks(size='18')
plt.ylim(0.4,600)
plt.legend(frameon=False, markerscale=1.75, fontsize=20)


#a = plt.axes([0.27,0.52,0.4,0.34], facecolor='#edeee3')
#for first, marker in zip([30,40,51,60,70],['.','^','*','x','D']):
#  cost = np.loadtxt('./nn_layer1_%d.log' % first,
#      usecols=[6],
#      comments='O')[::2]
#  plt.plot(range(0,2000), cost, linewidth=1.1)
#  plt.scatter(range(0,2000,40), cost[::40], marker=marker, s=25, alpha=0.9)
#
#plt.xlim(400)
#plt.xticks(list(range(500,2001,500)))
#plt.ylim(0.6,1.5)
#plt.yticks(list(np.arange(0.8,1.5,0.2)))


plt.savefig('cost_epoch_all.png', bbox_inches='tight', transparent=True, dpi=300, pad_inches=0)
plt.show()
plt.close()
