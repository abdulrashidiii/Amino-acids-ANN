import os,time
from sys import argv
import numpy as np
from extrema import e3d


script,inputfile = argv

def main(inputfile):
    data = np.loadtxt(inputfile,delimiter=',',unpack=True).astype('float64')
    print(data.shape)
    if data.shape[0] != 3:
      grid = np.arange(-180,181,15)
      y,x = np.meshgrid(grid, grid)
      x = x.ravel()
      y = y.ravel()
      z = data
      
    # z = 2625.49965*(z-min(z))

    extmin,extmax = getextrema(x,y,z)

    
    with open('minima.dat', 'w') as f:
      f.write('Number of minima found: ' + str(len(extmin)) + '\n')
      f.write('Conformation     phi         psi      RelEnergy\n')
      f.write("------------------------------------------------\n")
      for i in range(len(extmin)):
          f.write("%s\t\t\t:\t%8.3f\t\t,\t%8.3f\t:%10.5f\n" % (conf(extmin[i,0],extmin[i,1]),extmin[i,0],extmin[i,1],extmin[i,2]))


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

def conf(x,y):
    if -180 <= x < -120:
        if -180 <= y < -120:
            z = 'beta-L'
        elif -120 <= y < 0:
            z = 'delta-D'
        elif 0 <= y < 120:
            z = 'delta-L'
        elif 120 <= y <= 180:
            z = 'beta-L'
    elif -120 <= x < 0:
        if -180 <= y < -120:
            z = 'epsilon-L'
        elif -120 <= y < 0:
            z = 'alpha-L'
        elif 0 <= y < 120:
            z = 'gamma-L'
        elif 120 <= y <= 180:
            z = 'epsilon-L'
    elif 0 <= x < 120:
        if -180 <= y < -120:
            z = 'epsilon-D'
        elif -120 <= y < 0:
            z = 'gamma-D'
        elif 0 <= y < 120:
            z = 'alpha-D'
        elif 120 <= y <= 180:
            z = 'epsilon-D'
    elif 120 <= x <= 180:
        if -180 <= y < -120:
            z = 'beta-L'
        elif -120 <= y < 0:
            z = 'delta-D'
        elif 0 <= y < 120:
            z = 'delta-L'
        elif 120 <= y <= 180:
            z = 'beta-L'

    return z

if __name__ == "__main__":
    main(inputfile)

