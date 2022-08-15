import numpy as np

data = np.loadtxt('./models/nn_layer1_51.log', comments='E', usecols=[2])
pred = np.loadtxt('./models/nn_layer1_51.log', comments='E', usecols=[5])

sst = np.square(data - data.mean()).sum()

ssr = np.square(data - pred).sum()

r2 = 1 - (ssr / sst)

print(r2)
