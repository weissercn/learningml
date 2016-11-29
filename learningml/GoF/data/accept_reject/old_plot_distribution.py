import numpy as np
import matplotlib.pyplot as plt

#2D double gauss scatter
features=np.loadtxt("gauss_data/data_double_high2Dgauss_10000_0.25_0.75_0.1_0.0_0.txt",dtype='d')
plt.figure()
plt.plot(features[:,0], features[:,1],'.')
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig("double_gauss_noCPV.png")

#1D double gauss histogram

plt.figure()
plt.hist(features[:,0], bins=100, facecolor='red', alpha=0.5)
plt.title("Double Gauss 1D Histogram")
plt.xlim(0,1)
plt.savefig("double_gauss_1D_hist_noCPV.png")

del features
#2D double gauss scatter
features=np.loadtxt("gauss_data/data_double_high2Dgauss_10000_0.25_0.75_0.1_0.1_0.txt",dtype='d')
plt.figure()
plt.plot(features[:,0], features[:,1],'.')
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig("double_gauss_CPV.png")

#1D double gauss histogram

plt.figure()
plt.hist(features[:,0], bins=100, facecolor='red', alpha=0.5)
plt.title("Double Gauss 1D Histogram")
plt.xlim(0,1)
plt.savefig("double_gauss_1D_hist_CPV.png")

