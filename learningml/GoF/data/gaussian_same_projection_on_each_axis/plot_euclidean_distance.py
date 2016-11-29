from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os

dimensions = range(1,11)

for dim in dimensions:
	file0_name = "gauss_data/gaussian_same_projection_on_each_axis_{0}D_10000_0.0_1.0_1.0_0_euclidean.txt".format(dim)
	file1_name = "gauss_data/gaussian_same_projection_on_each_axis_{0}D_10000_0.0_0.95_0.95_0_euclidean.txt".format(dim)

	name = "gaussian_same_projection_on_each_axis_{0}D_10000_0.0_0.95_0.95_0".format(dim)

	data0 = np.loadtxt(file0_name)[:,None]
	data1 = np.loadtxt(file1_name)[:,None]

	xmin = np.min( [np.min(data0[:,0]),np.min(data1[:,0])])
	xmax = np.max( [np.max(data0[:,0]),np.max(data1[:,0])])

	xbins = np.linspace(xmin, xmax, 20)


	#print("data0 : \n", data0[:10,:])

	plt.figure()
	plt.hist(data0[:,0], bins=xbins, alpha=0.5, color= 'blue', label='F0')
	plt.hist(data1[:,0], bins=xbins, alpha=0.5, color= 'red', label='F1')
	plt.title("Gauss distance from origin")
	#plt.xlim(-3,3)
	#plt.ylim(-3,3)
	#plt.plot(x1bins,10000*mlab.normpdf(x1bins, mu, sig),color='black',linewidth=2.)
	plt.xlabel('sqrt(r^2)')
	plt.ylabel('Number of events')
	plt.legend(loc='best')
	plt.savefig(name+"_1D_euclidean_proj_comp.png")
	print("plotting "+name+"_1D_euclidean_proj_comp.png")

