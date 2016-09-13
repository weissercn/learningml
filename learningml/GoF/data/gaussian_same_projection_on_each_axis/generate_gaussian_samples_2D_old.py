#!/usr/bin/env python

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#
# Name:     generate_gaussian_samples.py 
#
# Author:   Constantin Weisser (weisser@mit.edu)
#
# Purpose:  This is a python script to write a file containing 10000 data points
#	    sampled from a 2D Gaussian
#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 


#Constantin Weisser
from __future__ import print_function
from random import gauss
import sys
import numpy as np
import matplotlib.pyplot as plt

args = str(sys.argv)
#print ("Args list: %s " % args)
#The first argument is the name of this python file
total = len(sys.argv)

if(total==7):
	no_points = int(sys.argv[1])
	dist_origin_centre = float(sys.argv[2])
	sigma_parallel = float(sys.argv[3])
	sigma_perp = float(sys.argv[4])
	no_dim = int(sys.argv[5])
	label_no =float(sys.argv[6])
else:	
	print("Using standard arguments")

print("no_points : ",no_points, "dist_origin_centre : ",dist_origin_centre, "sigma_parallel : ",sigma_parallel, "sigma_perp : ", sigma_perp, "no_dim : ",no_dim, "label_no : ",label_no)
	
if no_dim!=2:
        print("Analysis in dimensions other than two has not been implemented yet")
else:   
	

	values_x = np.zeros((no_points,1))
        values_y = np.zeros((no_points,1))
	for i in range(no_points):
		values_parallel = gauss(dist_origin_centre,sigma_parallel)
		values_perp     = gauss(0,sigma_perp)
	
		values_x[i] = (values_parallel - values_perp)/2.0
		values_y[i] = (values_parallel + values_perp)/2.0
	full_cords= np.column_stack((values_x,values_y))

print(full_cords)

name = "gaussian_same_projection_on_each_axis_"+str(int(no_dim))+"D_"+str(int(no_points))+"_"+str(dist_origin_centre)+"_"+str(sigma_parallel)+"_"+str(sigma_perp)+"_"+str(int(label_no))

np.savetxt("gauss_data/"+name+".txt",full_cords)

plt.figure()
plt.hist2d(full_cords[:,0],full_cords[:,1], bins=20, range=np.array([(0,1),(0,1)]))
plt.title("Gauss Same Projection on each axis 2D Histogram")
plt.xlim(0,1)
plt.ylim(0,1)
cb= plt.colorbar()
cb.set_label("number of events")
plt.savefig(name+"_2Dhist.png")
print("plotting "+name+"_2Dhist.png")

plt.figure()
plt.hist(full_cords[:,0], bins=100, facecolor='red', alpha=0.5)
plt.title("Gauss Same Projection on each axis 1D Histogram")
plt.xlim(0,1)
plt.savefig(name+"_1Dhist.png")
print("plotting "+name+"_1Dhist.png")


