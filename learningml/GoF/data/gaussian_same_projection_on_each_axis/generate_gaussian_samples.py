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
from numpy import matrix
from numpy import linalg

DEBUG = False
GRAPHS = True
REDEF_SIGMA_PERP = True

args = str(sys.argv)
#print ("Args list: %s " % args)
#The first argument is the name of this python file
total = len(sys.argv)

if(total==8):
	no_points = int(sys.argv[1])
	loc_centre = float(sys.argv[2])
	sigma_parallel = float(sys.argv[3])
	sigma_perp_orig = float(sys.argv[4])
	no_dim = int(sys.argv[5])
	label_no =float(sys.argv[6])
	optimisation_mode = float(sys.argv[7])

	if REDEF_SIGMA_PERP:
                sigma_perp = np.divide(sigma_perp_orig,no_dim)
        else:
                sigma_perp = sigma_perp_orig
	
else:	
	print("Using standard arguments")
dist_origin_centre = loc_centre*np.sqrt(no_dim)

	
for n in range(1,no_dim):
	if n ==1:
		v = [list(np.divide(np.append(np.append([1.0]*(n),[-n]),[0.0]*(no_dim-n-1)),np.sqrt(np.square(n)+n)))]
	else:
		v.append(list(np.divide(np.append(np.append([1.0]*(n),[-n]),[0.0]*(no_dim-n-1)),np.sqrt(np.square(n)+n))))
	#print("v : ",v)

v.append([1.0/np.sqrt(no_dim)]*no_dim)
X_to_A = matrix (v )

for i in range(no_points):
	a = [[gauss(0,sigma_perp)]]
	for j in range(no_dim-2):
		a.append([gauss(0,sigma_perp)])
	a.append([gauss(dist_origin_centre,sigma_parallel)])

	A = matrix (a)
	#print("A : ",A)
	if i ==0:
		X = np.transpose(X_to_A.I*A)
	else:
		X = np.row_stack((X,np.transpose(X_to_A.I*A)))

X =  np.asarray(X)
if DEBUG:
	print("no_points : ",no_points, "dist_origin_centre : ",dist_origin_centre, "sigma_parallel : ",sigma_parallel, "sigma_perp : ", sigma_perp, "no_dim : ",no_dim, "label_no : ",label_no)
	print("v : ",v) 
	print("X_to_A : ",X_to_A)
	print("X : ",X)
	print("X[:,0] : ",X[:,0])
	print("X[:,1] : ",X[:,1])


name = "gaussian_same_projection_on_each_axis_"
if REDEF_SIGMA_PERP:
	name += "redefined_"
name += str(int(no_dim))+"D_"+str(int(no_points))+"_"+str(loc_centre)+"_"+str(sigma_parallel)+"_"+str(sigma_perp_orig)
if optimisation_mode==1:
	name += "_optimisation"
name += "_"+str(int(label_no))

np.savetxt("gauss_data/"+name+".txt",X)

if GRAPHS:
	plt.figure()
	plt.hist2d(X[:,0],X[:,1], bins=20, range=np.array([(0,1),(0,1)]))
	plt.title("Gauss Same Projection on each axis 2D Histogram")
	plt.xlim(0,1)
	plt.ylim(0,1)
	cb= plt.colorbar()
	cb.set_label("number of events")
	plt.savefig(name+"_2Dhist.png")
	print("plotting "+name+"_2Dhist.png")

	plt.figure()
	plt.hist(X[:,0], bins=100, facecolor='red', alpha=0.5)
	plt.title("Gauss Same Projection on each axis 1D Histogram")
	plt.xlim(0,1)
	plt.savefig(name+"_1Dhist.png")
	print("plotting "+name+"_1Dhist.png")

	print("\n\n")
