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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from numpy import matrix
from numpy import linalg

label_size = 28



################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
mpl.rc('font', family='serif', size=34, serif="Times New Roman")

#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [r'\boldmath']

mpl.rcParams['legend.fontsize'] = "medium"

mpl.rc('savefig', format ="pdf", bbox='tight', pad_inches= 0.1)

mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['figure.figsize']  = 8, 6

################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

colors_red = [(1, 1, 1), (1, 0, 0), (0, 0, 0)]
colors_blue= [(1, 1, 1), (0, 0, 1), (0, 0, 0)]

cm_red = LinearSegmentedColormap.from_list("GoF_red", colors_red, N=20)
cm_blue= LinearSegmentedColormap.from_list("GoF_blue", colors_blue, N=20)

DEBUG = False
GRAPHS = True
#REDEF_SIGMA_PERP = True

args = str(sys.argv)
#print ("Args list: %s " % args)
#The first argument is the name of this python file
total = len(sys.argv)

if(total==11):
	no_points 		= int(sys.argv[1])
	loc_centre 		= float(sys.argv[2])
	sigma_parallel 		= float(sys.argv[3])
	sigma_perp_orig 	= float(sys.argv[4])
	no_dim 			= int(sys.argv[5])
	label_no 		= float(sys.argv[6])
	optimisation_mode 	= float(sys.argv[7])
	rotated_imperfectly 	= int(sys.argv[8])
	REDEF_SIGMA_PERP    	= int(sys.argv[9])
	COLOR			= str(sys.argv[10])
	print("COLOR : ", COLOR) 

	if COLOR == "red": 
		cm = cm_red
		print("red chosen")
	else: 
		cm = cm_blue
		print("blue chosen")

	GRAPHS= not label_no

	if GRAPHS:
		print("rotated_imperfectly : ",rotated_imperfectly)
		print("REDEF_SIGMA_PERP : ",REDEF_SIGMA_PERP)
	if REDEF_SIGMA_PERP:
                sigma_perp = np.divide(sigma_perp_orig,no_dim)
        else:
                sigma_perp = sigma_perp_orig
	
else:	
	print("Using standard arguments")
dist_origin_centre = loc_centre*np.sqrt(no_dim)

v = []
	
for n in range(1,no_dim):
	v.append(list(np.divide(np.append(np.append([1.0]*(n),[-n]),[0.0]*(no_dim-n-1)),np.sqrt(np.square(n)+n))))
	#print("v : ",v)

v.append([1.0/np.sqrt(no_dim)]*no_dim)
X_to_A = matrix (v )

if not rotated_imperfectly:
	X_to_A = matrix(np.identity(no_dim))

for i in range(no_points):
	a = []
	for j in range(no_dim-1):
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
if not rotated_imperfectly: name += "rotated_perfectly_"
name += str(int(no_dim))+"D_"+str(int(no_points))+"_"+str(loc_centre)+"_"+str(sigma_parallel)+"_"+str(sigma_perp_orig)
if optimisation_mode==1:
	name += "_optimisation"
name += "_"+str(int(label_no))

np.savetxt("gauss_data/"+name+".txt",X)

if GRAPHS:
	if (no_dim>1 and False):
		plt.figure()
		plt.hist2d(X[:,0],X[:,1], bins=20, range=np.array([(-4,4),(-4,4)]))
		#plt.title("Gauss Same Projection on each axis 2D Histogram")
		plt.xlim(-4,4)
		plt.ylim(-4,4)
		plt.xlabel(r"$\mathit{x}_1$")
		plt.ylabel(r"$\mathit{x}_2$")
		#cb= plt.colorbar()
		#cb.set_label("number of events")
		plt.savefig(name+"_2Dhist_old.pdf")
		plt.close()
		print("plotting "+name+"_2Dhist_old.pdf")

	if no_dim>1:
		xedges = np.linspace(-4.,4.,30)
		yedges = np.linspace(-4.,4.,30)
		H, xedges, yedges = np.histogram2d(X[:,1], X[:,0], bins=(xedges, yedges))
		plt.figure()
		plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cm)
		plt.xlabel(r"$\mathit{x}_1$")
                plt.ylabel(r"$\mathit{x}_2$")
		plt.savefig(name+"_2Dhist.pdf")
		plt.close()
		print("plotting "+name+"_2Dhist.pdf")

	plt.figure()
	plt.hist(X[:,0], bins=100, facecolor='red', alpha=0.5)
	#plt.title("Gauss Same Projection on each axis 1D Histogram")
	plt.xlim(-4,4)
	plt.savefig(name+"_1Dhist.pdf")
	print("plotting "+name+"_1Dhist.pdf")

	print("\n\n")
