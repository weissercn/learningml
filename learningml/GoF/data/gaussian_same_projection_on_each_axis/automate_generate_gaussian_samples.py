#!/usr/local/bin/python

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#
# Name:     automate_generate_gaussian_samples.py 
#
# Author:   Constantin Weisser (weisser@mit.edu)
#
# Purpose:  This is a python script to automate writing file containing data points
#           sampled from a 2D Gaussian
#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 

from __future__ import print_function
import os
import math


number_of_files=1
#log_every_n = number_of_files // 10
log_every_n = 1
dim_list = [2]
#dim_list = [1,2,3,4,5,6,7,8,9,10]
optimisation_mode=0 #if it is on only one file gets generated and "optimisation" gets appended to the name
if optimisation_mode==1:
	number_of_files=1

for no_dim in dim_list:
	for i in range(number_of_files):
		if(i%log_every_n ==0):
			print("{0} files have been written so far for dimension {1}".format(i,no_dim))
		#          ./generate_gaussian_samples.py  no_samples  loc_centre  sigma parallel  sigmas perp  no_dim 	nth file	optimisation mode	rotated imperfectly   	REDEF_SIGMA_PERP 	COLOR
		os.system("./generate_gaussian_samples.py  10000        0.0         1.0            1.0  	{0}      {1} 		{2}			1			0			blue".format(no_dim,i,optimisation_mode)) 


print("{0} files have been generated".format(number_of_files*len(dim_list)))

