from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os

file_name_pattern =  os.environ['learningml']+"/GoF/data/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_{0}D_10000_0.0_0.95_0.95_{1}"
evaluation_dimensions = range(1,11)

for dim in evaluation_dimensions:
	for n in range(100):
		if dim ==1:	data = np.loadtxt(file_name_pattern.format(dim, n)+".txt")[:,None]
		else: 		data = np.loadtxt(file_name_pattern.format(dim, n)+".txt")

		euclidean_list= []
		for row in range(data.shape[0]):
			dist=0
			for column in range(data.shape[1]):
				dist += np.square(data[row,column])
			dist = np.sqrt(dist)
			euclidean_list.append(dist)

		data_out = np.array(euclidean_list)[:,None]
		print(file_name_pattern.format(dim, n)+".txt" +"\tdata_out : \n", data_out,"\n")
		np.savetxt(file_name_pattern.format(dim, n)+"_euclidean.txt", data_out, delimiter = " ")



