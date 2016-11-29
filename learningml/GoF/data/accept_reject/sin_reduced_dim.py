from __future__ import print_function
import numpy as np
import pandas as pd
import os
import sys
#sys.path.insert(0,'GoF_input')

# choose between 'alphaSvalue', 'alphaSvalue_monash_noCPV', 'pTmin' and 'pTmin_monash_noCPV'
# alphaSvalue: 0.133, pTmin: 0.9

#For block 1: 0 is 1-Thrust, 1 is C_Param
dimensions = range(2,11)


which_column = 0

name_orig_pattern       = "sin1diff_data/data_sin1diff_5_and_5_periods{}D_10000_sample_{}.txt"
name_orig_red_pattern   = "sin1diff_1Dproj_data/data_sin1diff_5_and_5_periods{}D_10000_sample_{}_1Dproj.txt"
name_mod_pattern        = "sin1diff_data/data_sin1diff_5_and_6_periods{}D_10000_sample_{}.txt"
name_mod_red_pattern    = "sin1diff_1Dproj_data/data_sin1diff_5_and_6_periods{}D_10000_sample_{}_1Dproj.txt"

for dim in dimensions:
	#features = ["%i" % number for number in np.linspace(-1*dim,-1,dim)]
	#print("features : ",features)
	print("dim : ", dim)
        for i in range(200):
		print("file : ", name_orig_pattern.format(dim,i))
                #data_orig = pd.read_csv(name_orig_pattern.format(dim,i),sep="\t", header = None, names=features)
		data_orig = np.loadtxt(name_orig_pattern.format(dim,i),dtype='d')
		print("data_orig : ", data_orig)
                if (i%20==0): print("iteration "+str(i)+" data_orig\n",data_orig)
                data_orig_1D = data_orig[:,-1]
                if (i%20==0):print("iteration "+str(i)+" data_orig_1D\n", data_orig_1D,"\n\n")
		np.savetxt(name_orig_red_pattern.format(dim,i), data_orig_1D)
                #data_orig_1D.to_csv(name_orig_red_pattern.format(dim,i),sep="\t", header = None, index=False)

        for i in range(100):
                print("file : ", name_mod_pattern.format(dim,i))
                #data_mod = pd.read_csv(name_mod_pattern.format(dim,i),sep="\t", header = None, names=features)
                data_mod = np.loadtxt(name_mod_pattern.format(dim,i),dtype='d')
                print("data_mod : ", data_mod)
                if (i%20==0): print("iteration "+str(i)+" data_mod\n",data_mod)
                data_mod_1D = data_mod[:,-1]
                if (i%20==0):print("iteration "+str(i)+" data_mod_1D\n", data_mod_1D,"\n\n")
                np.savetxt(name_mod_red_pattern.format(dim,i), data_mod_1D)
                #data_mod_1D.to_csv(name_mod_red_pattern.format(dim,i),sep="\t", header = None, index=False)
