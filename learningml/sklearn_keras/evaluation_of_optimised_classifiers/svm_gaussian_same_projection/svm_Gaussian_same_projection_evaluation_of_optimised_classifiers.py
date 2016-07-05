import numpy as np
import math
import sys 
sys.path.insert(0,'../..')
import os
import classifier_eval
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC


for dim in range(2,11):
	comp_file_list=[]
    
	####################################################################
	# Gaussian samples operation
	####################################################################

	for i in range(100):
		comp_file_list.append((os.environ['learningml']+"/sklearn_keras/data/gaussian_same_projection_on_each_axis/gaussian_same_projection_on_each_axis_redefined_{1}D_1000_0.6_0.2_0.1_{0}.txt".format(i,dim),os.environ['learningml']+"/sklearn_keras/data/gaussian_same_projection_on_each_axis/gaussian_same_projection_on_each_axis_redefined_{1}D_1000_0.6_0.2_0.075_{0}.txt".format(i,dim)))
	
	#Earlier for SVM we had C=496.6 and gamma=0.00767

        clf = SVC(C=290.4,gamma=0.0961,probability=True, cache_size=7000)

        ####################################################################


	classifier_eval.classifier_eval(name=(str(dim)+"Dgaussian_same_projection_redefined__0_1__0_075_optimised_svm"),comp_file_list=comp_file_list,clf=clf)


