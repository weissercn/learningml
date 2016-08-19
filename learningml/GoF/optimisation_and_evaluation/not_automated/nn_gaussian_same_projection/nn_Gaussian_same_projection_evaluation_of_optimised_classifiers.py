import numpy as np
import math
import sys 
import os
sys.path.insert(0,os.environ['learningml']+'/GoF/')
import classifier_eval
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from keras.wrappers.scikit_learn import KerasClassifier

for dim in range(2,11):
	comp_file_list=[]
    
	####################################################################
	# Gaussian samples operation
	####################################################################

	for i in range(100):
		comp_file_list.append((os.environ['learningml']+"/GoF/data/gaussian_same_projection_on_each_axis/gaussian_same_projection_on_each_axis_redefined_{1}D_1000_0.6_0.2_0.1_{0}.txt".format(i,dim),os.environ['learningml']+"/GoF/data/gaussian_same_projection_on_each_axis/gaussian_same_projection_on_each_axis_redefined_{1}D_1000_0.6_0.2_0.075_{0}.txt".format(i,dim)))

	#originally for nn we had 100 and 4

	clf = KerasClassifier(classifier_eval.make_keras_model,n_hidden_layers=6,dimof_middle=100,dimof_input=dim)

        ####################################################################


	classifier_eval.classifier_eval(name=str(dim)+"Dgaussian_same_projection_redefined__0_1__0_1_optimised_keras_mode_2_binary",comp_file_list=comp_file_list,clf=clf)


