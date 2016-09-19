import numpy as np
import math
import sys
import os 
sys.path.insert(0,os.environ['learningml']+'/GoF/')
import classifier_eval
from classifier_eval import name_to_nclf, nclf, experiment, make_keras_model
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from rep.estimators import XGBoostClassifier
from keras.wrappers.scikit_learn import KerasClassifier

import time

#nclf_list = [nclf()]
#nclf_list = [nclf(), name_to_nclf("bdt"), nclf('xgb',XGBoostClassifier(),['n_estimators','eta'], [[10,1000],[0.01,1.0]]) ]
#nclf_list = [nclf('xgb',XGBoostClassifier(),['n_estimators','eta'], [[10,1000],[0.01,1.0]], param_opt=[1000.,0.9738])]
#nclf_list = [nclf('nn',"no classifier needed for nn", ['n_hidden_layers','dimof_middle'], [[0,1],[100,500]],param_opt=[0,500])]
#nclf_list = [name_to_nclf("nn")]

#nclf_list = [name_to_nclf("bdt"), name_to_nclf("xgb"), name_to_nclf("svm"), name_to_nclf("nn")]
#nclf_list = [name_to_nclf("bdt"), name_to_nclf("xgb"),  name_to_nclf("nn")]
#nclf_list = [name_to_nclf("svm")]

nclf_list = [nclf('bdt',AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=2)), ['learning_rate','n_estimators'], [[0.01,2.0],[1,1000]], param_opt=[0.01, 992]),  nclf('xgb',XGBoostClassifier(), ['n_estimators','eta'], [[10,1000],[0.01,1.0]], param_opt=[423, 0.0104]),  nclf('nn',"no classifier needed for nn", ['n_hidden_layers','dimof_middle'], [[0,1],[100,500]],param_opt=[1,210]), nclf('svm',SVC(probability=True, cache_size=7000), ['C','gamma'], [[1.0,1000.0],[1E-6,0.1]], param_opt=[583.3,0.0012])]
#nclf_list = [nclf('bdt',AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=2)), ['learning_rate','n_estimators'], [[0.01,2.0],[1,1000]], param_opt=[0.432, 18]),  nclf('xgb',XGBoostClassifier(), ['n_estimators','eta'], [[10,1000],[0.01,1.0]], param_opt=[619, 0.1489]),  nclf('nn',"no classifier needed for nn", ['n_hidden_layers','dimof_middle'], [[0,1],[100,500]],param_opt=[0,174]), nclf('svm',SVC(probability=True, cache_size=7000), ['C','gamma'], [[1.0,1000.0],[1E-6,0.1]], param_opt=[5.269, 0.00453])]

#nclf_list = [nclf('bdt',AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=2)), ['learning_rate','n_estimators'], [[0.01,2.0],[1,1000]], param_opt=[0.01, 992]),  nclf('xgb',XGBoostClassifier(), ['n_estimators','eta'], [[10,1000],[0.01,1.0]], param_opt=[423, 0.0104])]
#nclf_list = [nclf('nn',"no classifier needed for nn", ['n_hidden_layers','dimof_middle'], [[0,1],[100,500]],param_opt=[1,210])] 

systematics_fraction = 0.01

file_name_patterns= [ os.environ['learningml']+"/GoF/data/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_{0}D_10000_0.0_1.0_1.0_{1}.txt", os.environ['learningml']+"/GoF/data/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_{0}D_10000_0.0_0.95_0.95_{1}.txt" ]
#file_name_patterns= [ os.environ['learningml']+"/GoF/data/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_redefined_{0}D_10000_0.0_1.0_1.0_optimisation_{1}.txt", os.environ['learningml']+"/GoF/data/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_redefined_{0}D_10000_0.0_1.0_0.9_optimisation_{1}.txt" ]

name_CPV= "{0}Dgauss__0_95__0_95_CPV_not_redefined"
name_noCPV= "{0}Dgauss__1_0__1_0_noCPV_not_redefined"
#name_CPV= "{0}Dgauss__1_0__0_95_CPV_chi2scoringopt"
#name_noCPV= "{0}Dgauss__1_0__1_0_noCPV_chi2scoringopt"
title_CPV = "Gauss 0.95 0.95 "
title_noCPV="Gauss 1.0 1.0 "
directory_name = "_gauss_final_plot"

expt = experiment(nclf_list=nclf_list, file_name_patterns=file_name_patterns, scoring='chi2',single_no_bins_list = [5], systematics_fraction = systematics_fraction, only_mod=True, title_CPV=title_CPV, title_noCPV=title_noCPV, name_CPV=name_CPV, name_noCPV=name_noCPV, directory_name=directory_name)

start_time = time.time()

#expt.optimise(optimisation_dimension = 4, number_of_iterations=50)
#optimisation gave nn param_opt

evaluation_start_time = time.time()

print(50*"-"+"\noptimisation took  ", (evaluation_start_time - start_time)/60. , "  minutes\n" +50*"-")



expt.evaluate(evaluation_dimensions = [2,4,10], number_of_evaluations=1)

end_time = time.time()

print(50*"-"+"\nevaluation took  ", (end_time - evaluation_start_time)/60. , "  minutes\n" +50*"-")
