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

#nclf_list = [nclf(), name_to_nclf("bdt"), nclf('xgb',XGBoostClassifier(),['n_estimators','eta'], [[10,1000],[0.01,1.0]]) ]
#nclf_list = [nclf('xgb',XGBoostClassifier(),['n_estimators','eta'], [[10,1000],[0.01,1.0]], param_opt=[100,0.1])]
#nclf_list = [nclf('nn',"no classifier needed for nn", ['n_hidden_layers','dimof_middle'], [[0,1],[100,500]],param_opt=[0,500])]
#nclf_list = [name_to_nclf("nn")]

#nclf_list = [name_to_nclf("bdt"), name_to_nclf("xgb"), name_to_nclf("svm"), name_to_nclf("nn")]
#nclf_list = [name_to_nclf("bdt"), name_to_nclf("xgb"), name_to_nclf("nn")]
#nclf_list = [name_to_nclf("svm")]

nclf_list = [nclf('bdt',AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=2)), ['learning_rate','n_estimators'], [[0.01,2.0],[1,1000]], param_opt=[0.01, 1.0]),  nclf('xgb',XGBoostClassifier(), ['n_estimators','eta'], [[10,1000],[0.01,1.0]], param_opt=[999, 0.01]),  nclf('nn',"no classifier needed for nn", ['n_hidden_layers','dimof_middle'], [[0,1],[100,500]],param_opt=[1, 302]),nclf('svm',SVC(probability=True, cache_size=7000), ['C','gamma'], [[1.0,1000.0],[1E-6,0.1]], [500.5, 0.0500005 ] )]
#nclf_list = [nclf('bdt',AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=2)), ['learning_rate','n_estimators'], [[0.01,2.0],[1,1000]], param_opt=[0.01, 1.0])]

#nclf_list = [nclf()]


systematics_fraction = 0.01

file_name_patterns= [ os.environ['learningml']+"/GoF/data/accept_reject/sin1diff_data/data_sin1diff_5_and_5_periods{0}D_10000_sample_{1}.txt", os.environ['learningml']+"/GoF/data/accept_reject/sin1diff_data/data_sin1diff_5_and_6_periods{0}D_10000_sample_{1}.txt"]

name_CPV= "{0}Dsin_5_6_10000_CPV"
name_noCPV= "{0}Dsin_5_5_10000_noCPV"
title_CPV = "Sin 5 6 periods"
title_noCPV="Sin 5 5 periods" 
directory_name = "10000_2files"

expt = experiment(nclf_list=nclf_list, file_name_patterns=file_name_patterns, scoring='chi2',single_no_bins_list = [5], systematics_fraction = systematics_fraction, only_mod=False,title_CPV=title_CPV, title_noCPV=title_noCPV, name_CPV=name_CPV, name_noCPV=name_noCPV, directory_name=directory_name)

start_time = time.time()

#expt.optimise(optimisation_dimension = 4, number_of_iterations=50)
#optimisation gave nn param_opt

evaluation_start_time = time.time()

print(50*"-"+"\noptimisation took  ", (evaluation_start_time - start_time)/60. , "  minutes\n" +50*"-")



#expt.evaluate(evaluation_dimensions = range(1,2), number_of_evaluations=100)


end_time = time.time()

print(50*"-"+"\nevaluation took  ", (end_time - evaluation_start_time)/60. , "  minutes\n" +50*"-")

expt.plot(plot_dimensions=[2,6,10],most_typical=[47, 72, 34])

