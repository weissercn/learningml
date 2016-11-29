import numpy as np
import sys
import os 
sys.path.insert(0,os.environ['learningml']+'/GoF/')
import setup_optimise_hyperparam 
from setup_optimise_hyperparam import name_to_nclf, nclf, experiment, make_keras_model
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from rep.estimators import XGBoostClassifier
from keras.wrappers.scikit_learn import KerasClassifier



#nclf_list = [nclf('bdt',AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=2)), ['learning_rate','n_estimators'], [[0.01,2.0],[1,1000]], param_opt=[0.01, 992]),  nclf('xgb',XGBoostClassifier(), ['n_estimators','eta'], [[10,1000],[0.01,1.0]], param_opt=[423, 0.0104]),  nclf('nn',"no classifier needed for nn", ['n_hidden_layers','dimof_middle'], [[0,1],[100,500]],param_opt=[1,210]), nclf('svm',SVC(probability=True, cache_size=7000), ['C','gamma'], [[1.0,1000.0],[1E-6,0.1]], param_opt=[583.3,0.0012])]

nclf_list = [name_to_nclf("bdt"), name_to_nclf("xgb"), name_to_nclf("svm"), name_to_nclf("nn")]
name_CPV= "{0}D_gauss"

file_name_patterns= [ os.environ['learningml']+"/optimise_hyperparam/gaussian_same_projection_on_each_axis_{0}D_10000_0.0_1.0_1.0_{1}.txt", os.environ['learningml']+"/optimise_hyperparam/gaussian_same_projection_on_each_axis_{0}D_10000_0.0_0.95_0.95_{1}.txt" ]

directory_name = "_gauss_final_2files_check"

expt = experiment(nclf_list=nclf_list, file_name_patterns=file_name_patterns, scoring='chi2',single_no_bins_list = [5], name_CPV=name_CPV, directory_name=directory_name)


expt.optimise(optimisation_dimension = 4, number_of_iterations=50)




