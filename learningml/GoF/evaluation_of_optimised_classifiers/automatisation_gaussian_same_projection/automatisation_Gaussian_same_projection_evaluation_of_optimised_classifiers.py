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


#nclf_list = [nclf(), name_to_nclf("bdt"), nclf('xgb',XGBoostClassifier(),['n_estimators','eta'], [[10,1000],[0.01,1.0]]) ]

nclf_list = [name_to_nclf("nn")]
#nclf_list = [nclf()]
#nclf_list = [nclf('xgb',XGBoostClassifier(),['n_estimators','eta'], [[10,1000],[0.01,1.0]],[10,0.1])]
#nclf_list = [nclf('nn',"no_classifier_needed", ['n_hidden_layers','dimof_middle'], [[0,1],[100,500]])]

file_name_patterns= [ os.environ['learningml']+"/GoF/data/gaussian_same_projection_on_each_axis/gaussian_same_projection_on_each_axis_redefined_{0}D_1000_0.6_0.2_0.1_{1}.txt", os.environ['learningml']+"/GoF/data/gaussian_same_projection_on_each_axis/gaussian_same_projection_on_each_axis_redefined_{0}D_1000_0.6_0.2_0.075_{1}.txt" ]
name_CPV= "{0}Dgaussian_same_projection_redefined__0_1__0_075_optimised"
name_noCPV= "{0}Dgaussian_same_projection_redefined__0_1__0_1_noCPV_optimised"


expt = experiment(nclf_list=nclf_list, file_name_patterns=file_name_patterns )

expt.optimise(optimisation_dimension = 4, number_of_iterations=1)


expt.evaluate(evaluation_dimensions = range(2,3))




