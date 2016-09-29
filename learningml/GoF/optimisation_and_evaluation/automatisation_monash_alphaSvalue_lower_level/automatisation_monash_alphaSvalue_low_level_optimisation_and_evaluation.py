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

#nclf_list = [name_to_nclf("nn")]
#nclf_list = [nclf('nn',"no classifier needed for nn", ['n_hidden_layers','dimof_middle'], [[0,1],[100,500]],param_opt=[0,500])]
#nclf_list = [name_to_nclf("bdt"), name_to_nclf("xgb"), name_to_nclf("svm"), name_to_nclf("nn")]

nclf_list = [nclf('bdt',AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=2)), ['learning_rate','n_estimators'], [[0.01,2.0],[1,1000]], param_opt=[0.0190, 837]),  nclf('xgb',XGBoostClassifier(), ['n_estimators','eta'], [[10,1000],[0.01,1.0]], param_opt=[15, 0.59]),  nclf('nn',"no classifier needed for nn", ['n_hidden_layers','dimof_middle'], [[0,1],[100,500]],param_opt=[0, 309]),nclf('svm',SVC(probability=True, cache_size=7000), ['C','gamma'], [[1.0,1000.0],[1E-6,0.1]], [331.4,1.445E-5 ] )]

#nclf_list = [nclf('bdt',AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=2)), ['learning_rate','n_estimators'], [[0.01,2.0],[1,1000]], param_opt=[0.0190, 837]),  nclf('xgb',XGBoostClassifier(), ['n_estimators','eta'], [[10,1000],[0.01,1.0]], param_opt=[15, 0.59]),  nclf('nn',"no classifier needed for nn", ['n_hidden_layers','dimof_middle'], [[0,1],[100,500]],param_opt=[0, 309])]

systematics_fraction = 0.01

param_name_list  = ["alphaSvalue"]
#param_list = [0.125,0.130,0.132,0.133,0.134,0.135,0.14] # alphaSvalue
param_list = [0.130]
param_to_optimise = 0.133
param_monash = 0.1365

file_name_patterns= [ os.environ['monash']+"/GoF_input/GoF_input_udsc_monash_lower_level_{1}.txt", os.environ['monash']+"/GoF_input/GoF_input_udsc_"+str(param_to_optimise)+ "_"+param_name_list[0]+"_lower_level_{1}.txt"]

name_CPV= "monash__CPV"
name_noCPV= "monash__noCPV"
title_CPV="monash alphaS "+str(param_to_optimise)
title_noCPV = "monash monash "
directory_name = "_monash_lower_level_2files"

# possible : 'opt', 'eval', 'plot' or combination thereof
MODE= 'eval'


start_time = time.time()
if 'opt' in MODE:
	expt = experiment(nclf_list=nclf_list, file_name_patterns=file_name_patterns, scoring='chi2',single_no_bins_list = [5], systematics_fraction = systematics_fraction, only_mod=True, title_CPV=title_CPV, title_noCPV=title_noCPV, name_CPV=name_CPV, name_noCPV=name_noCPV, directory_name=directory_name)

	expt.optimise(optimisation_dimension = 8, number_of_iterations=50)

evaluation_start_time = time.time()
print(50*"-"+"\noptimisation took  ", (evaluation_start_time - start_time)/60. , "  minutes\n" +50*"-")
#print(nclf_list[0].param_list)


if 'eval' in MODE:
	for param in param_list:
		#file_name_patterns= [ os.environ['learningml']+"/GoF/data/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_8D_10000_0.0_1.0_1.0_{1}.txt", os.environ['learningml']+"/GoF/data/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_8D_10000_0.0_0.95_0.95_{1}.txt" ]
		file_name_patterns= [ os.environ['monash']+"/GoF_input/GoF_input_udsc_monash_lower_level_{1}.txt", os.environ['monash']+"/GoF_input/GoF_input_udsc_"+str(param)+"_"+param_name_list[0]+"_lower_level_{1}.txt"]
		name_CPV= "monash_"+str(param)+"_"+param_name_list[0]+"_lower_level"
		name_noCPV= "monash_"+str(param_monash)+"_"+param_name_list[0]+"_lower_level"
		title_CPV = "Monash "+str(param)+" "+param_name_list[0]
		title_noCPV="Monash "+str(param_monash)+" "+param_name_list[0]
		directory_name = "_monash_lower_level_2files_attempt1"

		if param == param_list[-1]: only_mod=False
                else: only_mod = True

		expt = experiment(nclf_list=nclf_list, file_name_patterns=file_name_patterns, scoring='chi2',single_no_bins_list = [5], systematics_fraction = systematics_fraction, only_mod=only_mod, title_CPV=title_CPV, title_noCPV=title_noCPV, name_CPV=name_CPV, name_noCPV=name_noCPV, directory_name=directory_name)


		expt.evaluate(evaluation_dimensions = [param], keras_evaluation_dimensions = [8], number_of_evaluations=10)


end_time = time.time()

print(50*"-"+"\nevaluation took  ", (end_time - evaluation_start_time)/60. , "  minutes\n" +50*"-")



