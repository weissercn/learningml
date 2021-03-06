import os 
import signal 
import numpy as np 
import math 
import sys 
sys.path.insert(0,os.environ["learningml"]+"/GoF") 
import os 
import classifier_eval 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.svm import SVC 
from keras.wrappers.scikit_learn import KerasClassifier 
from rep.estimators import XGBoostClassifier 
# Write a function like this called "main" 
def main(job_id, params): 
	print "Anything printed here will end up in the output directory for job ", job_id 
	print params 

	if job_id>50:file = open("optimisation_done_flag", "a").close()

	comp_file_list= [("/Users/weisser/MIT_Dropbox/MIT/Research/learningml/learningml/GoF/data/accept_reject/sin1diff_data/data_sin1diff_5_and_5_periods4D_sample_optimisation_0.txt","/Users/weisser/MIT_Dropbox/MIT/Research/learningml/learningml/GoF/data/accept_reject/sin1diff_data/data_sin1diff_5_and_6_periods4D_sample_optimisation_0.txt")]

	clf = KerasClassifier(classifier_eval.make_keras_model,n_hidden_layers=params["n_hidden_layers"],dimof_middle=params["dimof_middle"],dimof_input=4)

	result= classifier_eval.classifier_eval(name="nn_4Dsin_5_6_CPV_syst_0_01_",title="nn Sin 5 6 periods syst0.01",comp_file_list=comp_file_list,clf=clf,mode="spearmint_optimisation",scoring="chi2", no_bins=5, systematics_fraction=0.01)

	with open("nn_4Dsin_5_6_CPV_syst_0_01__chi2scoring_5_optimisation_values.txt", "a") as myfile: 
		myfile.write(str(params["n_hidden_layers"][0])+"\t"+ str(params["dimof_middle"][0])+"\t"+str(result)+"\n") 
	return result