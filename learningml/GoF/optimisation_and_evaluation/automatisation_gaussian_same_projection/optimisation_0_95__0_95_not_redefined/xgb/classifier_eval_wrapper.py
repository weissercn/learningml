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

	comp_file_list= [("/Users/weisser/MIT_Dropbox/MIT/Research/learningml/learningml/GoF/data/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_4D_10000_0.0_1.0_1.0_optimisation_0.txt","/Users/weisser/MIT_Dropbox/MIT/Research/learningml/learningml/GoF/data/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_4D_10000_0.0_0.95_0.95_optimisation_0.txt")]

	clf = XGBoostClassifier(base_score=0.5,colsample=1.0,eta=params['eta'],features=None,gamma=None,max_depth=6,min_child_weight=1.0,missing=-999.0,n_estimators=params['n_estimators'],nthreads=16,num_feature=None,random_state=0,scale_pos_weight=1.0,subsample=1.0,verbose=0)

	result= classifier_eval.classifier_eval(name="xgb_4Dgauss__0_95__0_95_CPV_not_redefined_syst_0_01_",title="xgb Gauss 0.95 0.95 syst0.01",comp_file_list=comp_file_list,clf=clf,mode="spearmint_optimisation",scoring="chi2", no_bins=5, systematics_fraction=0.01)

	with open("xgb_4Dgauss__0_95__0_95_CPV_not_redefined_syst_0_01__chi2scoring_5_optimisation_values.txt", "a") as myfile: 
		myfile.write(str(params["n_estimators"][0])+"\t"+ str(params["eta"][0])+"\t"+str(result)+"\n") 
	return result