#import the important libraries
import setup_optimise_hyperparam 
from setup_optimise_hyperparam import name_to_nclf, nclf, experiment, make_keras_model
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from rep.estimators import XGBoostClassifier
from keras.wrappers.scikit_learn import KerasClassifier


def spearmint_example():

	#Quick start. Use the predefined classifiers
	#nclf_list = [name_to_nclf("bdt"), name_to_nclf("xgb"), name_to_nclf("svm"), name_to_nclf("nn")]

	#Or specify your classifiers yourself
	nclf_list = [nclf('xgb',XGBoostClassifier(), ['n_estimators','eta','max_depth'], [[10,1000],[0.01,1.0],[2,10]])]

	name= "{0}D_gauss"

	#These files are opened two folders higher, so either specify a full path or append "../../" to the 
	file_name_patterns= ["../../gaussian_same_projection_on_each_axis_4D_10000_0.0_1.0_1.0_optimisation_0.txt","../../gaussian_same_projection_on_each_axis_4D_10000_0.0_0.95_0.95_optimisation_0.txt"]


	directory_name = "_gauss"

	#expt = experiment(nclf_list=nclf_list, file_name_patterns=file_name_patterns, scoring='chi2',single_no_bins_list = [5], name_CPV=name, directory_name=directory_name)
	expt = experiment(nclf_list=nclf_list, file_name_patterns=file_name_patterns, name_CPV=name, directory_name=directory_name)

	expt.optimise(number_of_iterations=50)

if __name__ == "__main__":
	spearmint_example()


