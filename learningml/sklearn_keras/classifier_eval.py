

from __future__ import print_function
print(__doc__)
import os
import p_value_scoring_object
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize

from sklearn import cross_validation
from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils, visualize_util
from scipy import stats
import math

# Function definitions


def make_keras_model(n_hidden_layers, dimof_middle, dimof_input):
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Activation, Flatten
        from keras.utils import np_utils, generic_utils
        from keras.wrappers.scikit_learn import KerasClassifier

        dimof_output =1

        print("dimof_input : ",dimof_input, "dimof_output : ", dimof_output)

        batch_size = 100
        dropout = 0.5
        countof_epoch = 5

        model = Sequential()
        model.add(Dense(input_dim=dimof_input, output_dim=dimof_middle, init="glorot_uniform",activation='relu'))
        model.add(Dropout(dropout))

        for n in range(n_hidden_layers):
                model.add(Dense(input_dim=dimof_middle, output_dim=dimof_middle, init="glorot_uniform",activation='relu'))
                model.add(Dropout(dropout))

        model.add(Dense(input_dim=dimof_middle, output_dim=dimof_output, init="glorot_uniform",activation='sigmoid'))

        #Compiling (might take longer)
	model.compile(class_mode='binary',loss='binary_crossentropy', optimizer='adam',metrics=["accuracy"])

	visualize_util.plot(model, to_file='model.png')

        return model


class Counter(object):
    # Creating a counter object to be able to perform cross validation with only one split
    def __init__(self, list1,list2):
        self.current = 1
        self.list1 =list1
        self.list2 =list2

    def __iter__(self):
        'Returns itself as an iterator object'
        return self

    def __next__(self):
        'Returns the next value till current is lower than high'
        if self.current > 1:
            raise StopIteration
        else:
            self.current += 1
            return self.list1,self.list2
    next = __next__ #python2 


def histo_plot_pvalue(U_0,abins,axlabel,aylabel,atitle,aname):
        bins_probability=np.histogram(U_0,bins=abins)[1]

        #Finding the p values corresponding to 1,2 and 3 sigma significance.
        no_one_std_dev=sum(i < (1-0.6827) for i in U_0)
        no_two_std_dev=sum(i < (1-0.9545) for i in U_0)
        no_three_std_dev=sum(i < (1-0.9973) for i in U_0)

        print(no_one_std_dev,no_two_std_dev,no_three_std_dev)

        with open(aname+"_p_values_1_2_3_std_dev.txt",'w') as p_value_1_2_3_std_dev_file:
                p_value_1_2_3_std_dev_file.write(str(no_one_std_dev)+'\t'+str(no_two_std_dev)+'\t'+str(no_three_std_dev)+'\n')

        #plt.rc('text', usetex=True)
        textstr = '$1\sigma=%i$\n$2\sigma=%i$\n$3\sigma=%i$'%(no_one_std_dev, no_two_std_dev, no_three_std_dev)


        # Making a histogram of the probability predictions of the algorithm. 
        fig_pred_0= plt.figure()
        ax1_pred_0= fig_pred_0.add_subplot(1, 1, 1)
        n0, bins0, patches0 = ax1_pred_0.hist(U_0, bins=bins_probability, facecolor='red', alpha=0.5)
        ax1_pred_0.set_xlabel(axlabel)
        ax1_pred_0.set_ylabel(aylabel)
        ax1_pred_0.set_title(atitle)
        plt.xlim([0,1])

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        ax1_pred_0.text(0.85, 0.95, textstr, transform=ax1_pred_0.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

        fig_pred_0.savefig(aname+"_p_values_plot.png")
        #fig_pred_0.show()
        plt.close(fig_pred_0)



#######################################################################################################################################################################################################
#######################################################################################################################################################################################################
#######################################################################################################################################################################################################



def classifier_eval(*args,**kwargs):
#mode,keras_mode,args):
        ##############################################################################
        # Setting parameters
        #

	# Printing arguments supplied to this function
        print("\nArguments")
        for count, thing in enumerate(args):
                print(count, thing)
            
        print("\nKey word arguments")
        for name, value in kwargs.items():
                print(name, " = ", value)
	print("\n")


	#Setting the allowed values for the different modes
	mode_allowed=		["evaluation","spearmint_optimisation"]
	scoring_allowed= 	["standard","AD","visualisation","accuracy"]

	#Setting all parameters
	name		= kwargs.get('name',"name")
	sample1_name	= kwargs.get('sample1_name',"first sample A")
	sample2_name    = kwargs.get('sample2_name',"second sample B")
	shuffling_seed	= kwargs.get('shuffling_seed',100)
	comp_file_list	= kwargs.get('comp_file_list')
	cv_n_iter	= kwargs.get('cv_n_iter',1)
	clf		= kwargs.get('clf')

	mode		= kwargs.get('mode',"evaluation")
	scoring         = kwargs.get('scoring',"standard")

	#Testing if the parameters are as expected
	assert (isinstance(name, str)),			"The name needs to be a string"
	assert (isinstance(sample1_name, str)),         "The sample1_name needs to be a string"
	assert (isinstance(sample2_name, str)),         "The sample2_name needs to be a string"
	assert (isinstance(shuffling_seed, int)),       "The shuffling_seed needs to be an integer"	
	assert (isinstance(cv_n_iter, int)),       	"The cv_n_iter needs to be an integer"

	assert (mode in mode_allowed),                  "No valid mode chosen!"
	assert (scoring in scoring_allowed),		"No valid scoring strategy chosen!"


	score_list=[]

        ##############################################################################
        # Load and prepare data set
        #
        # dataset for grid search

        for comp_file_0,comp_file_1 in comp_file_list:
                print("Operating of files :"+comp_file_0+"   "+comp_file_1)

                #extracts data from the files
                features_0=np.loadtxt(comp_file_0,dtype='d')
                features_1=np.loadtxt(comp_file_1,dtype='d')

                #determine how many data points are in each sample
                no_0=features_0.shape[0]
                no_1=features_1.shape[0]
                no_tot=no_0+no_1
                #Give all samples in file 0 the label 0 and in file 1 the feature 1
                label_0=np.zeros((no_0,1))
                label_1=np.ones((no_1,1))

                #Create an array containing samples and features.
                data_0=np.c_[features_0,label_0]
                data_1=np.c_[features_1,label_1]

                data=np.r_[data_0,data_1]

                np.random.shuffle(data)

                X=data[:,:-1]
                y=data[:,-1]
                print("X : ",X)
                print("y : ",y)
                atest_size=0.2
                if cv_n_iter==1:
                        train_range = range(int(math.floor(no_tot*(1-atest_size))))
                        test_range  = range(int(math.ceil(no_tot*(1-atest_size))),no_tot)
                        #print("train_range : ", train_range)
                        #print("test_range : ", test_range)
                        acv = Counter(train_range,test_range)
                        #print(acv)
                else:
                        acv = StratifiedShuffleSplit(y, n_iter=cv_n_iter, test_size=atest_size, random_state=42)

                print("Finished with setting up samples")

                # It is usually a good idea to scale the data for SVM training.
                # We are cheating a bit in this example in scaling all of the data,
                # instead of fitting the transformation on the training set and
                # just applying it on the test set.

                if not scoring=="visualisation":
                        scaler = StandardScaler()
                        X = scaler.fit_transform(X)

		if scoring=="AD":
			scores = (-1)*cross_validation.cross_val_score(clf,X,y,cv=acv,scoring=p_value_scoring_object.p_value_scoring_object_AD)
		elif scoring=="visualisation":
			print("X[:,0].min() , ", X[:,0].min(), "X[:,0].max() : ", X[:,0].max())
			scores = (-1)*cross_validation.cross_val_score(clf,X,y,cv=acv,scoring=p_value_scoring_object.p_value_scoring_object_visualisation)
			import os
			os.rename("visualisation.png",name+"_visualisation.png")
		elif scoring=="accuracy":
			scores = cross_validation.cross_val_score(clf,X,y,cv=acv,scoring='accuracy')
		elif scoring=="standard":
			scores = (-1)*cross_validation.cross_val_score(clf,X,y,cv=acv,scoring=p_value_scoring_object.p_value_scoring_object)
		else:
			print("No valid mode chosen")

		print("scores : ",scores)
		score_list.append(np.mean(scores))
		print("\n",'#'*200,"\n")
		if mode=="spearmint_optimisation":
			return np.mean(scores)

        if mode=="evaluation":
                # The score list has been computed. Let's plot the distribution
                print(score_list)
                with open(name+"_p_values",'w') as p_value_file:
                        for item in score_list:
                                p_value_file.write(str(item)+'\n')
                histo_plot_pvalue(score_list,50,"p value","Frequency","p value distribution",name)



if __name__ == "__main__":
        print("Executing classifier_eval_simplified as a stand-alone script")
        print()

	from sklearn import tree
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.svm import SVC 


	for dim in range(5,6):
		comp_file_list=[]
	    
		####################################################################
		# Gaussian samples operation
		####################################################################

		for i in range(100):
			comp_file_list.append((os.environ['learningml']+"/sklearn_keras/data/gaussian_same_projection_on_each_axis/gaussian_same_projection_on_each_axis_redefined_{1}D_1000_0.6_0.2_0.1_{0}.txt".format(i,dim),os.environ['learningml']+"/sklearn_keras/data/gaussian_same_projection_on_each_axis/gaussian_same_projection_on_each_axis_redefined_{1}D_1000_0.6_0.2_0.075_{0}.txt".format(i,dim)))
	    
		#clf = SVC(C=290.4,gamma=0.0961,probability=True, cache_size=7000)
		clf = KerasClassifier(make_keras_model,n_hidden_layers=1,dimof_middle=300,dimof_input=dim)

		####################################################################


		classifier_eval(name=(str(dim)+"Dgaussian_same_projection_redefined__0_1__0_075_optimised_svm"),comp_file_list=comp_file_list,clf=clf, scoring="standard")




