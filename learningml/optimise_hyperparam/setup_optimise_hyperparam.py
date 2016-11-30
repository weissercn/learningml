#import importantant libraries
from __future__ import print_function
import os
import sys
import numpy as np
from sklearn import tree
from sklearn import cross_validation
from sklearn.ensemble import AdaBoostClassifier
from rep.estimators import XGBoostClassifier
from rep.estimators import SklearnClassifier
from sklearn.svm import SVC
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils, visualize_util
import subprocess
import multiprocessing
import shutil
import time

def make_keras_model(n_hidden_layers, dimof_middle, dimof_input):
	#This function sets up a neural network
	#The batch size, dropout and overall architecture should be changed here.
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

class nclf(object):
	# nclf is a class that contains a classifier, its name and parameters to optimise and their ranges
        def __init__(self, name = 'dt',  clf = tree.DecisionTreeClassifier(),  param_list = ['max_depth','min_samples_split'],  range_list = [[1, 60],[2,100]], param_opt=[],clf_nn_dict={} ):
                self.name        = name
                self.clf         = clf
                self.param_list  = param_list
                self.range_list  = range_list
                self.param_opt   = param_opt
                self.clf_nn_dict = clf_nn_dict

                #assert(len(param_list)==2), "only 2 parameters can be varied for now"

        def __str__(self):
                print("\nname : ", self.name, "\nclf : ", self.clf, "\nparam_list : ", self.param_list, "\nrange_list : ", self.range_list, "\nparam_opt : ", self.param_opt)



def name_to_nclf(name):
	#This function gives some standard versions of common machine learning classifiers.
        if name=="dt":
                anclf = nclf('dt',tree.DecisionTreeClassifier(),['max_depth','min_samples_split'], [[1, 60],[2,100]])
        if name=="bdt":
                anclf = nclf('bdt',AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=2)), ['learning_rate','n_estimators'], [[0.01,2.0],[100,1000]])
        if name=="xgb":
                anclf = nclf('xgb',XGBoostClassifier(), ['n_estimators','eta'], [[10,1000],[0.01,1.0]])
        if name=="svm":
                anclf = nclf('svm',SVC(probability=True, cache_size=7000), ['C','gamma'], [[1.0,1000.0],[1E-6,0.1]])
        if name=="nn":
                anclf = nclf('nn',"no classifier needed for nn", ['n_hidden_layers','dimof_middle'], [[0,1],[100,500]])
        return anclf


def optimise_job(expt,nclf,out_q):
	#Each classifier is optimised in a different job
        print("\n"+nclf.name+" nclf.clf :",nclf.clf,"\n")
        print(multiprocessing.current_process().name, " : ",nclf.name , " Starting")
        if not os.path.exists(nclf.name):
                os.makedirs(nclf.name)
        os.chdir(nclf.name)
        if os.path.exists("optimisation_done_flag"):os.remove("optimisation_done_flag")
        expt.write_config(nclf)
        expt.write_classifier_eval_wrapper(nclf)

        # Starting process 
        spearmint_output_file = open("spearmint_output_"+nclf.name, 'w')
        p = subprocess.Popen(["python", "{}/main.py".format(expt.spearmint_directory),"."],stdout=spearmint_output_file, stderr=subprocess.STDOUT)

        print("\n",nclf.name," optimisation is running")
        while not os.path.isfile("optimisation_done_flag"):time.sleep(1)
        p.kill()
        spearmint_output_file.close()
        print("\n",nclf.name," optimisation done")


        param_opt = plot_2D_opt_graph(expt, nclf)
        print("\n"+nclf.name, "nclf.param_list : ",nclf.param_list)
        print(nclf.name, "param_opt : ", param_opt)
        nclf.param_opt = []
        for i in range(len(nclf.param_list)):
                limits = nclf.range_list[i]
                lower_lim, upper_lim = limits
                if isinstance( lower_lim, ( int, long ) ) and isinstance( upper_lim, ( int, long ) ):
                        nclf.param_opt.append(int(param_opt[i]))
                else:
                        nclf.param_opt.append(param_opt[i])
        print(nclf.name, "nclf.param_opt : ", nclf.param_opt,"\n")

        for param_index, param in enumerate(nclf.param_list):
                if not nclf.name == "nn": setattr(nclf.clf,param ,nclf.param_opt[param_index])
        os.chdir("..")
        #print(nclf.name+" nclf.clf :",nclf.clf)
        out_q.put(nclf)
        print(multiprocessing.current_process().name, " : ",nclf.name ," Finishing")


class experiment(object):
	#A class with which the optimisation can be performed
        def __init__(self,**kwargs):

                self.nclf_list          = kwargs.get('nclf_list',[nclf()])
                self.file_name_patterns = kwargs.get('file_name_patterns') 
		self.name_CPV           = kwargs.get('name_CPV',"{0}Dname_CPV")
                self.directory_name     = kwargs.get('directory_name',"")


        def set_nclf_list(self,nclf_list):                      self.nclf_list=nclf_list

        def optimise(self,**kwargs):

                self.optimisation_dimension     = kwargs.get('optimisation_dimension',2)
                self.keras_optimisation_dimension = kwargs.get('keras_optimisation_dimension',self.optimisation_dimension)
                self.number_of_iterations       = kwargs.get('number_of_iterations',50)
                self.spearmint_directory        = kwargs.get('spearmint_directory', "/Users/weisser/Documents/Spearmint-master/spearmint")

                opt_dir = "optimisation"+self.directory_name
                if not os.path.exists(opt_dir):
                        os.makedirs(opt_dir)
                os.chdir(opt_dir)

                print(os.getcwd())
                #os.system(os.environ['learningml']+"/GoF/reinitialise_spearmint.sh")
		os.system("../reinitialise_spearmint.sh")

                if os.path.exists("MongoDB_bookkeeping"):
                        shutil.rmtree("MongoDB_bookkeeping")
                os.makedirs("MongoDB_bookkeeping")
                os.system("mongod --fork --logpath MongoDB_bookkeeping/example.log --dbpath MongoDB_bookkeeping")

                out_q = multiprocessing.Queue()

                jobs = []
                for jobid, nclf in enumerate(self.nclf_list):
                        p = multiprocessing.Process(target=optimise_job, args=(self,nclf,out_q,))
                        jobs.append(p)
                        p.start()

                for i in range(len(jobs)):
                        self.nclf_list[i] = out_q.get()

                # Wait until jobs have finished
                for j in jobs: j.join()
                for nclf in self.nclf_list: print(nclf.name, " : ",nclf.clf)
                print("\n","/"*100,"All OPTIMISATION jobs finished running","/"*100,"\n")
                os.chdir("..")
                return self.nclf_list

        def write_classifier_eval_wrapper(self, nclf):
                classifier_content = 'import os \nimport signal \nimport numpy as np \nimport math \nimport sys \n'
		#classifier_content += 'sys.path.insert(0,os.environ["learningml"]+"/optimise_hyperparam") \n'
		classifier_content += 'sys.path.insert(0,"../..") \n'
		classifier_content += 'import os \nimport setup_optimise_hyperparam \nfrom sklearn.tree import DecisionTreeClassifier \nfrom sklearn.ensemble import AdaBoostClassifier \nfrom sklearn.svm import SVC \nfrom keras.wrappers.scikit_learn import KerasClassifier \nfrom rep.estimators import XGBoostClassifier \n# Write a function like this called "main" \ndef main(job_id, params): \n\tprint "Anything printed here will end up in the output directory for job ", job_id \n\tprint params \n\n'
                classifier_content += '\tif job_id>{}:file = open("optimisation_done_flag", "a").close()\n\n'.format(self.number_of_iterations)
                #classifier_content += '\tassert (job_id<{}), "Final number of iterations reached" \n\n'.format(1+self.number_of_iterations)
                #classifier_content += '\tif job_id>{}: \n\t\tprint("Killing parent process : ", os.getppid(),"\\n"*3) \n\t\tos.kill(os.getppid(), signal.SIGTERM) \n\n'.format(self.number_of_iterations)
                classifier_content += '\tcomp_file_list= [("{}","{}")]\n\n'.format(self.file_name_patterns[0].format(self.optimisation_dimension,"optimisation_0"),self.file_name_patterns[1].format(self.optimisation_dimension,"optimisation_0"))
                if nclf.name == "nn":
                        classifier_content += '\tclf = KerasClassifier(setup_optimise_hyperparam.make_keras_model,n_hidden_layers=params["n_hidden_layers"],dimof_middle=params["dimof_middle"],dimof_input={})'.format(self.keras_optimisation_dimension)
                else:
                        clf_repr_list =  repr(nclf.clf).split()
                        for param in nclf.param_list:
                                for index, item  in enumerate(clf_repr_list):
                                        if param in item:
                                                head, sep, tail = item.partition(param+"=")
                                                if index ==0: rest = head+sep+ "params['" + param + "'],"
                                                elif index == len(clf_repr_list) -1: rest = sep + "params['" + param + "'])"
                                                else: rest = sep + "params['" + param + "'],"
                                                clf_repr_list[index] = rest

                        classifier_content += '\tclf = '+ ''.join(clf_repr_list)
		
		classifier_content += '\n\n\tresult=setup_optimise_hyperparam.classifier_eval(comp_file_list=comp_file_list,clf=clf)\n'
                #classifier_content += '\n\n\tresult= classifier_eval.classifier_eval_1file_old(name="{}",title="{}",comp_file_list=comp_file_list,clf=clf,mode="spearmint_optimisation",scoring="{}", no_bins={}, systematics_fraction={})\n\n'.format(nclf.name + "_" +self.name_CPV.format(self.optimisation_dimension),nclf.name+" "+self.title_CPV,self.scoring,self.optimisation_no_bins,self.systematics_fraction)
		self_name_CPV= self.name_CPV
                #if self.scoring=="chi2":        self_name_CPV= self.name_CPV+ "_chi2scoring_" + str(self.optimisation_no_bins)
                #else:                           self_name_CPV= self.name_CPV
                classifier_content += '\twith open("{}_optimisation_values.txt", "a") as myfile: \n\t\tmyfile.write(str(params["{}"][0])+"\\t"+ str(params["{}"][0])+"\\t"+str(result)+"\\n") \n\treturn result'.format(nclf.name +"_"+ self_name_CPV.format(self.optimisation_dimension), nclf.param_list[0], nclf.param_list[1])
                with open('classifier_eval_wrapper.py', 'w') as f:
                        f.write(classifier_content)


        def write_config(self,nclf):
                config_content = '{{\n    "language"        : "PYTHON",\n    "main-file"       : "classifier_eval_wrapper.py",\n    "experiment-name" : "{}_{}",\n    "likelihood"      : "NOISELESS",\n    "variables" : {{'.format(nclf.name,self.name_CPV.format(self.optimisation_dimension))
                for param_index, param in enumerate(nclf.param_list):
                        config_content += '\n        "{}" : {{\n            "type" : "'.format(param)
                        lower_lim, upper_lim = nclf.range_list[param_index]
                        if isinstance( lower_lim, ( int, long ) ) and isinstance( upper_lim, ( int, long ) ):
                                config_content += "INT"
                        else:
                                config_content += "FLOAT"
                        config_content += '",\n            "size" : 1,\n            "min"  : {},\n            "max"  : {}\n        }},'.format(lower_lim, upper_lim)
                config_content= config_content[:-1] + '\n    }\n}'
                with open('config.json', 'w') as f:
                        f.write(config_content)


def classifier_eval(*args,**kwargs):
	comp_file_list  = kwargs.get('comp_file_list')
	clf             = kwargs.get('clf')

	for comp_file_0,comp_file_1 in comp_file_list:
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

		result = np.average(cross_validation.cross_val_score(clf, X, y, cv=3))
		return result






