

from __future__ import print_function
print(__doc__)
import os
import sys
import numpy as np
#import matplotlib
#matplotlib.use('AGG')  # Do this BEFORE importing matplotlib.pyplot
import matplotlib.pyplot as plt 
#from matplotlib.colors import Normalize
import matplotlib.colors as colors
import matplotlib.cm as cm


from sklearn import cross_validation
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from rep.estimators import XGBoostClassifier
from sklearn.svm import SVC
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils, visualize_util
from scipy import stats
import math

import p_value_scoring_object
import subprocess
import multiprocessing
import shutil
import time
#import signal
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

        print(aname, " : ", no_one_std_dev,no_two_std_dev,no_three_std_dev)

        with open(aname+"_p_values_1_2_3_std_dev.txt",'w') as p_value_1_2_3_std_dev_file:
                p_value_1_2_3_std_dev_file.write(str(no_one_std_dev)+'\t'+str(no_two_std_dev)+'\t'+str(no_three_std_dev)+'\n')

        #plt.rc('text', usetex=True)
        textstr = '$1\sigma=%i$\n$2\sigma=%i$\n$3\sigma=%i$'%(no_one_std_dev, no_two_std_dev, no_three_std_dev)

	plt.figure()
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

def plot_2D_opt_graph(expt,nclf):

	class_name = nclf.name
	name = expt.name_CPV.format(expt.optimisation_dimension)

	if expt.scoring=="chi2":        self_name_CPV= expt.name_CPV+ "_chi2scoring_" + str(expt.optimisation_no_bins)
	else:                           self_name_CPV= expt.name_CPV

	filename= nclf.name +"_"+ self_name_CPV.format(expt.optimisation_dimension)+"_optimisation_values"
	data=np.loadtxt(filename+".txt",dtype='d')
	number_of_iterations = data.shape[0]
	if number_of_iterations > expt.number_of_iterations: number_of_iterations = expt.number_of_iterations 

	#print("data : ", data)
	#print("number_of_iterations : ",number_of_iterations)
	x=data[:number_of_iterations,0]
	y=data[:number_of_iterations,1]
	z=data[:number_of_iterations,2]
	avmin= np.min(z)
	cm = plt.cm.get_cmap('RdYlBu')

	fig= plt.figure()
	ax1= fig.add_subplot(1, 1, 1)

	sc = ax1.scatter(x,y,c=z,s=35,cmap=cm, norm=colors.LogNorm(),vmin=avmin,vmax=1)

	#print("z : ",z)
	index = np.argmin(z)
	#print("index of max : ",index)
	val_max = [x[index],y[index],z[index]]
	#print("\n",nclf.name," values of max : ",val_max,"\n")
	ax1.scatter(x[index],y[index],c=z[index], norm=colors.LogNorm(),s=50, cmap=cm,vmin=avmin,vmax=1)

	avmin_power = int(math.floor(math.log10(avmin)))
	ticks, ticklabels = [1],['1']
	for i in range(1,-avmin_power+1):
		ticks.append(np.power(10.,float(-i)))
		ticklabels.append('1E-'+str(i))

	cb=fig.colorbar(sc,ticks=ticks)
	cb.ax.set_yticklabels(ticklabels)
	cb.set_label('p value')

	ax1.set_xlabel(nclf.param_list[0])
	ax1.set_ylabel(nclf.param_list[1])
	ax1.set_title('hyperparam opt '+class_name+"\n"+expt.title_CPV)
	print(nclf.name," saving to "+filename+".png \n")
	fig.savefig(filename+".png")

	return val_max

#######################################################################################################################################################################################################
#######################################################################################################################################################################################################
#######################################################################################################################################################################################################


class nclf(object):
	def __init__(self, name = 'dt',  clf = tree.DecisionTreeClassifier(),  param_list = ['max_depth','min_samples_split'],  range_list = [[1, 60],[2,100]], param_opt=[],clf_nn_dict={} ):
		self.name        = name 
		self.clf         = clf
		self.param_list  = param_list
		self.range_list  = range_list
		self.param_opt   = param_opt
		self.clf_nn_dict = clf_nn_dict

		assert(len(param_list)==2), "only 2 parameters can be varied"

	def __str__(self):
		print("\nname : ", self.name, "\nclf : ", self.clf, "\nparam_list : ", self.param_list, "\nrange_list : ", self.range_list, "\nparam_opt : ", self.param_opt)



def name_to_nclf(name):
        if name=="dt":
        	anclf = nclf('dt',tree.DecisionTreeClassifier(),['max_depth','min_samples_split'], [[1, 60],[2,100]])
	if name=="bdt":
		anclf = nclf('bdt',AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=2)), ['learning_rate','n_estimators'], [[0.01,2.0],[1,1000]])
	if name=="xgb":
		anclf = nclf('xgb',XGBoostClassifier(), ['n_estimators','eta'], [[10,1000],[0.01,1.0]])
	if name=="svm":
		anclf = nclf('svm',SVC(probability=True, cache_size=7000), ['C','gamma'], [[1.0,1000.0],[1E-6,0.1]])
	if name=="nn":
		anclf = nclf('nn',"no classifier needed for nn", ['n_hidden_layers','dimof_middle'], [[0,1],[100,500]])
        return anclf 


#######################################################################################################################################################################################################
#######################################################################################################################################################################################################
#######################################################################################################################################################################################################

def optimise_job(expt,nclf,out_q):

	print("\n"+nclf.name+" nclf.clf :",nclf.clf,"\n")
	print(multiprocessing.current_process().name, " : ",nclf.name , " Starting")
	if not os.path.exists(nclf.name):
		os.makedirs(nclf.name)
	os.chdir(nclf.name)
	if os.path.exists("optimisation_done_flag"):os.remove("optimisation_done_flag")
	expt.write_config(nclf)
	expt.write_classifier_eval_wrapper(nclf)
	
	# Starting process and 
	spearmint_output_file = open("spearmint_output_"+nclf.name, 'w')	
	p = subprocess.Popen(["python", "{}/main.py".format(expt.spearmint_directory),"."],stdout=spearmint_output_file, stderr=subprocess.STDOUT)

	print("\n",nclf.name," optimisation is running")
	while not os.path.isfile("optimisation_done_flag"):time.sleep(1)
	#os.killpg(os.getpgid(p.pid), signal.SIGTERM)
	p.kill()
	spearmint_output_file.close()
	#(output, err) = p.communicate()
	#print("Command output : ",output)	
	print("\n",nclf.name," optimisation done")


	param_opt = plot_2D_opt_graph(expt, nclf)
	print("\n"+nclf.name, "nclf.param_list : ",nclf.param_list)
	print(nclf.name, "param_opt : ", param_opt)
	nclf.param_opt = []
	for i in range(2):
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


def evaluate_job(expt,nclf,out_q):

	print("\n",multiprocessing.current_process().name, " : ",nclf.name ," Starting")

	if not os.path.exists(nclf.name):
		os.makedirs(nclf.name)
	os.chdir(nclf.name)
	for param_index, param in enumerate(nclf.param_list):
		if nclf.name == "nn":
			for dim in expt.evaluation_dimensions:
				nclf.clf_nn_dict[str(dim)] = KerasClassifier(build_fn=make_keras_model,n_hidden_layers=nclf.param_opt[0],dimof_middle=nclf.param_opt[1],dimof_input=dim)
		else:
			#print("nclf.param_list :",nclf.param_list)
			#print("nclf.param_opt :",nclf.param_opt)
			setattr(nclf,param ,nclf.param_opt[param_index])
	print("\n",nclf.name+" nclf.clf :",nclf.clf,"\n")
	for dim in expt.evaluation_dimensions:
		comp_file_list=[]
		for i in range(expt.number_of_evaluations):
			comp_file_list.append((expt.file_name_patterns[0].format(dim,i),expt.file_name_patterns[1].format(dim,i)))
		if nclf.name == "nn": aclf = nclf.clf_nn_dict[str(dim)]
		else: aclf=nclf.clf
		if expt.scoring=='chi2':
			for no_bins in expt.single_no_bins_list:
				classifier_eval(name= nclf.name + "_" + expt.name_CPV.format(dim) , comp_file_list=comp_file_list, clf =aclf, verbose=False, scoring=expt.scoring, no_bins=no_bins, systematics_fraction=expt.systematics_fraction, title=expt.title_CPV+" "+ str(dim)+"D" )
		else:
			classifier_eval(name= nclf.name + "_" + expt.name_CPV.format(dim) , comp_file_list=comp_file_list, clf =aclf, verbose=False, scoring=expt.scoring, title=expt.title_noCPV+" "+ str(dim)+"D")
	if not expt.only_mod:
		print(nclf.name,"Running NoCPV ")
		for dim in expt.evaluation_dimensions:
			comp_file_list=[]
			for i in range(expt.number_of_evaluations):
				comp_file_list.append((expt.file_name_patterns[0].format(dim,i),expt.file_name_patterns[0].format(dim,100+i)))
			if nclf.name == "nn": aclf = nclf.clf_nn_dict[str(dim)]
			else: aclf=nclf.clf
			if expt.scoring=='chi2':
				for no_bins in expt.single_no_bins_list:
					classifier_eval(name= nclf.name + "_" +expt.name_noCPV.format(dim) , comp_file_list=comp_file_list, clf =aclf, verbose=False, scoring=expt.scoring, no_bins=no_bins, systematics_fraction=expt.systematics_fraction, title=expt.title+" "+ str(dim)+"D")
			else:
				classifier_eval(name= nclf.name + "_" +expt.name_noCPV.format(dim) , comp_file_list=comp_file_list, clf =aclf, verbose=False, scoring=expt.scoring, title=expt.title+" "+ str(dim)+"D")

	os.chdir("..")
	print(multiprocessing.current_process().name, " : ",nclf.name ," Finishing")

def worker(expt, nclf):
    """thread worker function"""
    name = multiprocessing.current_process().name
    print(name, " : ",nclf.name , ' Starting')
    if nclf.name == "xgb": time.sleep(5)
    else: time.sleep(10)
    print(name, " : ",nclf.name , ' Exiting')
    return



#######################################################################################################################################################################################################
#######################################################################################################################################################################################################
#######################################################################################################################################################################################################

class experiment(object):
	def __init__(self,**kwargs): 

		self.nclf_list 		= kwargs.get('nclf_list',[nclf()])
		self.file_name_patterns = kwargs.get('file_name_patterns', [ os.environ['learningml']+"/GoF/data/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_redefined_{0}D_1000_0.6_0.2_0.1_{1}.txt", os.environ['learningml']+"/GoF/data/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_redefined_{0}D_1000_0.6_0.2_0.075_{1}.txt" ])
		self.only_mod 		= kwargs.get('only_mod',False)
		self.n_cores 		= kwargs.get('n_cores',7)
		self.name_CPV 		= kwargs.get('name_CPV',"{0}Dname_CPV")
		self.name_noCPV		= kwargs.get('name_noCPV',"{0}Dname_noCPV")
		self.title_CPV		= kwargs.get('title_CPV','title_CPV')
		self.title_noCPV	= kwargs.get('title_noCPV','title_noCPV')
		self.scoring		= kwargs.get('scoring',"standard")
		self.single_no_bins_list= kwargs.get('single_no_bins_list',[2])
		self.systematics_fraction= kwargs.get('systematics_fraction',0.01)

		self.name_CPV		= self.name_CPV+ "_syst_"+str(self.systematics_fraction).replace(".","_") + "_"

		self.title_CPV		= self.title_CPV + " syst" + str(self.systematics_fraction)
		self.title_noCPV          = self.title_noCPV + " syst" + str(self.systematics_fraction)	

	def set_name_CPV(self,name_CPV):			self.name_CPV = name_CPV
	def set_name_noCPV(self,name_noCPV): 			self.name_noCPV = name_noCPV
	def set_nclf_list(self,nclf_list): 			self.nclf_list=nclf_list
	def set_file_name_patterns(self,file_name_patterns): 	self.file_name_patterns = file_name_patterns


	def optimise(self,**kwargs):

		self.optimisation_dimension 	= kwargs.get('optimisation_dimension',2)
		self.number_of_iterations	= kwargs.get('number_of_iterations',50)
		self.optimisation_no_bins 	= kwargs.get('optimisation_no_bins',self.single_no_bins_list[0])
		self.spearmint_directory 	= kwargs.get('spearmint_directory', "/Users/weisser/Documents/Spearmint-master/spearmint")	
		
		opt_dir = "optimisation"
		if not os.path.exists(opt_dir):
			os.makedirs(opt_dir)
		os.chdir(opt_dir)
		
		print(os.getcwd())
		os.system(os.environ['learningml']+"/GoF/reinitialise_spearmint.sh")
		

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

	def evaluate(self,evaluation_dimensions=[2],number_of_evaluations=100, **kwargs):

		self.evaluation_dimensions = evaluation_dimensions
		self.number_of_evaluations = number_of_evaluations
		eval_dir = "evaluation"
		if not os.path.exists(eval_dir):
                        os.makedirs(eval_dir)
		os.chdir(eval_dir)

		out_q = multiprocessing.Queue()

		print("operating on file_name_patterns :" , self.file_name_patterns)
		jobs = []
		for nclf in self.nclf_list:
			p = multiprocessing.Process(target=evaluate_job, args=(self,nclf,out_q,))
			jobs.append(p)
			p.start()

                # Wait until jobs have finished
                for j in jobs: j.join()
		print("\n","/"*100,"All EVALUATION jobs finished running","/"*100)
		os.chdir("..")





	def write_classifier_eval_wrapper(self, nclf):
		classifier_content = 'import os \nimport signal \nimport numpy as np \nimport math \nimport sys \nsys.path.insert(0,os.environ["learningml"]+"/GoF") \nimport os \nimport classifier_eval \nfrom sklearn.tree import DecisionTreeClassifier \nfrom sklearn.ensemble import AdaBoostClassifier \nfrom sklearn.svm import SVC \nfrom keras.wrappers.scikit_learn import KerasClassifier \nfrom rep.estimators import XGBoostClassifier \n# Write a function like this called "main" \ndef main(job_id, params): \n\tprint "Anything printed here will end up in the output directory for job ", job_id \n\tprint params \n\n'
		classifier_content += '\tif job_id>{}:file = open("optimisation_done_flag", "a").close()\n\n'.format(self.number_of_iterations)
		#classifier_content += '\tassert (job_id<{}), "Final number of iterations reached" \n\n'.format(1+self.number_of_iterations)
		#classifier_content += '\tif job_id>{}: \n\t\tprint("Killing parent process : ", os.getppid(),"\\n"*3) \n\t\tos.kill(os.getppid(), signal.SIGTERM) \n\n'.format(self.number_of_iterations)
		classifier_content += '\tcomp_file_list= [("{}","{}")]\n\n'.format(self.file_name_patterns[0].format(self.optimisation_dimension,"optimisation_0"),self.file_name_patterns[1].format(self.optimisation_dimension,"optimisation_0"))
		if nclf.name == "nn":
			classifier_content += '\tclf = KerasClassifier(classifier_eval.make_keras_model,n_hidden_layers=params["n_hidden_layers"],dimof_middle=params["dimof_middle"],dimof_input={})'.format(self.optimisation_dimension)
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
		classifier_content += '\n\n\tresult= classifier_eval.classifier_eval(name="{}",title="{}",comp_file_list=comp_file_list,clf=clf,mode="spearmint_optimisation",scoring="{}", no_bins={}, systematics_fraction={})\n\n'.format(nclf.name + "_" +self.name_CPV.format(self.optimisation_dimension)+"_optimisation",nclf.name+" "+self.title_CPV,self.scoring,self.optimisation_no_bins,self.systematics_fraction)
		if self.scoring=="chi2": 	self_name_CPV= self.name_CPV+ "_chi2scoring_" + str(self.optimisation_no_bins)
		else:				self_name_CPV= self.name_CPV
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
		
#######################################################################################################################################################################################################
#######################################################################################################################################################################################################
#######################################################################################################################################################################################################


def classifier_eval(*args,**kwargs):
#mode,keras_mode,args):
        ##############################################################################
        # Setting parameters
        #


	#Setting the allowed values for the different modes
	mode_allowed=		["evaluation","spearmint_optimisation"]
	scoring_allowed= 	["standard","AD","visualisation","accuracy","chi2","chi2_2bin","closure_test_delete"]

	#Setting all parameters
	name		= kwargs.get('name',"name")
	title		= kwargs.get('title','title')
	sample1_name	= kwargs.get('sample1_name',"original")
	sample2_name    = kwargs.get('sample2_name',"modified")
	shuffling_seed	= kwargs.get('shuffling_seed',100)
	comp_file_list	= kwargs.get('comp_file_list')
	cv_n_iter	= kwargs.get('cv_n_iter',1)
	clf		= kwargs.get('clf')
	systematics_fraction = kwargs.get('systematics_fraction',0.01)


	mode		= kwargs.get('mode',"evaluation")
	scoring         = kwargs.get('scoring',"standard")
	verbose		= kwargs.get('verbose',True)
	single_no_bins_list = kwargs.get('single_no_bins_list',[2])
	no_bins		= kwargs.get('no_bins',2)
	#Testing if the parameters are as expected
	assert (isinstance(name, str)),			"The name needs to be a string"
	assert (isinstance(sample1_name, str)),         "The sample1_name needs to be a string"
	assert (isinstance(sample2_name, str)),         "The sample2_name needs to be a string"
	assert (isinstance(shuffling_seed, int)),       "The shuffling_seed needs to be an integer"	
	assert (isinstance(cv_n_iter, int)),       	"The cv_n_iter needs to be an integer"

	assert (mode in mode_allowed),                  "No valid mode chosen!"
	assert (scoring in scoring_allowed),		"No valid scoring strategy chosen!"

	if verbose:
		# Printing arguments supplied to this function
		print("\nArguments")
		for count, thing in enumerate(args):
			print(count, thing)
	    
		print("\nKey word arguments")
		for item_name, item_value in kwargs.items():
			print(item_name, " = ", item_value)
        	print("\n")

	if scoring=="chi2": name = name + "_chi2scoring_{}".format(no_bins)

	score_list=[]
	
	if "chi2_2bin" in scoring: score_chi_list_list=[]

        ##############################################################################
        # Load and prepare data set
        #
	counter = 0
        for comp_file_0,comp_file_1 in comp_file_list:
		if counter in [10,20,30,40,50,60,70,80,90]: print(name, " iteration : ", counter)
		
		if not verbose: sys.stdout = open(os.devnull, "w")
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
			os.rename("visualisation.png",name+"_visualisation.png")
		elif scoring=="chi2":
			#We have to use function closure
			scoring_function = p_value_scoring_object.make_p_value_scoring_object_binned_chisquared(no_bins,systematics_fraction,title,name,not counter )
			scores = (-1)*cross_validation.cross_val_score(clf,X,y,cv=acv,scoring=scoring_function)
			#if not counter:  os.rename("1D_" + str(no_bins)+"_bins" +"_bin_definitions_1D.png",name+"_bin_definitions_1D.png")
		elif scoring=="chi2_2bin":
                        scores_chi2_list=[]
                        #We have to use function closure
                        for no_bins in single_no_bins_list:
                                scores = (-1)*cross_validation.cross_val_score(clf,X,y,cv=acv,scoring=p_value_scoring_object.p_value_scoring_object_binned_chisquared_2bin)
                                scores_chi2_list.append(np.mean(scores))
			
		elif scoring=="accuracy":
			scores = cross_validation.cross_val_score(clf,X,y,cv=acv,scoring='accuracy')
		elif scoring=="standard":
			scores = (-1)*cross_validation.cross_val_score(clf,X,y,cv=acv,scoring=p_value_scoring_object.p_value_scoring_object)
		elif scoring=="closure_test_delete":
			scores = (-1)*cross_validation.cross_val_score(clf,X,y,cv=acv,scoring=p_value_scoring_object.make_p_value_scoring_object_test("hi"))
		else:
			print("No valid mode chosen")

		if "chi2_2bin" in scoring:
			score_chi_list_list.append(scores_chi2_list)
		else:
			print("scores : ",scores)
			score_list.append(np.mean(scores))
		print("\n",'#'*200,"\n")
		
		counter += 1
		if not verbose: sys.stdout = sys.__stdout__
		if mode=="spearmint_optimisation":
			return np.mean(scores)

        if mode=="evaluation":
                # The score list has been computed. Let's plot the distribution

		if scoring=="chi2_2bin":
			score_chi_list_list=np.array(score_chi_list_list).T.tolist()
			if verbose: print("score_chi_list_list : \n",score_chi_list_list)
			for no_bins_index, no_bins in enumerate(single_no_bins_list):
				with open(name+"chi2scoring_{}_bins_p_values".format(no_bins),'w') as p_value_file:
	                                for item in score_chi_list_list[no_bins_index]:
						p_value_file.write(str(item)+'\n')
					histo_plot_pvalue(score_chi_list_list[no_bins_index],50,"p value","Frequency","p value distribution",name)
                else:
			if verbose: print("score_list : \n",score_list)
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




