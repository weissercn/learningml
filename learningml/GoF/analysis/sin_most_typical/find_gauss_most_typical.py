from __future__ import print_function
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import time


# Options for mode 'find_most_typical_gauss'  
MODE = 'find_most_typical_gauss'

label_size = 28



################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
mpl.rc('font', family='serif', size=34, serif="Times New Roman")

#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [r'\boldmath']

mpl.rcParams['legend.fontsize'] = "medium"

mpl.rc('savefig', format ="pdf", pad_inches= 0.1)

mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['figure.figsize']  = 8, 6
mpl.rcParams['lines.linewidth'] = 2

def binomial_error(l1):
	err_list = []
	for item in l1:
		if item==1. or item==0.: err_list.append(np.sqrt(100./101.*(1.-100./101.)/101.))
		else: err_list.append(np.sqrt(item*(1.-item)/100.))
	return err_list

################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################


#                                                               T Y P I C A L   -   G A U S S  


################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################



if MODE == 'find_most_typical_gauss':
        dimensions              = [2,6,10]

        ml_classifiers          = ['nn','bdt','xgb','svm']
	
	ml_classifiers_colors   = ['blue','black','green']
        ml_classifiers_bin      = 5

        chi2_color              = 'red'
        chi2_splits             = [1,2,3,4,5,6,7,8,9,10]

        ml_folder_name          = "automatisation_sin/evaluation10000_2files"
        chi2_folder_name        = "sin"

        ml_file_name            = "{1}_{0}Dsin_5_6_10000_CPV_syst_0_01__chi2scoring_{2}_p_values"
        chi2_file_name          = "sin_5_6_10000_CPV_syst_0_01__{0}D_chi2_{1}_splits_p_values"

        title                   = "Sin 5 6"
        name                    = "sin_5_6_"
        CL                      = 0.95

        ml_classifiers_dict={}
        chi2_splits_dict={}

	xwidth = [0.5]*len(dimensions)
	fig = plt.figure()
	ax = fig.add_axes([0.2,0.15,0.75,0.8])

        for ml_classifier_index, ml_classifier in enumerate(ml_classifiers):
                ml_classifiers_dict[ml_classifier]= []
                for dim in dimensions:
			p_values = np.loadtxt(os.environ['learningml']+"/GoF/optimisation_and_evaluation/"+ml_folder_name+"/"+ml_classifier+"/"+ml_file_name.format(dim,ml_classifier,ml_classifiers_bin)).tolist()
			p_diffthan0 = np.square(np.searchsorted(np.sort(p_values),p_values)-len(p_values)/2.0)
                        ml_classifiers_dict[ml_classifier].append(p_diffthan0)
	

	chi2_best = []
        ml_classifiers_dict['chi2']=[]
	for dim in dimensions:
		chi2_splits_dict = {}
		for chi2_split_index, chi2_split in enumerate(chi2_splits):
			p_values = np.loadtxt(os.environ['learningml']+"/GoF/chi2/"+chi2_folder_name+"/"+chi2_file_name.format(dim,chi2_split)).tolist()
	                chi2_splits_dict[str(chi2_split)]=p_values
		p_values_best = []
		for i in range(len(p_values)):
			single_sample_p = []
			for chi2_split_index, chi2_split in enumerate(chi2_splits):
				single_sample_p.append(chi2_splits_dict[str(chi2_split)][i])	
			p_values_best.append(np.min(single_sample_p))
		p_diffthan0 = np.square(np.searchsorted(np.sort(p_values_best),p_values_best)-len(p_values_best)/2.0)
		ml_classifiers_dict['chi2'].append(p_diffthan0)

	#print("len(ml_classifiers_dict['bdt']) : \n", ml_classifiers_dict['chi2'])

	most_typical_index=[]
	for D_index in range(len(dimensions)):
		typicality_value_all_samples=[]
		for i in range(len(p_values)):
			typicality_value=[]
			for key, value in ml_classifiers_dict.items():	
				try: typicality_value.append(value[D_index][i])
				except: print("ERROR : ",key," key, ", D_index,"D_index, ",i , "i") 
			typicality_value_all_samples.append(np.sum(typicality_value))
		#print(D, "D: typicality_value_all_samples : ", typicality_value_all_samples)
		most_typical_index.append(typicality_value_all_samples.index(min(typicality_value_all_samples)))

	print("most_typical_index : ", most_typical_index)
	print("nn most typical : ", [item[most_typical_index[dimens]] for dimens, item in enumerate(ml_classifiers_dict['nn']) ])
	print("nn for 0th sample : ", [item[0] for item in ml_classifiers_dict['nn'] ])

        with open(name+'_most_typical.txt', 'w') as f:
		f.writelines(["%f,\t" % item  for item in most_typical_index])


