from __future__ import print_function
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import time


# Options for mode 'Gauss1', 'Gauss1_euclidean' 
MODE = 'Gauss1'
#MODE =  'Gauss1_euclidean'

label_size = 28



################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
mpl.rc('font', family='serif', size=34, serif="Times New Roman")

#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [r'\boldmath']

mpl.rcParams['legend.fontsize'] = "medium"

mpl.rc('savefig', format ="pdf", bbox='tight', pad_inches= 0.1)

mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['figure.figsize']  = 8, 6
mpl.rcParams['lines.linewidth'] = 3


################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################


#                                                               G A U S S   1  -  Euclidean


################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################



if MODE == 'Gauss1':
        dimensions              = [2,3,4,5,6,7,8,9,10]
        #dimensions             = [1,2,3,4,5]

        ml_classifiers_colors   = ['blue','black','green','slategrey']
        #ml_classifiers          = ['nn','bdt','xgb','svm']
        ml_classifiers          = ['nn','bdt','svm']
	ml_classifiers_labels   = ['ANN','BDT','SVM']
	#ml_classifiers          = ['nn']
        ml_classifiers_bins     = [5,5,5,5]

        chi2_colors             = ['red','maroon','orange']
        chi2_splits             = [5]

        ml_folder_name          = "automatisation_gaussian_same_projection/evaluation_0_95__0_95_not_redefined"
        chi2_folder_name        = "gauss"

        ml_file_name            = "{1}_{0}Dgauss__0_95__0_95_CPV_not_redefined_syst_0_01__chi2scoring_{2}_p_values_1_2_3_std_dev.txt"
        chi2_file_name          = "chi2_gauss__0_95__0_95_CPV_not_redefined_syst_0_01__{0}D_chi2_{1}_splits_p_values_1_2_3_std_dev.txt"

        title                   = "Gauss 0.95 0.95"
        name                    = "gauss_0_95__0_95_not_redefined_"

        ml_classifiers_dict={}
        chi2_splits_dict={}

        fig = plt.figure()
        ax  = fig.add_subplot(1,1,1)

        for ml_classifier_index, ml_classifier in enumerate(ml_classifiers):
                ml_classifiers_dict[ml_classifier+str(ml_classifiers_bins[ml_classifier_index])]= []
                for dim in dimensions:
                        ml_classifiers_dict[ml_classifier+str(ml_classifiers_bins[ml_classifier_index])].append(np.loadtxt(os.environ['learningml']+"/GoF/optimisation_and_evaluation/"+ml_folder_name+"/"+ml_classifier+"/"+ml_file_name.format(dim,ml_classifier,ml_classifiers_bins[ml_classifier_index]))[1])
                ax.plot(dimensions, ml_classifiers_dict[ml_classifier+str(ml_classifiers_bins[ml_classifier_index])], label=ml_classifiers_labels[ml_classifier_index], color=ml_classifiers_colors[ml_classifier_index])

        for chi2_split_index, chi2_split in enumerate(chi2_splits):
                chi2_splits_dict[str(chi2_split)]=[]
                for dim in dimensions:
                        chi2_splits_dict[str(chi2_split)].append(np.loadtxt(os.environ['learningml']+"/GoF/chi2/"+chi2_folder_name+"/"+chi2_file_name.format(dim,chi2_split))[1])
                #ax.plot(dimensions,chi2_splits_dict[str(chi2_split)], label="chi^2 "+str(chi2_split)+"split", color= chi2_colors[chi2_split_index])
                #ax.plot(dimensions,chi2_splits_dict[str(chi2_split)], label=r"$\chi^2$ "+str(chi2_split)+"split", color= chi2_colors[chi2_split_index])
		ax.plot(dimensions,chi2_splits_dict[str(chi2_split)], label=r"$\chi^2$", color= chi2_colors[chi2_split_index])	
        print("ml_classifiers_dict : ",ml_classifiers_dict)
        print(" chi2_splits_dict : ", chi2_splits_dict)

        plt.ylim([-5,120])
        ax.set_xlabel("Number of dimensions")
        ax.set_ylabel("Number of samples")
        ax.legend(loc='best')
        fig_name=name+"dimensionality_analysis"
        fig.savefig(fig_name+".pdf")
        fig.savefig(fig_name+"_"+time.strftime("%b_%d_%Y")+".pdf")
        print("Saved the figure as" , fig_name+".pdf")


################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################


#								G A U S S   1  -  Euclidean


################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################




if MODE == 'Gauss1_euclidean':
        dimensions		= [2,3,4,5,6,7,8,9,10]
	#dimensions		= [1,2,3,4,5]

	ml_classifiers_colors	= ['blue','black','green','slategrey']
	#ml_classifiers          = ['nn','bdt','xgb','svm']
	ml_classifiers          = ['nn','bdt','svm']
	#ml_classifiers          = ['nn']
	ml_classifiers_labels	= ['ANN', 'BDT', 'SVM']
	ml_classifiers_bins 	= [5,5,5,5]

	chi2_colors		= ['red','maroon','orange']
	chi2_splits		= [5]
	
	ml_folder_name		= "automatisation_gaussian_same_projection/evaluation_0_95__0_95_not_redefined_euclidean"
	chi2_folder_name	= "gauss"

	ml_file_name		= "{1}_{0}Dgauss__0_95__0_95_CPV_not_redefined_euclidean_syst_0_01__chi2scoring_{2}_p_values_1_2_3_std_dev.txt"
	chi2_file_name		= "chi2_gauss_0_95__0_95_CPV_not_redefined_syst_0_01_euclidean__{0}D_chi2_{1}_splits_p_values_1_2_3_std_dev.txt"

	title			= "Gauss 0.95 0.95 Euclidean Distance"
	name			= "gauss_0_95__0_95_not_redefined_euclidean_"

	ml_classifiers_dict={}
	chi2_splits_dict={}

	fig = plt.figure()
        ax  = fig.add_subplot(1,1,1)

	for ml_classifier_index, ml_classifier in enumerate(ml_classifiers):
		ml_classifiers_dict[ml_classifier+str(ml_classifiers_bins[ml_classifier_index])]= []
		for dim in dimensions:
			ml_classifiers_dict[ml_classifier+str(ml_classifiers_bins[ml_classifier_index])].append(np.loadtxt(os.environ['learningml']+"/GoF/optimisation_and_evaluation/"+ml_folder_name+"/"+ml_classifier+"/"+ml_file_name.format(dim,ml_classifier,ml_classifiers_bins[ml_classifier_index]))[1])
		ax.plot(dimensions, ml_classifiers_dict[ml_classifier+str(ml_classifiers_bins[ml_classifier_index])], label=ml_classifiers_labels[ml_classifier_index] , color=ml_classifiers_colors[ml_classifier_index])	
	
	for chi2_split_index, chi2_split in enumerate(chi2_splits):
		chi2_splits_dict[str(chi2_split)]=[]
		for dim in dimensions:
			chi2_splits_dict[str(chi2_split)].append(np.loadtxt(os.environ['learningml']+"/GoF/chi2/"+chi2_folder_name+"/"+chi2_file_name.format(dim,chi2_split))[1])
		#ax.plot(dimensions,chi2_splits_dict[str(chi2_split)], label="chi2_"+str(chi2_split)+"split", color= chi2_colors[chi2_split_index])
		#ax.plot(dimensions,chi2_splits_dict[str(chi2_split)], label=r"$\chi^2$ "+str(chi2_split)+"split", color= chi2_colors[chi2_split_index])
		ax.plot(dimensions,chi2_splits_dict[str(chi2_split)], label=r"$\chi^2$", color= chi2_colors[chi2_split_index])
	print("ml_classifiers_dict : ",ml_classifiers_dict)
	print(" chi2_splits_dict : ", chi2_splits_dict)

        plt.ylim([-5,120])
        ax.set_xlabel("Number of dimensions")
        ax.set_ylabel("Number of samples")
        ax.legend(loc='best')
        fig_name=name+"dimensionality_analysis"
        fig.savefig(fig_name+".pdf")
        fig.savefig(fig_name+"_"+time.strftime("%b_%d_%Y")+".pdf")
        print("Saved the figure as" , fig_name+".pdf")

################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################


#                                                                               N O     C P V


################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
