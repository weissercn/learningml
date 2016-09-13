import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import os
import time

# Options for mode 'single_p_value', 'ensemble' 
MODE= 'single_p_value'

################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################


#										SINGLE P VALUE


################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################




if MODE == 'single_p_value':
        dimensions		= [2,3,4,5,6,7,8,9,10]
	#dimensions		= [1,2,3]
	
	cmap= cm.get_cmap('jet')
	
	#ml_classifiers_bins     = [2,3,5,7,8,10,15,20,30,50,100,1000]
	ml_classifiers_bins 	= [5,8,10]
	ml_classifiers_colors_number = np.linspace(0.,1.,num=len(ml_classifiers_bins))
        ml_classifiers_colors   = cmap(ml_classifiers_colors_number).tolist()
        ml_classifiers          = ['nn']*len(ml_classifiers_bins)


	chi2_colors		= ['red','maroon','orange']
	chi2_splits		= [1,5,10]
	
	ml_folder_name		= "automatisation_gaussian_same_projection/evaluation_0_95_chi2scoringopt"
	chi2_folder_name	= "gauss"

	ml_file_name		= "{1}_{0}Dgauss__1_0__0_95_CPV_chi2scoringopt_syst_0_01__chi2scoring_{2}_p_values"
	#chi2_file_name		= "chi2_gauss__1_0__0_9_CPV_syst_0_01__{0}D_chi2_{1}_splits_p_values"

	title			= "Gauss 0.95 nn"
	name			= "gauss__1_0__0_95_nn"

	ml_classifiers_dict={}
	chi2_splits_dict={}

	fig = plt.figure()
        ax  = fig.add_subplot(1,1,1)
	ax.set_yscale("log", nonposy='clip')

	for ml_classifier_index, ml_classifier in enumerate(ml_classifiers):
		ml_classifiers_dict[ml_classifier+str(ml_classifiers_bins[ml_classifier_index])]= []
		for dim in dimensions:
			ml_classifiers_dict[ml_classifier+str(ml_classifiers_bins[ml_classifier_index])].append(np.loadtxt(os.environ['learningml']+"/GoF/optimisation_and_evaluation/"+ml_folder_name+"/"+ml_classifier+"/"+ml_file_name.format(dim,ml_classifier,ml_classifiers_bins[ml_classifier_index])))
		print("ml_classifiers_dict : ",ml_classifiers_dict)
		ax.plot(dimensions, ml_classifiers_dict[ml_classifier+str(ml_classifiers_bins[ml_classifier_index])], label=ml_classifier+ str(ml_classifiers_bins[ml_classifier_index])+"bins" , color=ml_classifiers_colors[ml_classifier_index])	

	if False:
		for chi2_split_index, chi2_split in enumerate(chi2_splits):
			chi2_splits_dict[str(chi2_split)]=[]
			for dim in dimensions:
				chi2_splits_dict[str(chi2_split)].append(np.loadtxt(os.environ['learningml']+"/GoF/chi2/"+chi2_folder_name+"/"+chi2_file_name.format(dim,chi2_split)))
			ax.plot(dimensions,chi2_splits_dict[str(chi2_split)], label="chi2_"+str(chi2_split)+"split", color= chi2_colors[chi2_split_index])

        #plt.ylim([-5,120])
        ax.set_xlabel("Number of dimensions")
        ax.set_ylabel("Number of samples")
        ax.set_title("Dimensionality analysis "+title)
        # Shrink current axis by 20%
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

        # Put a legend to the right of the current axis
        #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))        

        ax.legend(loc='best')
        fig_name=name+"_dimensionality_analysis"
        fig.savefig(fig_name)
        fig.savefig(fig_name+"_"+time.strftime("%b_%d_%Y"))
        print("Saved the figure as" , fig_name+".png")



################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################


#                                                                               ENSEMBLE


################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

if MODE == 'ensemble':
        dimensions              = [1,2,3,4,5,6,7,8,9,10]
        #dimensions             = [1,2,3]

        cmap= cm.get_cmap('jet')

        ml_classifiers_bins     = [5,8,10]
        ml_classifiers_colors_number = np.linspace(0.,1.,num=len(ml_classifiers_bins))
        ml_classifiers_colors   = cmap(ml_classifiers_colors_number).tolist()
        ml_classifiers          = ['nn']*len(ml_classifiers_bins)


        chi2_colors             = ['red','maroon','orange']
        chi2_splits             = [1,5,10]

        ml_folder_name          = "automatisation_gaussian_same_projection/evaluation_0_9_chi2scoringopt"
        chi2_folder_name        = "gauss"

        ml_file_name            = "{1}_{0}Dgauss__1_0__0_95_CPV_chi2scoringopt_syst_0_01__chi2scoring_{2}_p_values_1_2_3_std_dev.txt"
        #chi2_file_name         = "chi2_gauss__1_0__0_9_CPV_syst_0_01__{0}D_chi2_{1}_splits_p_values_1_2_3_std_dev.txt"

        title                   = "Gauss 0.95 nn"
        name                    = "gauss__1_0__0_95_nn"

        ml_classifiers_dict={}
        chi2_splits_dict={}

        fig = plt.figure()
        ax  = fig.add_subplot(1,1,1)

        for ml_classifier_index, ml_classifier in enumerate(ml_classifiers):
                ml_classifiers_dict[ml_classifier+str(ml_classifiers_bins[ml_classifier_index])]= []
                for dim in dimensions:
                        ml_classifiers_dict[ml_classifier+str(ml_classifiers_bins[ml_classifier_index])].append(np.loadtxt(os.environ['learningml']+"/GoF/optimisation_and_evaluation/"+ml_folder_name+"/"+ml_classifier+"/"+ml_file_name.format(dim,ml_classifier,ml_classifiers_bins[ml_classifier_index]))[1])
                print("ml_classifiers_dict : ",ml_classifiers_dict)
                ax.plot(dimensions, ml_classifiers_dict[ml_classifier+str(ml_classifiers_bins[ml_classifier_index])], label=ml_classifier+ str(ml_classifiers_bins[ml_classifier_index])+"bins" , color=ml_classifiers_colors[ml_classifier_index])

        if False:
                for chi2_split_index, chi2_split in enumerate(chi2_splits):
                        chi2_splits_dict[str(chi2_split)]=[]
                        for dim in dimensions:
                                chi2_splits_dict[str(chi2_split)].append(np.loadtxt(os.environ['learningml']+"/GoF/chi2/"+chi2_folder_name+"/"+chi2_file_name.format(dim,chi2_split)))
                        ax.plot(dimensions,chi2_splits_dict[str(chi2_split)], label="chi2_"+str(chi2_split)+"split", color= chi2_colors[chi2_split_index])

        plt.ylim([-5,120])
        ax.set_xlabel("Number of dimensions")
        ax.set_ylabel("Number of samples")
        ax.set_title("Dimensionality analysis "+title)
        # Shrink current axis by 20%
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

        # Put a legend to the right of the current axis
        #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        #ax.legend(loc='best')
        fig_name=name+"_dimensionality_analysis"
        fig.savefig(fig_name)
        fig.savefig(fig_name+"_"+time.strftime("%b_%d_%Y"))
        print("Saved the figure as" , fig_name+".png")




