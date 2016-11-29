from __future__ import print_function
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import time


# Options for mode 'Gauss1', 'Gauss1_euclidean' 'Gauss1_noCPV', 'Gauss1_euclidean_noCPV' 
#MODE = 'Gauss1'
MODE = 'Gauss1_noCPV'
#MODE =  'Gauss1_euclidean'

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


#                                                               G A U S S   1  


################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################



if MODE == 'Gauss1':
        dimensions              = [2,3,4,5,6,7,8,9,10]
        #dimensions             = [1,2,3,4,5]

        ml_classifiers          = ['nn','bdt','xgb','svm']
	
	ml_classifiers_colors   = ['green','magenta','cyan']
        ml_classifiers_bin      = 5

        chi2_color              = 'red'
        chi2_splits             = [1,2,3,4,5,6,7,8,9,10]

	#ml_folder_name          = "automatisation_gaussian_same_projection/evaluation_0_95__0_95_not_redefined"
        ml_folder_name          = "automatisation_gaussian_same_projection/evaluation_gauss_final_2files"
        #ml_folder_name          = "automatisation_gaussian_same_projection/evaluation_gauss_final"
	chi2_folder_name        = "gauss"

        ml_file_name            = "{1}_{0}Dgauss__0_95__0_95_CPV_not_redefined_syst_0_01__chi2scoring_{2}_p_values"
        chi2_file_name          = "chi2_gauss__0_95__0_95_CPV_not_redefined_syst_0_01__{0}D_chi2_{1}_splits_p_values"

        title                   = "Gauss 0.95 0.95"
        name                    = "gauss_0_95__0_95_not_redefined_"
	CL 			= 0.95  

        ml_classifiers_dict={}
        chi2_splits_dict={}

	xwidth = [0.5]*len(dimensions)
	fig = plt.figure()
	ax = fig.add_axes([0.2,0.15,0.75,0.8])

        for ml_classifier_index, ml_classifier in enumerate(ml_classifiers):
                ml_classifiers_dict[ml_classifier]= []
                for dim in dimensions:
			p_values = np.loadtxt(os.environ['learningml']+"/GoF/optimisation_and_evaluation/"+ml_folder_name+"/"+ml_classifier+"/"+ml_file_name.format(dim,ml_classifier,ml_classifiers_bin)).tolist()
			p_values_in_CL = sum(i < (1-CL) for i in p_values)
                        ml_classifiers_dict[ml_classifier].append(p_values_in_CL)
		ml_classifiers_dict[ml_classifier]= np.divide(ml_classifiers_dict[ml_classifier],100.)
	

	ax.errorbar(dimensions,ml_classifiers_dict['nn'], yerr=binomial_error(ml_classifiers_dict['nn']), linestyle='-', marker='s', markeredgewidth=0.0, markersize=12, color=ml_classifiers_colors[0], label=r'$ANN$',clip_on=False)
	print("bdt : ", ml_classifiers_dict['bdt'])
	print("xgb : ", ml_classifiers_dict['xgb'])
	ml_classifiers_dict['BDT_best']= [max(item1,item2) for item1, item2 in zip(ml_classifiers_dict['bdt'],ml_classifiers_dict['xgb'])]
	print("BDT : ", ml_classifiers_dict['BDT_best'])
	ax.errorbar(dimensions,ml_classifiers_dict['BDT_best'], yerr=binomial_error(ml_classifiers_dict['BDT_best']), linestyle='-', marker='o', markeredgewidth=0.0, markersize=12, color=ml_classifiers_colors[1], label=r'$BDT$', clip_on=False)
	ax.errorbar(dimensions,ml_classifiers_dict['svm'], yerr=binomial_error(ml_classifiers_dict['svm']),  linestyle='-', marker='^', markeredgewidth=0.0, markersize=12, color=ml_classifiers_colors[2], label=r'$SVM$', clip_on=False)

        for chi2_split_index, chi2_split in enumerate(chi2_splits):
                chi2_splits_dict[str(chi2_split)]=[]
        
	chi2_best = []
        for dim in dimensions:
		chi2_best_dim = []
        	for chi2_split_index, chi2_split in enumerate(chi2_splits):
			p_values = np.loadtxt(os.environ['learningml']+"/GoF/chi2/"+chi2_folder_name+"/"+chi2_file_name.format(dim,chi2_split)).tolist()
			p_values_in_CL = sum(i < (1-CL) for i in p_values)
			temp = float(p_values_in_CL) /100.
	                chi2_splits_dict[str(chi2_split)].append(temp)
			chi2_best_dim.append(temp)
		temp_best = np.max(chi2_best_dim)
		#print(str(dim)+"D chi2_best_dim : ", chi2_best_dim)	
		#print(str(dim)+"D temp_best : ",np.max(temp_best))
		chi2_best.append(temp_best)
		#print("chi2_best : ",chi2_best)

	#ax.errorbar(dimensions,chi2_best, xerr=xwidth, yerr=binomial_error(chi2_best), linestyle='', marker='x', markersize=15, color='magenta', ecolor='blue', label=r'$\chi^2$', clip_on=False)
        ax.errorbar(dimensions,chi2_best, yerr=binomial_error(chi2_best), linestyle='--', marker='$\chi$', markeredgecolor='none', markersize=18, color='black', label=r'$\chi^2$', clip_on=False)
	print("ml_classifiers_dict : ",ml_classifiers_dict)
	print("chi2_best : ", chi2_best)

	ax.set_xlim([1.,11.])
        ax.set_ylim([0.,1.])
        ax.set_xlabel("Number of dimensions")
        ax.set_ylabel("Fraction rejected")
        #ax.legend(loc='best', frameon=False, numpoints=1)
        fig_name=name+"dimensionality_analysis"
	fig_name= "gauss_results"
        fig.savefig(fig_name+".pdf")
        fig.savefig(fig_name+"_"+time.strftime("%b_%d_%Y")+".pdf")
        print("Saved the figure as" , fig_name+".pdf")



################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################


#                                                               G A U S S   1  -  Euclidean


################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################



if MODE == 'Gauss1_euclidean':
        dimensions              = [2,3,4,5,6,7,8,9,10]
        #dimensions             = [1,2,3,4,5]

        ml_classifiers          = ['nn','bdt','xgb','svm']

        ml_classifiers_colors   = ['green','magenta','cyan']
        ml_classifiers_bin      = 5

        chi2_color              = 'red'
        chi2_splits             = [1,2,3,4,5,6,7,8,9,10]

        ml_folder_name          = "automatisation_gaussian_same_projection/evaluation_0_95__0_95_not_redefined_euclidean"
        chi2_folder_name        = "gauss"

        ml_file_name            = "{1}_{0}Dgauss__0_95__0_95_CPV_not_redefined_euclidean_syst_0_01__chi2scoring_{2}_p_values"
        chi2_file_name          = "chi2_gauss_0_95__0_95_CPV_not_redefined_syst_0_01_euclidean__{0}D_chi2_{1}_splits_p_values"

        title                   = "Gauss 0.95 0.95 Euclidean Distance"
        name                    = "gauss_0_95__0_95_not_redefined_euclidean_"
	CL                      = 0.95

        ml_classifiers_dict={}
        chi2_splits_dict={}

        xwidth = [0.5]*len(dimensions)
        fig = plt.figure()
        ax = fig.add_axes([0.2,0.15,0.75,0.8])

	if False:
		for ml_classifier_index, ml_classifier in enumerate(ml_classifiers):
			ml_classifiers_dict[ml_classifier]= []
			for dim in dimensions:
				p_values = np.loadtxt(os.environ['learningml']+"/GoF/optimisation_and_evaluation/"+ml_folder_name+"/"+ml_classifier+"/"+ml_file_name.format(dim,ml_classifier,ml_classifiers_bin)).tolist()
				p_values_in_CL = sum(i < (1-CL) for i in p_values)
				ml_classifiers_dict[ml_classifier].append(p_values_in_CL)
			ml_classifiers_dict[ml_classifier]= np.divide(ml_classifiers_dict[ml_classifier],100.)

		ax.errorbar(dimensions,ml_classifiers_dict['nn'], xerr=xwidth, yerr=binomial_error(ml_classifiers_dict['nn']), fmt='none', label='ANN', ecolor=ml_classifiers_colors[0])
		print("bdt : ", ml_classifiers_dict['bdt'])
		print("xgb : ", ml_classifiers_dict['xgb'])
		ml_classifiers_dict['BDT_best']= [max(item1,item2) for item1, item2 in zip(ml_classifiers_dict['bdt'],ml_classifiers_dict['xgb'])]
		print("BDT : ", ml_classifiers_dict['BDT_best'])
		ax.errorbar(dimensions,ml_classifiers_dict['BDT_best'], xerr=xwidth, yerr=binomial_error(ml_classifiers_dict['BDT_best']), fmt='none', label='BDT', ecolor=ml_classifiers_colors[1])
		ax.errorbar(dimensions,ml_classifiers_dict['svm'], xerr=xwidth, yerr=binomial_error(ml_classifiers_dict['svm']),  fmt='none', label='SVM', ecolor=ml_classifiers_colors[2])

        for chi2_split_index, chi2_split in enumerate(chi2_splits):
                chi2_splits_dict[str(chi2_split)]=[]

        chi2_best = []
        for dim in dimensions:
                chi2_best_dim = []
                for chi2_split_index, chi2_split in enumerate(chi2_splits):
                        p_values = np.loadtxt(os.environ['learningml']+"/GoF/chi2/"+chi2_folder_name+"/"+chi2_file_name.format(dim,chi2_split)).tolist()
                        p_values_in_CL = sum(i < (1-CL) for i in p_values)
                        temp = float(p_values_in_CL) /100.
                        chi2_splits_dict[str(chi2_split)].append(temp)
                        chi2_best_dim.append(temp)
                temp_best = np.max(chi2_best_dim)
                #print(str(dim)+"D chi2_best_dim : ", chi2_best_dim)    
                #print(str(dim)+"D temp_best : ",np.max(temp_best))
                chi2_best.append(temp_best)
                #print("chi2_best : ",chi2_best)

	#ax.errorbar(dimensions,chi2_best, xerr=xwidth, yerr=binomial_error(chi2_best), linestyle='', marker='x', markersize=15, color='magenta', ecolor='blue', label=r'$\chi^2$', clip_on=False)
	ax.errorbar(dimensions,chi2_best, yerr=binomial_error(chi2_best), linestyle='--', marker='$\chi$', markeredgecolor='none', markersize=18, color='black', label=r'$\chi^2$', clip_on=False)
	print("ml_classifiers_dict : ",ml_classifiers_dict)
        print("chi2_best : ", chi2_best)

	ax.set_xlim([1.,11.])
        ax.set_ylim([0.,1.])
        ax.set_xlabel("Number of dimensions")
        ax.set_ylabel("Fraction rejected")
        #ax.legend(loc='best', frameon=False, numpoints=1)
        fig_name=name+"dimensionality_analysis"
	fig_name="gauss_euclidean_results"
        fig.savefig(fig_name+".pdf")
        fig.savefig(fig_name+"_"+time.strftime("%b_%d_%Y")+".pdf")
        print("Saved the figure as" , fig_name+".pdf")




################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################


#                                                               G A U S S   1  - noCPV


################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################



if MODE == 'Gauss1_noCPV':
        dimensions              = [2,3,4,5,6,7,8,9,10]
        #dimensions             = [2,3,4,5,6,7,8]

        ml_classifiers          = ['nn','bdt','xgb','svm']
	
	ml_classifiers_colors   = ['green','magenta','cyan']
	ml_classifiers_bin      = 5

        chi2_color              = 'red'
        chi2_splits             = [1,2,3,4,5,6,7,8,9,10]

        #ml_folder_name          = "automatisation_gaussian_same_projection/evaluation_0_95__0_95_not_redefined"
        #ml_folder_name          = "automatisation_gaussian_same_projection/evaluation_gauss_final_2files"
        ml_folder_name          = "automatisation_gaussian_same_projection/evaluation_gauss_final_2files_check"
	#ml_folder_name          = "automatisation_gaussian_same_projection/evaluation_gauss_final"
        chi2_folder_name        = "gauss"

        ml_file_name            = "{1}_{0}Dgauss__1_0__1_0_noCPV_not_redefined_syst_0_01__chi2scoring_{2}_p_values"
        chi2_file_name          = "chi2_gauss__1_0__1_0_noCPV_not_redefined_syst_0_01__{0}D_chi2_{1}_splits_p_values"

        title                   = "Gauss 1.0 1.0 noCPV"
        name                    = "gauss_1_0__1_0_not_redefined_"
	CL 			= 0.95  

        ml_classifiers_dict={}
        chi2_splits_dict={}

        xwidth = [0.5]*len(dimensions)
        fig = plt.figure()
        ax = fig.add_axes([0.2,0.15,0.75,0.8])

        for ml_classifier_index, ml_classifier in enumerate(ml_classifiers):
                ml_classifiers_dict[ml_classifier]= []
                for dim in dimensions:
                        p_values = np.loadtxt(os.environ['learningml']+"/GoF/optimisation_and_evaluation/"+ml_folder_name+"/"+ml_classifier+"/"+ml_file_name.format(dim,ml_classifier,ml_classifiers_bin)).tolist()
                        p_values_in_CL = sum(i < (1-CL) for i in p_values)
                        ml_classifiers_dict[ml_classifier].append(p_values_in_CL)
                ml_classifiers_dict[ml_classifier]= np.divide(ml_classifiers_dict[ml_classifier],100.)


        ax.errorbar(dimensions,ml_classifiers_dict['nn'], yerr=binomial_error(ml_classifiers_dict['nn']), linestyle='-', marker='s', markeredgecolor='none',markersize=12, color=ml_classifiers_colors[0], label=r'$ANN$', clip_on=False)
        print("bdt : ", ml_classifiers_dict['bdt'])
        print("xgb : ", ml_classifiers_dict['xgb'])
        ml_classifiers_dict['BDT_best']= [max(item1,item2) for item1, item2 in zip(ml_classifiers_dict['bdt'],ml_classifiers_dict['xgb'])]
        print("BDT : ", ml_classifiers_dict['BDT_best'])
        ax.errorbar(dimensions,ml_classifiers_dict['BDT_best'], yerr=binomial_error(ml_classifiers_dict['BDT_best']), linestyle='-', marker='o', markeredgecolor='none', markersize=12, color=ml_classifiers_colors[1], label=r'$BDT$', clip_on=False)
        ax.errorbar(dimensions,ml_classifiers_dict['svm'], yerr=binomial_error(ml_classifiers_dict['svm']),  linestyle='-', marker='^', markeredgecolor='none', markersize=12, color=ml_classifiers_colors[2],label=r'$SVM$', clip_on=False)

        for chi2_split_index, chi2_split in enumerate(chi2_splits):
                chi2_splits_dict[str(chi2_split)]=[]

        chi2_best = []
        for dim in dimensions:
                chi2_best_dim = []
                for chi2_split_index, chi2_split in enumerate(chi2_splits):
                        p_values = np.loadtxt(os.environ['learningml']+"/GoF/chi2/"+chi2_folder_name+"/"+chi2_file_name.format(dim,chi2_split)).tolist()
                        p_values_in_CL = sum(i < (1-CL) for i in p_values)
                        temp = float(p_values_in_CL) /100.
                        chi2_splits_dict[str(chi2_split)].append(temp)
                        chi2_best_dim.append(temp)
                temp_best = np.max(chi2_best_dim)
                #print(str(dim)+"D chi2_best_dim : ", chi2_best_dim)    
                #print(str(dim)+"D temp_best : ",np.max(temp_best))
                chi2_best.append(temp_best)
                #print("chi2_best : ",chi2_best)

        #ax.errorbar(dimensions,chi2_best, xerr=xwidth, yerr=binomial_error(chi2_best), linestyle='', marker='x', markersize=15, color='magenta', ecolor='red', label=r'$\chi^2$')
        ax.errorbar(dimensions,chi2_best, yerr=binomial_error(chi2_best), linestyle='--', marker='$\chi$', markeredgecolor='none', markersize=18, color='black', label=r'$\chi^2$', clip_on=False)
	print("ml_classifiers_dict : ",ml_classifiers_dict)
        print("chi2_best : ", chi2_best)

	ax.plot((1.,11),(0.05,0.05),c="grey",linestyle="--")

        ax.set_xlim([1.,11.])
        ax.set_ylim([0.,1.])
        ax.set_xlabel("Number of dimensions")
        ax.set_ylabel("Fraction rejected")
        ax.legend(loc='best', frameon=False, numpoints=1)
        fig_name=name+"dimensionality_analysis"
	fig_name= "gauss_results_noCPV"
        fig.savefig(fig_name+".pdf")
        fig.savefig(fig_name+"_"+time.strftime("%b_%d_%Y")+".pdf")
        print("Saved the figure as" , fig_name+".pdf")




################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################


#                                                               G A U S S   1  -  Euclidean noCPV


################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################



if MODE == 'Gauss1_euclidean_noCPV':
        dimensions              = [2,3,4,5,6,7,8,9,10]
        #dimensions             = [1,2,3,4,5]

        ml_classifiers          = ['nn','bdt','xgb','svm']

        ml_classifiers_colors   = ['blue','black','green']
        ml_classifiers_bin      = 5

        chi2_color              = 'red'
        chi2_splits             = [1,2,3,4,5,6,7,8,9,10]

        ml_folder_name          = "automatisation_gaussian_same_projection/evaluation_0_95__0_95_not_redefined_euclidean"
        chi2_folder_name        = "gauss"

        ml_file_name            = "{1}_{0}Dgauss__1_0__1_0_noCPV_not_redefined_euclidean_syst_0_01__chi2scoring_{2}_p_values"
        chi2_file_name          = "chi2_gauss__1_0__1_0_noCPV_not_redefined_syst_0_01_euclidean__{0}D_chi2_{1}_splits_p_values"

        title                   = "Gauss 1.0 1.0 Euclidean Distance"
        name                    = "gauss_1_0__1_0_not_redefined_euclidean_"
	CL                      = 0.95

        ml_classifiers_dict={}
        chi2_splits_dict={}

        xwidth = [0.5]*len(dimensions)
        fig = plt.figure()
        ax = fig.add_axes([0.2,0.15,0.75,0.8])

	if False:
		for ml_classifier_index, ml_classifier in enumerate(ml_classifiers):
			ml_classifiers_dict[ml_classifier]= []
			for dim in dimensions:
				p_values = np.loadtxt(os.environ['learningml']+"/GoF/optimisation_and_evaluation/"+ml_folder_name+"/"+ml_classifier+"/"+ml_file_name.format(dim,ml_classifier,ml_classifiers_bin)).tolist()
				p_values_in_CL = sum(i < (1-CL) for i in p_values)
				ml_classifiers_dict[ml_classifier].append(p_values_in_CL)
			ml_classifiers_dict[ml_classifier]= np.divide(ml_classifiers_dict[ml_classifier],100.)

		ax.errorbar(dimensions,ml_classifiers_dict['nn'], xerr=xwidth, yerr=binomial_error(ml_classifiers_dict['nn']), fmt='none', label='ANN', ecolor=ml_classifiers_colors[0], clip_on=False)
		print("bdt : ", ml_classifiers_dict['bdt'])
		print("xgb : ", ml_classifiers_dict['xgb'])
		ml_classifiers_dict['BDT_best']= [max(item1,item2) for item1, item2 in zip(ml_classifiers_dict['bdt'],ml_classifiers_dict['xgb'])]
		print("BDT : ", ml_classifiers_dict['BDT_best'])
		ax.errorbar(dimensions,ml_classifiers_dict['BDT_best'], xerr=xwidth, yerr=binomial_error(ml_classifiers_dict['BDT_best']), fmt='none', label='BDT', ecolor=ml_classifiers_colors[1], clip_on=False)
		ax.errorbar(dimensions,ml_classifiers_dict['svm'], xerr=xwidth, yerr=binomial_error(ml_classifiers_dict['svm']),  fmt='none', label='SVM', ecolor=ml_classifiers_colors[2], clip_on=False)

        for chi2_split_index, chi2_split in enumerate(chi2_splits):
                chi2_splits_dict[str(chi2_split)]=[]

        chi2_best = []
        for dim in dimensions:
                chi2_best_dim = []
                for chi2_split_index, chi2_split in enumerate(chi2_splits):
                        p_values = np.loadtxt(os.environ['learningml']+"/GoF/chi2/"+chi2_folder_name+"/"+chi2_file_name.format(dim,chi2_split)).tolist()
                        p_values_in_CL = sum(i < (1-CL) for i in p_values)
                        temp = float(p_values_in_CL) /100.
                        chi2_splits_dict[str(chi2_split)].append(temp)
                        chi2_best_dim.append(temp)
                temp_best = np.max(chi2_best_dim)
                #print(str(dim)+"D chi2_best_dim : ", chi2_best_dim)    
                #print(str(dim)+"D temp_best : ",np.max(temp_best))
                chi2_best.append(temp_best)
                #print("chi2_best : ",chi2_best)

	#ax.errorbar(dimensions,chi2_best, xerr=xwidth, yerr=binomial_error(chi2_best), linestyle='', marker='x', markersize=15, color='magenta', ecolor='red', label=r'$\chi^2$')
        ax.errorbar(dimensions,chi2_best, xerr=xwidth, yerr=binomial_error(chi2_best), linestyle='--', marker='$\chi$', markeredgecolor='none', markersize=15, color='black', ecolor='blue', label=r'$\chi^2$', clip_on=False)
	print("ml_classifiers_dict : ",ml_classifiers_dict)
        print("chi2_best : ", chi2_best)

	ax.set_xlim([1.,11.])
        ax.set_ylim([0.,1.])
        ax.set_xlabel("Number of dimensions")
        ax.set_ylabel("Fraction rejected")
        ax.legend(loc='best', frameon=False, numpoints=1)
        fig_name=name+"dimensionality_analysis"
        fig.savefig(fig_name+".pdf")
        fig.savefig(fig_name+"_"+time.strftime("%b_%d_%Y")+".pdf")
        print("Saved the figure as" , fig_name+".pdf")




