from __future__ import print_function
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import time


# Options for mode 'thrust' 
MODE = 'thrust'

label_size = 28



################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
mpl.rc('font', family='serif', size=34, serif="Times New Roman")

#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [r'\boldmath']

mpl.rcParams['legend.fontsize'] = "medium"

mpl.rc('savefig', format ="pdf")

mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['figure.figsize']  = 8, 6
mpl.rcParams['lines.linewidth'] = 3


def binomial_error(l1):
	err_list = []
	for item in l1:
		if item==1. or item==0.: err_list.append(np.sqrt(100./101.*(1.-100./101.)/101.))
		else: err_list.append(np.sqrt(item*(1.-item)/100.))
	return err_list

################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################


#                                                                        E V E N T   S H A P E S   -   T H R U S T   


################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################



if MODE == 'thrust':
        #dimensions              = [2,3,4,5,6,7,8,9,10]
        #dimensions             = [1,2,3,4,5]
	#param_list 		= [0.125,0.130,0.132,0.133,0.134,0.135,0.1365,0.14]
	#param_list             = [0.130,0.132,0.133,0.134,0.135,0.1365,0.14]
	param_list 		= [0.130, 0.132,0.133,0.134,0.1345,0.135,0.1355,0.136,0.1365,0.137,0.1375,0.138,0.139,0.14]

        ml_classifiers          = ['nn','bdt','xgb','svm']
	
	ml_classifiers_colors   = ['blue','black','green']
        ml_classifiers_bin      = 5

        chi2_color              = 'red'
        chi2_splits             = [1,2,3,4,5,6,7,8,9,10]

        #ml_folder_name          = "automatisation_monash_alphaSvalue_low_level/evaluation_monash_lower_level_2files"
        chi2_folder_name        = "event_shapes_thrust"

        ml_file_name            = "{1}_monash_{0}_alphaSvalue_lower_level_chi2scoring_{2}_p_values"
        chi2_file_name          = "event_shapes_thrust_syst_0_01_attempt4__{0}D_chi2_{1}_splits_p_values"
        
	title                   = "Event shapes thrust"
        name                    = "event_shapes_thrust"
	CL 			= 0.95  

        ml_classifiers_dict={}
        chi2_splits_dict={}

        #xwidth = [0.5]*len(param_list)
        xwidth = np.subtract(param_list[1:],param_list[:-1])/2.
	xwidth_left = np.append(xwidth[0] , xwidth)
	xwidth_right = np.append(xwidth,xwidth[-1])
	print("xwidth : ", xwidth)
	fig = plt.figure()
        ax = fig.add_axes([0.2,0.15,0.75,0.8])

	if False:
		for ml_classifier_index, ml_classifier in enumerate(ml_classifiers):
			ml_classifiers_dict[ml_classifier]= []
			for param in param_list:
				p_values = np.loadtxt(os.environ['learningml']+"/GoF/optimisation_and_evaluation/"+ml_folder_name+"/"+ml_classifier+"/"+ml_file_name.format(param,ml_classifier,ml_classifiers_bin)).tolist()
				p_values_in_CL = sum(i < (1-CL) for i in p_values)
				ml_classifiers_dict[ml_classifier].append(p_values_in_CL)
			ml_classifiers_dict[ml_classifier]= np.divide(ml_classifiers_dict[ml_classifier],100.)


		ax.errorbar(param_list,ml_classifiers_dict['nn'], xerr=xwidth, yerr=binomial_error(ml_classifiers_dict['nn']), linestyle='', marker='s', markersize=15, color='green', ecolor='blue', label=r'$ANN$', clip_on=False)
		print("bdt : ", ml_classifiers_dict['bdt'])
		print("xgb : ", ml_classifiers_dict['xgb'])
		ml_classifiers_dict['BDT_best']= [max(item1,item2) for item1, item2 in zip(ml_classifiers_dict['bdt'],ml_classifiers_dict['xgb'])]
		print("BDT : ", ml_classifiers_dict['BDT_best'])
		ax.errorbar(param_list,ml_classifiers_dict['BDT_best'], xerr=xwidth, yerr=binomial_error(ml_classifiers_dict['BDT_best']), linestyle='', marker='o', markersize=15, color='green', ecolor='blue', label=r'$BDT$', clip_on=False)
		ax.errorbar(param_list,ml_classifiers_dict['svm'], xerr=xwidth, yerr=binomial_error(ml_classifiers_dict['svm']),  linestyle='', marker='^', markersize=15, color='green', ecolor='blue', label=r'$SVM$', clip_on=False)



	for chi2_split_index, chi2_split in enumerate(chi2_splits):
		chi2_splits_dict[str(chi2_split)]=[]

        chi2_best = []
        for param in param_list:
                chi2_best_dim = []
                for chi2_split_index, chi2_split in enumerate(chi2_splits):
                        p_values = np.loadtxt(os.environ['learningml']+"/GoF/chi2/"+chi2_folder_name+"/"+chi2_file_name.format(param,chi2_split)).tolist()
                        p_values_in_CL = sum(i < (1-CL) for i in p_values)
                        temp = float(p_values_in_CL) /100.
                        chi2_splits_dict[str(chi2_split)].append(temp)
                        chi2_best_dim.append(temp)
                temp_best = np.max(chi2_best_dim)
                #print(str(dim)+"D chi2_best_dim : ", chi2_best_dim)    
                #print(str(dim)+"D temp_best : ",np.max(temp_best))
                chi2_best.append(temp_best)
                #print("chi2_best : ",chi2_best)

	print("param_list : ",param_list)
	print("chi2_best : ", chi2_best)
	#plt.errorbar(param_list,chi2_best, xerr=(xwidth_left,xwidth_right), yerr=binomial_error(chi2_best), linestyle='', marker='$\chi$', markeredgecolor='none', markersize=15, color='magenta', ecolor='blue', label=r'$\chi^2$', clip_on=False)
	ax.errorbar(param_list,chi2_best, yerr=binomial_error(chi2_best), linestyle='--', marker='$\chi$', markeredgecolor='none', markersize=18, color='black', label=r'$\chi^2$', clip_on=False)
	print("ml_classifiers_dict : ",ml_classifiers_dict)
        print("chi2_best : ", chi2_best)
	ax.plot((0.1365,0.1365),(0.,1.),c="grey",linestyle="--")

        #ax.set_xlim([0.12,0.145])
        ax.set_xlim([0.129,0.1405])
	ax.set_ylim([0.,1.])
        ax.set_xlabel(r"$\alpha_{S}$")
        ax.set_ylabel("Fraction rejected")
        a, b, c  = [0.130,0.133], [0.1365],[0.14]
        ax.set_xticks(a+b+c)
        xx, locs = plt.xticks()
        ll = ['%.3f' % y for y in a] + ['%.4f' % y for y in b] + ['%.3f' % y for y in c]
        plt.xticks(xx, ll)

        #ax.legend(loc='lower left', frameon=False, numpoints=1)
        #fig_name=name+"_alphaSvalue_analysis"
        fig_name= "event_shapes_thrust_analysis"
	fig.savefig(fig_name+".pdf")
        fig.savefig(fig_name+"_"+time.strftime("%b_%d_%Y")+".pdf")
        print("Saved the figure as" , fig_name+".pdf")


