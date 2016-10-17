from __future__ import print_function
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import time
import csv

# Options for mode 'sin' 
MODE = 'sin'

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


################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################


#                                                               S I N  -    M L - R E S P O N S E 


################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
SCALING = False


if MODE == 'sin':
        dimensions             = [10]

        ml_classifiers_colors   = ['blue','black','green','slategrey']
        #ml_classifiers          = ['nn','bdt','xgb','svm']
        ml_classifiers          = ['nn','bdt','svm']
	#ml_classifiers          = ['nn']
	col_f1                  = ['darkblue', 'blue', 'deepskyblue']
        col_f2                  = ['maroon' , 'red' , 'deeppink' ]
        marker_f1               = 's'
        marker_f2               = 'o'

	ml_classifiers_labels   = ['ANN','BDT','SVM']
	#ml_classifiers          = ['nn']
        ml_classifiers_bins     = [5,5,5,5]


        ml_folder_name          = "automatisation_sin/plot10000_2files"

        chi2_folder_name        = "sin"

        ml_file_name            = "{1}_{0}Dsin_5_6_10000_CPV_syst_0_01__chi2scoring_{2}_bin_definitions_1D_features_boundaries.txt"

        title                   = "Sin 5 6"
        name                    = "sin_5_6_"
        CL                      = 0.95

        ml_classifiers_dict={}
        chi2_splits_dict={}


	for dim in dimensions:
		fig = plt.figure()
		ax = fig.add_axes([0.2,0.15,0.75,0.8])
		p1s, p2s = [], []
		for ml_classifier_index, ml_classifier in enumerate(ml_classifiers):
			fname = os.environ['learningml']+"/GoF/optimisation_and_evaluation/"+ml_folder_name+"/"+ml_classifier+"/"+ml_file_name.format(dim,ml_classifier,ml_classifiers_bins[ml_classifier_index])
			print("fname : ", fname)
			print("ml_classifier : ", ml_classifier)
			with open(fname) as f:
				features_0 = f.readline()
				features_1 = f.readline()
				bin_boundaries = f.readline()
				
				
				#print("features_0 : ", features_0)
				features_0 = np.array(map(float,features_0[1:-3].split("]\t[")))
				features_1 = np.array(map(float,features_1[1:-3].split("]\t[")))
				bin_boundaries = np.array(map(float,bin_boundaries.split("\t")))
				no_entries = features_0.shape[0]
				#print("features_0 : ", features_0)
				#print("bin_boundaries : ",bin_boundaries)
				print("features_0.shape() : ",features_0.shape)
				print("features_1.shape() : ",features_1.shape)

			hist0, hist0_edges = np.histogram(features_0, bins=bin_boundaries)
			hist1, hist1_edges = np.histogram(features_1, bins=bin_boundaries)
			print("hist0 : ",hist0)
			print("hist1 : ",hist1)
			bin_middle = (hist0_edges[1:] + hist0_edges[:-1]) / 2
			xwidths = (hist0_edges[1:]-hist0_edges[:-1])/2.
			no_bins = hist0.shape[0]

			hist0_sqrt = np.sqrt(hist0)
			hist1_sqrt = np.sqrt(hist1)

			if SCALING:
				scaling0 = float(xwidths.shape[0])/sum(hist0)
				scaling1 = float(xwidths.shape[0])/sum(hist1)

				hist0 = np.multiply(hist0,scaling0)
				hist1 = np.multiply(hist1,scaling1)
				hist0_sqrt = np.multiply(hist0_sqrt,scaling0)
				hist1_sqrt = np.multiply(hist1_sqrt,scaling1)
				ax.plot((0.,1.),(1.,1.),c="grey",linestyle="--")
			else:	ax.plot((0.,1.),(10000./len(bin_middle),10000./len(bin_middle)),c="grey",linestyle="--")

			#ax.plot((0., 1.), (float(no_entries)/float(no_bins), float(no_entries)/float(no_bins)), c="grey", linestyle='--')
			#ax.errorbar(bin_middle, hist0, xerr=xwidths, yerr=hist0_sqrt, linestyle='', marker=markers[ml_classifier_index], markersize=15, color='green', ecolor='blue', label=ml_classifiers_labels[ml_classifier_index])
			#ax.errorbar(bin_middle, hist1, xerr=xwidths, yerr=hist1_sqrt, linestyle='', marker=markers[ml_classifier_index], markersize=15, color='green', ecolor='red')
                        p1 = ax.errorbar(bin_middle, hist0, xerr=xwidths, yerr=hist0_sqrt, linestyle='', marker=marker_f1, markersize=10, markeredgewidth=0.0, color=col_f1[ml_classifier_index], ecolor=col_f1[ml_classifier_index], label=ml_classifiers_labels[ml_classifier_index])
                        p2 = ax.errorbar(bin_middle, hist1, xerr=xwidths, yerr=hist1_sqrt, linestyle='', marker=marker_f2, markersize=10, markeredgewidth=0.0, color=col_f2[ml_classifier_index], ecolor=col_f2[ml_classifier_index])
                        p1s.append(p1)
                        p2s.append(p2)

		#ax.text(0.5, 0.1,'{}D'.format(dim), ha='center', va='center', transform=ax.transAxes)
		ax.text(0.40, 0.95,'d = {}'.format(dim), horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
		ax.set_xlim(0.,1.)
		#ax.set_ylim(1000,3500)
		ax.set_ylim(1500,2500)
		#ax.set_ylim(0.8,1.2)
		#plt.gca().set_ylim(bottom=0)
		ax.set_xlabel("ML response")
		ax.set_ylabel("normalised entries")
		#plt.title(title + " bin definitions" )
		#ax.legend(loc='upper center',frameon=False, numpoints=1)

		if dim ==2:
                        ax.legend(loc=(0.,0.),frameon=False, numpoints=1, ncol=2)
                        #ax.errorbar(0.0,1500,xerr=0.035, yerr=35, linestyle='--', marker=marker_f2, markersize=10, markeredgewidth=0.0, color=col_f2[1],ecolor=col_f2[1])
                        ax.errorbar(0.2,1734,xerr=0.04, yerr=50, linestyle='--', marker=marker_f2, markersize=10, markeredgewidth=0.0, color=col_f2[0],ecolor=col_f2[0])
                        ax.errorbar(0.2,1595,xerr=0.04, yerr=50, linestyle='--', marker=marker_f2, markersize=10, markeredgewidth=0.0, color=col_f2[1],ecolor=col_f2[1])
                        ax.errorbar(0.748,1734,xerr=0.04, yerr=50, linestyle='--', marker=marker_f2, markersize=10, markeredgewidth=0.0, color=col_f2[2],ecolor=col_f2[2])


		#fig.savefig(name+"_{}D_ML_response.pdf".format(dim))
		fig.savefig("sin_MLresponse_{}D.pdf".format(dim))
		plt.close(fig)
		print("The plot {}_{}D_ML_response.pdf has been made\n\n".format(name,dim))

			



