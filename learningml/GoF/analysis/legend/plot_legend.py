from __future__ import print_function
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import time


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

def binomial_error(l1):
        err_list = []
        for item in l1:
                if item==1. or item==0.: err_list.append(np.sqrt(100./101.*(1.-100./101.)/101.))
                else: err_list.append(np.sqrt(item*(1.-item)/100.))
        return err_list
################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
dimensions= [1.,2.]
ml_classifiers_dict={'nn':[2.,3.], 'svm':[3.,4.], 'BDT_best':[4.,5.]}
xwidth=[0.1,0.1]
chi2_best=[6.,7.]

fig = plt.figure()
figlegend = plt.figure(figsize=(8,6))
ax = fig.add_axes([0.2,0.15,0.75,0.8])

l1=ax.errorbar(dimensions,ml_classifiers_dict['nn'], xerr=xwidth, yerr=ml_classifiers_dict['nn'], linestyle='', marker='s', markersize=15, color='green', label=r'$ANN$')
l2=ax.errorbar(dimensions,ml_classifiers_dict['BDT_best'], xerr=xwidth, yerr=ml_classifiers_dict['BDT_best'], linestyle='', marker='o', markersize=15, color='green', label=r'$BDT$')
l3=ax.errorbar(dimensions,ml_classifiers_dict['svm'], xerr=xwidth, yerr=ml_classifiers_dict['svm'],  linestyle='', marker='^', markersize=15, color='green', label=r'$SVM$')
l4=ax.errorbar(dimensions,chi2_best, xerr=xwidth, yerr=chi2_best, linestyle='', marker='x', markersize=15, color='magenta', label=r'$\chi^2$')

l = [l1,l2,l3,l4]

figlegend.legend(l,('ANN','BDT','SVM', r'$\chi^2$'),frameon=False, numpoints=1)
figlegend.savefig('legend.pdf')
