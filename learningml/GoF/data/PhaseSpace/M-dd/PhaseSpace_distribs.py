from __future__ import print_function
import ROOT
from ROOT import TGenPhaseSpace, TLorentzVector, TH1D, TH2F, TCanvas
import array
import matplotlib.pyplot as plt
import random 
import numpy.random
import pandas as pd
import numpy as np

def generateM_dd(p_signal, n_files=100, n_entries=10000, PLOT=False):
	mMs = 1.0
	mMs_resolution = 0.01
	mMlow = 0.5
	mMhigh = 1.5
	masses = array.array('d', [0.1,0.1])
	features = ['p1x', 'p1y', 'p1z', 'E1', 'p2x', 'p2y', 'p2z', 'E2']

	for n_file in range(n_files+1):
		if n_file%10==0: print("n_file : ",n_file)
		mass, p1x, p1y, p1z, E1, p2x, p2y, p2z, E2 = [], [], [], [], [], [], [], [], []
		for n in range(n_entries):

			choice = random.random()
			if choice < p_signal: mM = numpy.random.normal(loc=mMs,scale=mMs_resolution)
			else: mM = numpy.random.uniform(mMlow,mMhigh)

			MotherMom = TLorentzVector(0.0, 0.0, 0.0, mM)

			event = TGenPhaseSpace()
			event.SetDecay(MotherMom, 2, masses)

			while True:
				weight = event.Generate()
				weightmax = event.GetWtMax()
				assert (0. < weight/weightmax) & (weight/weightmax < 1.)
				if random.random() < weight/(weightmax) : break	
		
			pd1 = event.GetDecay(0)
			pd2 = event.GetDecay(1)
		
			p1x.append(pd1.Px())
			p1y.append(pd1.Py())
			p1z.append(pd1.Pz())
			E1.append(pd1.E())

			p2x.append(pd2.Px())
			p2y.append(pd2.Py())
			p2z.append(pd2.Pz())
			E2.append(pd2.E())
			mass.append(mM)


                if PLOT:
                        plt.hist(mass)
                        plt.xlabel("mass")
                        plt.savefig("mass_bs_p_signal_{}.png".format(p_signal))
                        plt.clf()

                        plt.hist(p1x)
                        plt.xlabel("p1x")
                        plt.savefig("p1x_bs_p_signal_{}.png".format(p_signal))
                        plt.clf()
                        plt.hist(p1y)
                        plt.xlabel("p1y")
                        plt.savefig("p1y_bs_p_signal_{}.png".format(p_signal))
                        plt.clf()
                        plt.hist(p1z)
                        plt.xlabel("p1z")
                        plt.savefig("p1z_bs_p_signal_{}.png".format(p_signal))
                        plt.clf()
                        plt.hist(E1)
                        plt.xlabel("E1")
                        plt.savefig("E1_bs_p_signal_{}.png".format(p_signal))
                        plt.clf()


                        plt.hist(p2x)
                        plt.xlabel("p2x")
                        plt.savefig("p2x_bs_p_signal_{}.png".format(p_signal))
                        plt.clf()
                        plt.hist(p2y)
                        plt.xlabel("p2y")
                        plt.savefig("p2y_bs_p_signal_{}.png".format(p_signal))
                        plt.clf()
                        plt.hist(p2z)
                        plt.xlabel("p2z")
                        plt.savefig("p2z_bs_p_signal_{}.png".format(p_signal))
                        plt.clf()
                        plt.hist(E2)
                        plt.xlabel("E2")
                        plt.savefig("E2_bs_p_signal_{}.png".format(p_signal))
                        plt.clf()



		df = pd.DataFrame(np.array([p1x, p1y, p1z, E1, p2x, p2y, p2z, E2]).transpose(), columns =features)
		df2= pd.DataFrame(np.array([mass]).transpose(), columns = ['mM'])
		#print('df : ', df) 
		if n_file==n_files: n_file = "optimisation_0" 
		#df.to_csv("../data/M-dd/M-dd_{}_res_{}_{}.txt".format(p_signal, mMs_resolution, n_file), sep="\t", header = None, index=False)
		df2.to_csv("../data/M-dd/M-dd_{}_res_{}_1Dmass_{}.txt".format(p_signal, mMs_resolution, n_file), sep="\t", header = None, index=False)


#param_list = [0.1, 0.08, 0.06, 0.04, 0.02]
#param_list = [0.01,0.03,0.05,0.07,0.09]
#param_list = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
#param_noCPV = 0.0

#for p in param_list:
#	generateM_dd(p)

#generateM_dd(param_noCPV, 200)


# For Plotting
if True:
        generateM_dd(1.0, 1,10000,True)
        generateM_dd(0.0, 1,10000,True)
        generateM_dd(0.5, 1,10000,True)
        generateM_dd(0.05, 1,10000,True)

