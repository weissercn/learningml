from __future__ import print_function
import ROOT
from ROOT import TGenPhaseSpace, TLorentzVector, TH1D, TH2F, TCanvas
import array
import matplotlib.pyplot as plt
import random 
import numpy.random
import pandas as pd
import numpy as np
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

def generateM_dd(p_signal, n_files=100, n_entries=10000, PLOT=False):
	mMs = 1.0
	mMs_resolution = 0.01
	mMlow = 0.5
	mMhigh = 1.5
	masses = [0.1,0.1]
	features = ['p1x', 'p1y', 'p1z', 'E1', 'p2x', 'p2y', 'p2z', 'E2']
	#distrib  = 'uniform'
	distrib  = 'cos2theta'

	for n_file in range(n_files+1):
		if n_file%10==0: print("n_file : ",n_file)
		p1x, p1y, p1z, E1, p2x, p2y, p2z, E2, ltheta, lphi = [], [], [], [], [], [], [], [], [], []
		xx,yy,zz = [], [], []
		for n in range(n_entries):

			choice = random.random()
			if choice < p_signal: mM = numpy.random.normal(loc=mMs,scale=mMs_resolution)
			else: mM = numpy.random.uniform(mMlow,mMhigh)

			#random sampling on sphere http://mathworld.wolfram.com/SpherePointPicking.html
			u, v = np.random.rand(), np.random.rand()
			phi = 2.* np.pi* u
			
			theta = np.arccos(2.*v-1.)			
			
			if (distrib == 'cos2theta') & (choice <p_signal):  theta = np.arccos(random.choice([1., -1.])*np.sqrt(v))

			# sphere areas are uniformly distributed

			p = np.sqrt(masses[0]**4+masses[1]**4-2*mM**2*(masses[0]**2+masses[1]**2) - 2 *masses[0]**2 * masses[1]**2+ mM**4) /(2.*mM)

			if False:
				if (distrib == "cos2theta") & (choice < p_signal):
					while True:
						weight = (np.cos(theta))**2
						weightmax = 1.
						assert (0. < weight/weightmax) & (weight/weightmax < 1.)
						if random.random() < weight/(weightmax) : break	
						v = random.random()
						theta = np.arccos(2.*v-1.) 
		

			theta1 = theta
			phi1   = phi
			theta2 = np.pi - theta
			phi2   = np.pi + phi
	
			ltheta.append(theta)
			lphi.append(phi)	

			xx.append(np.sin(theta1)*np.cos(phi1))
			yy.append(np.sin(theta1)*np.sin(phi1))
			zz.append(np.cos(theta1))

			p1x.append(p*np.sin(theta1)*np.cos(phi1))
			p1y.append(p*np.sin(theta1)*np.sin(phi1))
			p1z.append(p*np.cos(theta1))
			E1.append(np.sqrt(p**2+masses[0]**2))

			p2x.append(p*np.sin(theta2)*np.cos(phi2))
			p2y.append(p*np.sin(theta2)*np.sin(phi2))
			p2z.append(p*np.cos(theta2))
			E2.append(np.sqrt(p**2+masses[1]**2))

		#print("ltheta : ", ltheta)
		#print("np.cos(ltheta) : ", np.cos(ltheta))

		if PLOT:
			plt.hist(lphi)
			plt.xlabel("phi")
			plt.savefig("phi_bs_{}_p_signal_{}.png".format(distrib,p_signal))
			plt.clf()
                        plt.hist(np.cos(ltheta))
                        plt.xlabel("costheta")
                        plt.savefig("costheta_bs_{}_p_signal_{}.png".format(distrib,p_signal))
                        plt.clf()
			plt.hist(np.cos(ltheta)**2)
                        plt.xlabel("cos2theta")
                        plt.savefig("cos2theta_bs_{}_p_signal_{}.png".format(distrib,p_signal))
			plt.clf()

		if False:
			# Create a sphere
			r = 1
			pi = np.pi
			cos = np.cos
			sin = np.sin
			phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
			x = r*sin(phi)*cos(theta)
			y = r*sin(phi)*sin(theta)
			z = r*cos(phi)

			#Set colours and render
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')

			ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)

			ax.scatter(xx,yy,zz,color="k",s=20)

			ax.set_xlim([-1,1])
			ax.set_ylim([-1,1])
			ax.set_zlim([-1,1])
			ax.set_aspect("equal")
			plt.tight_layout()
			plt.savefig("sphere_bs_{}_p_signal_{}.png".format(distrib,p_signal))


		df = pd.DataFrame(np.array([p1x, p1y, p1z, E1, p2x, p2y, p2z, E2]).transpose(), columns =features)
		#print('df : ', df) 
		if n_file==n_files: n_file = "optimisation_0" 
		df.to_csv("../data/M-dd/M-dd_{}_{}_res_{}_{}.txt".format(distrib,p_signal, mMs_resolution, n_file), sep="\t", header = None, index=False)



#param_list = [0.1, 0.08, 0.06, 0.04, 0.02]
param_list = [0.01,0.03,0.05,0.07,0.09]
param_noCPV = 0.0

for p in param_list:
	generateM_dd(p)

#generateM_dd(param_noCPV, 200)

# For Plotting
if False:
	generateM_dd(1.0, 1,10000,True)
	generateM_dd(0.0, 1,10000,True)
	generateM_dd(0.5, 1,10000,True)
	generateM_dd(0.1, 1,10000,True)



