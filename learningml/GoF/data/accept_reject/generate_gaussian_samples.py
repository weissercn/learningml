#!/usr/bin/env python

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#
# Name:     generate_gaussian_samples.py 
#
# Author:   Constantin Weisser (weisser@mit.edu)
#
# Purpose:  This is a python script to write a file containing 10000 data points
#	    sampled from a 2D Gaussian
#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# 


#Constantin Weisser
from __future__ import print_function
from random import gauss
import sys
import numpy as np
import json		#needed to read in means and stdev as numpy arrays
import random
from scipy import stats
from scipy.special import legendre
from scipy.special import sph_harm
import matplotlib.pyplot as plt 
from termcolor import colored
from matplotlib.colors import LogNorm
import os
from scipy.interpolate import interp1d

def Legendre_sum(x,contrib):
        "Compute the sum of Legendre Polynomicals. Expect input (x1,x2,...xn) and ((AI,n1I,n2I,..),(AII,n1II,n2II,...),...)" 
        ampl = 0
	ndim = contrib.shape[1]-1
	ncontrib = contrib.shape[0]
	#print("Number of dimensions : ", ndim )
	assert x.shape[0]==ndim
	#print("Number of contributions : ", ncontrib)
        for c in range(ncontrib):
		ampl_counting = contrib[c,0]
		for d in range(ndim):
			ampl_counting= ampl_counting*legendre(contrib[c,d+1])(x[d])
			#print("ampl_counting : ",ampl_counting)	
                ampl += ampl_counting
		#print("ampl : ",ampl)
        prob = np.real(ampl* np.conjugate(ampl))
        return prob

def LegendreSqu_sum(x,contrib):
        "Compute the sum of Legendre Polynomicals. Expect input (x1,x2,...xn) and ((AI,n1I,n2I,..),(AII,n1II,n2II,...),...)" 
        ampl = 0 
        ndim = contrib.shape[1]-1
        ncontrib = contrib.shape[0]
        #print("Number of dimensions : ", ndim )
        assert x.shape[0]==ndim
        #print("Number of contributions : ", ncontrib)
        for c in range(ncontrib):
                ampl_counting = contrib[c,0]
                for d in range(ndim):
                        ampl_counting= ampl_counting*np.square(legendre(contrib[c,d+1])(x[d]))
                        #print("ampl_counting : ",ampl_counting)        
                ampl += ampl_counting
                #print("ampl : ",ampl)
        prob = np.real(ampl* np.conjugate(ampl))
        return prob

def SinSqu(x_,SinSqu_periods_):
	sin_contrib = np.square(np.sin(x_*np.pi*SinSqu_periods_/2.0))
	#print("sin_contrib : ",sin_contrib)
	return np.prod(np.square(np.sin(x_*np.pi*SinSqu_periods_/2.0)))

def SinPlusOne(x_,Sin_periods_):
        sin_contrib = np.sin(x_*np.pi*Sin_periods_)
        #print("sin_contrib : ",sin_contrib)
	return 1+ np.prod(sin_contrib)

def B_decay_K_mu_mu(tk,tl,p,dict_of_changes):
	#tk = theta_K, tl = theta_{\ell} and p = phi have ranges {0,pi}, {0,pi},{-pi,pi}. Let's make change these ranges to {-1,1}
	p  = p/np.pi
	tk = tk*2.0/np.pi-1.0
	tl = tl*2.0/np.pi-1.0
	# the values declared below are either measured or the Standard model predictions
	# let's put them in a dictionary called d
	d = {'FL': 0.690,'FLe': 0.04,'AFB': -0.075,'AFBe': 0.035,'S3': 0.012,'S4': -0.155, 'S5': -0.23, 'S7': -0.077, 'S8': 0.028, 'S9': -0.064, 'S3e': 0.038, 'S4e': 0.058, 'S5e': 0.05, 'S7e': 0.052, 'S8e': 0.058, 'S9e': 0.042}
	#print("original d: ",d)
	#print("dict_of_changes: ",dict_of_changes)
	# the following entries don't seem to be in the formula: FLe, AFBe, S3e, S4e, S5e, S7e, S8e, S9e 
	for key, value in dict_of_changes.iteritems():
		d[key]=value
	#print("d after changes: ", d)

  	return 0.75*(1-d['FL'])*np.power(np.sin(tk),2) + d['FL']*np.power(np.cos(tk),2)+ 0.25*(1-d['FL'])*np.power(np.sin(tk),2)*np.cos(2*tl) - d['FL']*np.power(np.cos(tk),2)*np.cos(2*tl) + d['S3']*np.power(np.sin(tk)*np.sin(tl),2)*np.cos(2*p) + d['S4']*np.sin(2*tk)*np.sin(2*tl)*np.cos(p) + d['S5']*np.sin(2*tk)*np.sin(tl)*np.cos(p) + 1.33*d['AFB']*np.power(np.sin(tk),2)*np.cos(tl) + d['S7']*np.sin(2*tk)*np.sin(tl)*np.sin(p) + d['S8']*np.sin(2*tk)*np.sin(2*tl)*np.sin(p) + d['S9']*np.power(np.sin(tk)*np.sin(tl),2)*np.sin(2*p)

	

nsam=200

optimisation_mode=0 #if it is on nsam=1 and "optimisation" gets appended to the name

if optimisation_mode==1:
	nsam=1

points_per_sample=10000

dimensions=[2,3,4,5,6,7,8,9,10]
#dimensions=[4]
n_maxvalue_too_low_all_samples=0
coefficients=[1,0.5,2,0.7]

high_oscillation_mode=1
order_of_poly_high_oscillation = 100

# 'legendre', 'legendre_squared', 'sin', 'sin1diff', 'sinplusone', 'B_decay_K_mu_mu'
# Only sin1diff was actually used for the paper
function_mode='sin1diff'
#if 'sin'
SinSqu_periods=39
#if 'sin1diff' or 'sinplusone'
Sin1diff_normal_period = 5
Sin1diff_diff_period   = 5

#if 'B_decay_K_mu_mu'
adict_of_changes = {'AFB': -0.03}

# acceptance mode =1: acceptance introduced and kept the same for all samples; =2: acceptance introduced and varied for each sample; =anything else: acceptance not introduced 
acceptance_mode=0
acceptance_points_per_axis = 100


####################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################


#introducing a different acceptance for each value of x,...
if acceptance_mode==1:
	f_list = []
	for dim in range(np.max(dimensions)):
		r = [random.uniform(0.9, 1.0) for _ in range(acceptance_points_per_axis)]
		# The range is always from -1 to 1
		rx = np.linspace(-1.0, 1.0, acceptance_points_per_axis)
		f = interp1d(rx, r)
		# f is a <class 'scipy.interpolate.interpolate.interp1d'>
		f_list.append(f)
	


if False:
	# To test the functional form of a function
	x=np.arange(-1.0,1.01,0.01)
	y=np.zeros_like(x)
	for i in range(len(x)):
		y[i]=SinPlusOne(np.array([x[i]]),np.array([2]))
	print("y : ",y)
	plt.figure()
	plt.plot(x,y)
	plt.savefig("testing_SinPlusOne_1D.png")

	#x2=np.arange(-1.0,1.01,0.01)
	#y2=np.arange(-1.0,1.01,0.01)
	#xx, yy = np.meshgrid(x2, y2, sparse=True)
	#z = SinPlusOne(np.array([xx,yy]),np.array([2,5]))
	#plt.figure()
	#plt.contourf(x,y,z)
	#plt.savefig("testing_SinPlusOne_2D.png")

for number_of_dimensions in dimensions:

	for i in range(len(coefficients)):
		l = [coefficients[i]]
		for j in range(number_of_dimensions):
			 l.append((i+j)%4)
		if i==0:
			contrib=[l]
		else:
			contrib.append(l)

	contrib=np.array(contrib)
	if high_oscillation_mode==1:
		l=[1]
		for n in range(number_of_dimensions):
			l.append(order_of_poly_high_oscillation)
		contrib=np.array([l])
	print("contrib : ", contrib)

	if acceptance_mode==2:
		f_list = []
		for dim in range(number_of_dimensions):
			r = [random.uniform(0.9, 1.0) for _ in range(acceptance_points_per_axis)]
			# The range is always from -1 to 1
			rx = np.linspace(-1.0, 1.0, acceptance_points_per_axis)
			f = interp1d(rx, r)
			# f is a <class 'scipy.interpolate.interpolate.interp1d'>
			f_list.append(f)

	for nth_sample in range(nsam):
		#Accept reject
		n_sampled =0
		x=np.zeros(number_of_dimensions)
		#print("x : ",x)
		ndim = contrib.shape[1]-1
		ncontrib = contrib.shape[0]
		n_maxvalue_too_low =0
		cutoff_max = 0
		while (n_sampled < points_per_sample):
			for d in range(ndim):
				x[d]=random.uniform(-1,1)
			#print("x : ",x)
			if function_mode=='legendre_squared':
				cutoff=LegendreSqu_sum(x,contrib)
			elif function_mode=='legendre':
				cutoff=Legendre_sum(x,contrib)
			elif function_mode=='sin':
				cutoff=SinSqu(x,SinSqu_periods)
			elif function_mode=='sin1diff':
				periods = [Sin1diff_normal_period] * (number_of_dimensions -1)	
				periods.append(Sin1diff_diff_period)
				cutoff=SinSqu(x,periods)
			elif function_mode=='sinplusone':
				periods = [Sin1diff_normal_period] * (number_of_dimensions -1)  
                                periods.append(Sin1diff_diff_period)
                                cutoff=SinPlusOne(x,periods)
			elif function_mode=='B_decay_K_mu_mu':
				if number_of_dimensions !=3:
					exit_code=os.system('osascript -e '"'"'tell app "Terminal" to display dialog "B_decay_K_mu_mu needs to have 3D data. no more or less" '"'"'')
					sys.exit()
				cutoff=B_decay_K_mu_mu(x[0],x[1],x[2],adict_of_changes)
			else:
				print("No valid mode was chosen")

			if acceptance_mode==1 or acceptance_mode==2:
				for i in range(number_of_dimensions):
					cutoff = cutoff * f_list[i](x[i])
				
	
			if cutoff>cutoff_max:
				cutoff_max = cutoff
			#print("x : ",x," cutoff : ", cutoff)
			# the maximum of each legendre polynomial between x [-1,1] is 1. Hence, the maximum amplitude is the sum of the coefficients
			maxampl=0
			for c in range(ncontrib):
				maxampl += complex(abs(np.real(contrib[c,0])),abs(np.imag(contrib[c,0])))
			maxvalue = np.real(maxampl*np.conjugate(maxampl))
			if function_mode=='sin' or function_mode=='sin1diff':
				maxvalue = 1.0 
			if function_mode=='sinplusone':
				maxvalue = 2.0
			if function_mode=='B_decay_K_mu_mu':
				# from cutoff_max
				maxvalue = 3.0
			random_number = random.uniform(0,maxvalue)
			#print("random_number : ",random_number)
			#print("x : ",x,"  cutoff : ",cutoff,"  maxvalue : ", maxvalue, "  random_number : ", random_number)
			if cutoff > maxvalue:
				n_maxvalue_too_low +=1
				print("The Lengendre sum value was bigger than the maxvalue expected. The probability distribution might not be correctly represented now.")
			if cutoff > random_number:
				#print("n_sampled : ",n_sampled)
				if n_sampled ==0:
					p_sh = x
				else:
					p_sh=np.vstack((p_sh,x))               

				n_sampled = n_sampled + 1
				if n_sampled%200==0:		 
					print("x : ",x)
					print( n_sampled, colored(" points have been sampled for sample ",'red'),nth_sample)

		n_maxvalue_too_low_all_samples += n_maxvalue_too_low
		print("\n ######################################################## \n")
		print("p_sh[-5:] : ", p_sh[-5:])
		if n_maxvalue_too_low > 0:
			print("2*cutoff_max : ",2*cutoff_max)
			exit_code=os.system('osascript -e '"'"'tell app "Terminal" to display dialog "n_maxvlue_too_low_all_samples is now {0}" '"'"''.format(n_maxvalue_too_low_all_samples)) 
			if exit_code!=0:
				sys.exit()
		print("n_maxvalue_too_low : ",n_maxvalue_too_low)
		#print("The inputs were: \n contrib : ", contrib)
		
		name= function_mode+ "_"
		if function_mode=='sin':
			name+= str(SinSqu_periods)+"_periods_"+str(number_of_dimensions)+"D_"
		elif function_mode=='sin1diff' or function_mode=='sinplusone':
			name+= str(Sin1diff_normal_period) + "_and_" +str(Sin1diff_diff_period) + "_periods"+str(number_of_dimensions)+"D_" 
		elif function_mode=='B_decay_K_mu_mu':
			# if adict_of_changes is empty
			if not bool(adict_of_changes):
				name+= "original_"
			for key, value in adict_of_changes.iteritems():
				name+= key + "_" + str(value) +"_"
		else:
			for c in range(ncontrib):
				name+="contrib"+str(c)+"__"
				for d in range(ndim+1):
					name+=str(contrib[c,d]).replace(".", "_")+"__"
		name+= str(points_per_sample)+"_"
		if acceptance_mode==1:
			name+="acceptance_all_sample_"
		elif acceptance_mode==2:
			name+="acceptance_each_sample_"		
		
		if optimisation_mode==1:
			name+="sample_optimisation_"	
			name+= str(nth_sample)
		else:
			name+= "sample_"+str(nth_sample)
		
		data_name = function_mode+"_data/data_"+name+".txt"
		print("data_name : ",data_name)
		np.savetxt( data_name,p_sh)

		if nth_sample==0:

			plt.figure()
			plt.hist(p_sh[:,0], bins=100, facecolor='red', alpha=0.5)
			plt.title(name+" 1D Histogram")
			plt.xlim(-1,1)
			plt.savefig(name+"1Dhist.png")
			print("plotting "+name+"1Dhist.png")
		
			if ndim >1:
				plt.figure()
				plt.hist2d(p_sh[:,0],p_sh[:,1], bins=20)
				plt.title(name+" 2D Histogram")
				plt.xlim(-1,1)
				plt.ylim(-1,1)
				cb= plt.colorbar()
				cb.set_label("number of events")
				plt.savefig(name+"2Dhist.png")
				print("plotting "+name+"2Dhist.png")
	


print("n_maxvalue_too_low_all_samples : ",n_maxvalue_too_low_all_samples)

################################################################################################################################################################################################################################################################################################################################################################################################################
#########################################################################   O L D !!!   #####################################################################################################################################################################################################################################################################################################################################

if 0:
	def SphericalHarmonics_sum(x,contrib):
		"Compute the sum of spherical harmonicsExpect input (theta1, phi1,theta2,phi2) and ((A,l1,m1,l2,m2),...)"
		ampl = 0
		for i in range(contrib.shape[0]):
			#print("A , m1 : ", contrib[i,0], contrib[i,2])
			ampl += contrib[i,0]*sph_harm(contrib[i,1],contrib[i,2],x[0],x[1])*sph_harm(contrib[i,3],contrib[i,4],x[2],x[3])
		#print("ampl : ",ampl)
		prob = np.real(ampl* np.conjugate(ampl))
		return prob

	class gaussian_gen_rv_continuous(stats.rv_continuous):
		"Gaussian distribution"
		def _pdf(self, x): 
			return np.exp(-x**2 / 2.) / np.sqrt(2.0 * np.pi)

	gauss = gaussian_gen_rv_continuous(name='gaussian')
	print("gauss.pdf([-7,-1,0,1,7]) : ", gauss.pdf([-7,-1,0,1,7]))
	print("gauss.rvs() : ",gauss.rvs(size=5))

	class Legendre_sum_rv_continuous(stats.rv_continuous):
		def _pdf(self,x):
			return abs(legendre(3)(x))
			#return A[0]*legendre(0)(x)+A[1]*legendre(1)(x)+A[2]*legendre(2)(x)+A[3]*legendre(0)(x)*legendre(1)(x)+A[4]*legendre(0)(x)*legendre(2)(x)+A[5]*legendre(1)(x)*legendre(2)(x)+A[6]*legendre(0)(x)*legendre(1)(x)*legendre(2)(x)
	legen= Legendre_sum_rv_continuous(name='legendre')
	print("legen.pdf([-1,-0.5,0,0.5,1]) : ", legen.pdf([-1,-0.5,0,0.5,1]))
	#print("legen.rvs() : ", legen.rvs(size=5))


	#p=legen.rvs(size=1000)

	#plt.figure()
	#plt.hist(p, bins=100, facecolor='red', alpha=0.5)
	#plt.title("Legendre 1D Histogram")
	#plt.xlim(-1,1)
	#plt.savefig("legendre_1D_hist_noCPV.png")
	#print("plotting legendre_1D_hist_noCPV.png")

	print("########################################################")



	#Accept reject
	n_sampled =0
	contrib=np.array([[1,1,2,1,2]])
	print("contrib.shape[0] : ", contrib.shape[0])
	x=np.array([0.0,0.0,0.0,0.0])
	while (n_sampled < 3):
		x[0]=random.uniform(0, 2*np.pi)
		x[1]=random.uniform(0, np.pi) 
		x[2]=random.uniform(0, 2*np.pi) 
		x[3]=random.uniform(0, np.pi)
		#print("x : ",x)
		cutoff=SphericalHarmonics_sum_fn(x,contrib)
		#print("cutoff : ", cutoff)
		random_number = random.random()
		#print("random_number : ",random_number)
		if  cutoff > random_number:
			n_sampled = n_sampled + 1
			print( n_sampled," points have been sampled.")
			ax = np.array([x[0],x[1],x[2],x[3]])
			print ("ax : " , ax)
			if n_sampled ==1:
				p_sh = ax
			else:
				print("p_sh previous : ", p_sh)
				np.concatenate((p_sh,ax))
			print("p_sh : ", p_sh)
	print("p_sh : ", p_sh)

	#spherharm= SphericalHarmonics_sum(name='spherical_harmonics')
	#print("spherharm.pdf([[-1,-1],[-0.5,-0.5],[0,0],[0.5,0.5],[1,1]]) : ", spherharm.pdf([[-1,-1],[-0.5,-0.5],[0,0],[0.5,0.5],[1,1]]))
	#print("spherharm.rvs() : ", spherharm.rvs(size=5))

	#p_sh=spherharm.rvs(size=1000)

	plt.figure()
	plt.hist(p_sh[:,0], bins=100, facecolor='red', alpha=0.5)
	plt.title("Spherical Harmonics 1D Histogram")
	plt.xlim(-1,1)
	plt.savefig("spherical_harmonics_1D_hist_noCPV.png")
	print("plotting spherical_harmonics_1D_hist_noCPV.png")


	import time
	time.sleep(200)

	no_points=10000
	original_mean1 = 0.2
	original_mean2 = 0.8
	original_std = 0.05 
	label_no = 1

	args = str(sys.argv)
	#print ("Args list: %s " % args)
	#The first argument is the name of this python file
	total = len(sys.argv)
	verbose=True

	if(total==8):
		no_points = int(sys.argv[1])
		#mean = np.array(json.loads(sys.argv[2]))
		#std = np.array(json.loads(sys.argv[3]))
		original_mean1 = float(sys.argv[2])
		original_mean2 = float(sys.argv[3])
		original_std = float(sys.argv[4])
		distance_to_original = float(sys.argv[5])
		no_dim = int(sys.argv[6])
		label_no =float(sys.argv[7])
	else:	
		print("Using standard arguments")

	if verbose:
		print("original_mean1 : ", original_mean1)
		print("original_mean2 : ", original_mean2)
		print("original_std : ",original_std)


	#print(mean.shape[0])

	for dim in range(no_dim):
		values = np.zeros((no_points,1))
		for i in range(no_points):
			if bool(random.getrandbits(1)):
				values[i] = gauss(original_mean1+distance_to_original,original_std)
			else:
				values[i] = gauss(original_mean2-distance_to_original,original_std)
		#print(values)
		if dim==0:
			full_cords=values
		else:
			full_cords=np.column_stack((full_cords,values))


	print(full_cords)

	np.savetxt("gauss_data/data_double_high{0}Dgauss_".format(int(no_dim))+str(int(no_points))+"_"+str(original_mean1)+"_"+str(original_mean2)+"_"+str(original_std)+"_"+str(distance_to_original)+"_"+str(int(label_no))+  ".txt",full_cords)





