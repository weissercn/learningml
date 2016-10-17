from __future__ import print_function

import sys
import numpy as np
from scipy import stats
import adaptive_binning_chisquared_2sam

def weisser_searchsorted(l_test1, l_test2):
        l_test1, l_test2 = np.array(l_test1), np.array(l_test2)
        #print("l_test1 : ", l_test1)
        l_tot = np.sort(np.append(l_test1, l_test2))
        #print("l_tot : ", l_tot)
        l_tot_cp, l_test1_cp, l_test2_cp  = l_tot.tolist(), l_test1.tolist(), l_test2.tolist()
        pos1, pos2 = [],[]
        #print("l_tot_cp : ",l_tot_cp)
        for item_number, item in enumerate(l_tot_cp):
                n1 = l_test1_cp.count(item)
                n2 = l_test2_cp.count(item)
                if np.random.choice(2, 1, p=[n1/float(n1+n2),n2/float(n1+n2) ]):
                        l_test2_cp.remove(item)
                        pos2.append(item_number)
                else:
                        l_test1_cp.remove(item)
                        pos1.append(item_number)

        pos1 = np.array(pos1)
        pos2 = np.array(pos2)
        return (pos1,pos2)


def p_value_scoring_object(clf, X, y):
	"""
	p_value_getter is a scoring callable that returns the negative p value from the KS test on the prediction probabilities for the particle and antiparticle samples.  
	"""

	#Finding out the prediction probabilities
	prob_pred=clf.predict_proba(X)[:,1]
	#print(prob_pred)

	#This can be deleted if not using Keras
	#For Keras turn cathegorical y back to normal y
	if y.ndim==2:
		if y.shape[0]!=1 and y.shape[1]!=1:
			#Then we have a cathegorical vector
			y = y[:,1]

	#making sure the inputs are row vectors
	y         = np.reshape(y,(1,y.shape[0]))
	prob_pred = np.reshape(prob_pred,(1,prob_pred.shape[0]))

	#Separate prob into particle and antiparticle samples
	prob_0    = prob_pred[np.logical_or.reduce([y==0])]
	prob_1    = prob_pred[np.logical_or.reduce([y==1])]
	#if __debug__:
		#print("Plot")
	p_KS_stat=stats.ks_2samp(prob_0,prob_1)
	print(p_KS_stat)
	p_KS=-p_KS_stat[1]
	return p_KS

def make_p_value_scoring_object_test(greeting):
	def p_value_scoring_object_test(clf, X, y):
		"""
		p_value_getter is a scoring callable that returns the negative p value from the KS test on the prediction probabilities for the particle and antiparticle samples.  
		"""
		print("Greeting : ", greeting)

		#Finding out the prediction probabilities
		prob_pred=clf.predict_proba(X)[:,1]
		#print(prob_pred)

		#This can be deleted if not using Keras
		#For Keras turn cathegorical y back to normal y
		if y.ndim==2:
			if y.shape[0]!=1 and y.shape[1]!=1:
                        #Then we have a cathegorical vector
                        	y = y[:,1]

		#making sure the inputs are row vectors
		y         = np.reshape(y,(1,y.shape[0]))
		prob_pred = np.reshape(prob_pred,(1,prob_pred.shape[0]))

		#Separate prob into particle and antiparticle samples
		prob_0    = prob_pred[np.logical_or.reduce([y==0])]
		prob_1    = prob_pred[np.logical_or.reduce([y==1])]
		#if __debug__:
			#print("Plot")
		p_KS_stat=stats.ks_2samp(prob_0,prob_1)
		print(p_KS_stat)
		p_KS=-p_KS_stat[1]
		return p_KS
	return p_value_scoring_object_test

def make_p_value_scoring_object_binned_chisquared(no_bins,systematics_fraction,title,name,PLOT):
	def p_value_scoring_object_binned_chisquared(clf, X, y): 
		""" 
		p_value_getter is a scoring callable that returns the negative p value from a binned chi2 test on the prediction probabilities for the particle and antiparticle samples. Both a number of bins and a list of number of bins can be entered. 
		"""
		#if not isinstance(single_no_bins_list, (list,tuple)): single_no_bins_list = [single_no_bins_list]

		#no_bins= 2

		print("no_bins : ",no_bins) 
		#Finding out the prediction probabilities
		prob_pred=clf.predict_proba(X)[:,1]
		#print(prob_pred)

		#This can be deleted if not using Keras
		#For Keras turn cathegorical y back to normal y
		if y.ndim==2:
			if y.shape[0]!=1 and y.shape[1]!=1:
				#Then we have a cathegorical vector
				y = y[:,1]

		#making sure the inputs are row vectors
		y         = np.reshape(y,(1,y.shape[0]))
		prob_pred = np.reshape(prob_pred,(1,prob_pred.shape[0]))
		y = y[0]
		prob_pred = prob_pred[0]
		print("prob_pred : ", prob_pred)
		#print("\ny : ",y)
		#print("\nprob_pred : ",prob_pred )

		#Separate prob into particle and antiparticle samples
		prob_0    = prob_pred[np.logical_or.reduce([y==0])]
		prob_1    = prob_pred[np.logical_or.reduce([y==1])]
		print("prob_0 : ",prob_0)
		print("prob_1 : ",prob_1)
		total_no  = len(prob_0)+len(prob_1)
		print("\ntotal_no : ",total_no)
		assert total_no == prob_pred.shape[0]

		#new
		prob_pred = np.append(prob_0, prob_1) 

		#prob_0, prob_1, prob_pred = list(prob_0), list(prob_1), list(prob_pred)
		#print("prob_0 : ",prob_0)
		# Create transformation to turn distributions into uniform distributions if they aren't different
		prob_pred_sorted = np.sort(prob_pred)
		# Create cumulative distribution. Make every element a bin
		#prob_0_pos = np.searchsorted(prob_pred_sorted,prob_0)
		#prob_1_pos = np.searchsorted(prob_pred_sorted,prob_1)
		prob_0_pos, prob_1_pos = weisser_searchsorted(prob_0, prob_1)

		print("prob_0_pos : ", prob_0_pos)
		prob_0_pos_float = [float(i) for i in prob_0_pos]
		prob_1_pos_float = [float(i) for i in prob_1_pos]
		#These are now the bin positions.
		prob_0_pos_scaled = np.divide(prob_0_pos_float,float(total_no-1))
		prob_1_pos_scaled = np.divide(prob_1_pos_float,float(total_no-1))

		print("prob_0_pos_scaled : ",prob_0_pos_scaled)
		#prob_0_pos_scaled = prob_0_pos_scaled[:,None]
		#prob_1_pos_scaled = prob_1_pos_scaled[:,None]
		#print("prob_0_pos_scaled : ",prob_0_pos_scaled)
		bin_boundaries = np.linspace(0.0,1.0,20)
		hist0, hist0_edges = np.histogram(prob_0_pos_scaled, bins=bin_boundaries)
		hist1, hist1_edges = np.histogram(prob_1_pos_scaled, bins=bin_boundaries)
		hist_comb=np.add(hist0,hist1)
		
		print("hist0 : ", hist0)
		print("hist1 : ", hist1)
		print("hist_comb : ", hist_comb)

		for item_number, item in enumerate(hist_comb):
			assert item >= np.floor(np.sum(hist_comb)/float(hist_comb.shape[0]))
			assert item <= np.ceil(np.sum(hist_comb)/float(hist_comb.shape[0]))
		

		p_miranda = adaptive_binning_chisquared_2sam.chi2_regular_binning(prob_0_pos_scaled,prob_1_pos_scaled,no_bins,systematics_fraction,title,name,PLOT)

		#prob_0_hist = np.histogram(prob_0_pos, bins=np.linspace(0.0, 1.0, num=total_nu+1))
		#prob_1_hist = np.histogram(prob_1_pos, bins=np.linspace(0.0, 1.0, num=total_nu+1))


		#p_KS_stat=stats.ks_2samp(prob_0,prob_1)
		#print(p_KS_stat)
		#p_KS=-p_KS_stat[1]

		return - p_miranda
	return p_value_scoring_object_binned_chisquared


def please_delete_this_function():
		#def make_p_value_scoring_object_binned_chisquared(no_bins):
		"""
		This is a function that returns another function(p_value_scoring_object_binned_chisquared) for a certain number of bins. It does this by using function closure.
		"""
		#def p_value_scoring_object_binned_chisquared(clf, X, y):
		"""
		p_value_getter is a scoring callable that returns the negative p value from a binned chi2 test on the prediction probabilities for the particle and antiparticle samples. Both a number of bins and a list of number of bins can be entered. 
		"""
		#if not isinstance(single_no_bins_list, (list,tuple)): single_no_bins_list = [single_no_bins_list]

		#Finding out the prediction probabilities
		prob_pred=clf.predict_proba(X)[:,1]
		#print(prob_pred)

		#This can be deleted if not using Keras
		#For Keras turn cathegorical y back to normal y
		if y.ndim==2:
			if y.shape[0]!=1 and y.shape[1]!=1:
				#Then we have a cathegorical vector
				y = y[:,1]

		#making sure the inputs are row vectors
		y         = np.reshape(y,(1,y.shape[0]))
		prob_pred = np.reshape(prob_pred,(1,prob_pred.shape[0]))
		total_no  = prob_pred.shape[0]
		print("total_no : ",total_no)

		#Separate prob into particle and antiparticle samples
		prob_0    = prob_pred[np.logical_or.reduce([y==0])]
		prob_1    = prob_pred[np.logical_or.reduce([y==1])]
		
		# Create transformation to turn distributions into uniform distributions if they aren't different
		prob_pred_sorted = np.sort(prob_pred)
		# Create cumulative distribution. Make every element a bin
		prob_0_pos = np.searchsorted(prob_pred_sorted,prob_0)
		prob_1_pos = np.searchsorted(prob_pred_sorted,prob_1)
		#These are now the bin positions.
		prob_0_pos_scaled = float(prob_0_pos)/float(total_no-1)
		prob_1_pos_scaled = float(prob_1_pos)/float(total_no-1) 

		print("prob_0_pos_scaled : ",prob_0_pos_scaled)
		prob_0_pos_scaled = prob_0_pos_scaled[:,None]
		prob_1_pos_scaled = prob_1_pos_scaled[:,None]
		print("prob_0_pos_scaled : ",prob_0_pos_scaled)

		p_miranda_list = adaptive_binning_chisquared_2sam.chi2_regular_binning(prob_0_pos_scaled,prob_1_pos_scaled,[no_bins],0.01,True)

		#prob_0_hist = np.histogram(prob_0_pos, bins=np.linspace(0.0, 1.0, num=total_nu+1))
		#prob_1_hist = np.histogram(prob_1_pos, bins=np.linspace(0.0, 1.0, num=total_nu+1))


		#p_KS_stat=stats.ks_2samp(prob_0,prob_1)
		#print(p_KS_stat)
		#p_KS=-p_KS_stat[1]
		
		return - p_miranda_list[0]
		#p_miranda_list = [i * -1 for i in p_miranda_list]
		#if not isinstance(single_no_bins_list, (list,tuple)):	
		#	return p_miranda_list[0]
		#else:
		#	return p_miranda_list
		#<- return p_value_scoring_object_binned_chisquared

def p_value_scoring_object_visualisation(clf, X, y): 
        """ 
        p_value_getter is a scoring callable that returns the negative p value from the KS test on the prediction probabilities for the particle and antiparticle samples.  
        """
	
        #Finding out the prediction probabilities
        prob_pred=clf.predict_proba(X)[:,1]
        #print(prob_pred)

        #This can be deleted if not using Keras
        #For Keras turn cathegorical y back to normal y
        if y.ndim==2:
                if y.shape[0]!=1 and y.shape[1]!=1:
                        #Then we have a cathegorical vector
                        y = y[:,1]

        #making sure the inputs are row vectors
        y         = np.reshape(y,(1,y.shape[0]))
        prob_pred = np.reshape(prob_pred,(1,prob_pred.shape[0]))

        #Separate prob into particle and antiparticle samples
        prob_0    = prob_pred[np.logical_or.reduce([y==0])]
        prob_1    = prob_pred[np.logical_or.reduce([y==1])]
        #if __debug__:
                #print("Plot")
        p_KS_stat=stats.ks_2samp(prob_0,prob_1)

	#http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html#example-tree-plot-iris-py
	#import matplotlib.pyplot as plt
	n_classes =2 
        plot_colors = "br"

	if X.shape[1]!=2:
		print("The visualisation mode has only been implemented for 2 dimensions.")
		sys.exit(1)	

	#print("X[:,0].min() , ", X[:,0].min(), "X[:,0].max() : ", X[:,0].max())
	#print("X[:,0].min()*0.9 , ", X[:,0].min()*0.9, "X[:,0].max()*1.1 : ", X[:,0].max()*1.1)
	x_min, x_max = X[:, 0].min()*0.9 , X[:, 0].max() *1.1
    	y_min, y_max = X[:, 1].min() * 0.9, X[:, 1].max() * 1.1
	x_plot_step = (x_max - x_min)/20.0
	y_plot_step = (y_max - y_min)/20.0
	print("x_min : ", x_min, "x_max : ", x_max, "x_plot_step : ", x_plot_step)
	print("y_min : ", y_min, "y_max : ", y_max, "y_plot_step : ", y_plot_step)
	x_list=np.arange(x_min, x_max, x_plot_step)
	#print("x_list : ",x_list)
	y_list=np.arange(y_min, y_max, y_plot_step)
	#print("y_list : ", y_list)
    	xx, yy = np.meshgrid(x_list, y_list)
	print("np.c_[xx.ravel(), yy.ravel()] : ",np.c_[xx.ravel(), yy.ravel()])

	Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
	print("Z : ",Z)
	Z = np.array(Z)[:,0]
	#print("Z : ",Z)
	Z_norm = [(float(i)-min(Z))/(max(Z)-min(Z)) for i in Z]
	#if you want the pure output of the machine learning algorithm uncomment the following line
	Z = Z_norm 
	Z = np.array(Z)
	print("Z : ",Z)
	Z = Z.reshape(xx.shape)
	print("Z : ",Z)
    	cs = plt.contourf(xx, yy, Z)
	plt.colorbar()
	plt.title("Visualisation of decision boundary normalised")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.savefig("visualisation.png")
	plt.close()

	#plt.figure()
	#plt.pcolor(xx,yy,Z)
	#plt.colorbar()
	#plt.savefig("visualisation2.png")
	
	print(p_KS_stat)
        p_KS=-p_KS_stat[1]
        return p_KS

def p_value_scoring_object_AD(clf, X, y): 
        """ 
        p_value_getter is a scoring callable that returns the negative p value from the KS test on the prediction probabilities for the particle and antiparticle samples.  
        """

        #Finding out the prediction probabilities
        prob_pred=clf.predict_proba(X)[:,1]
        #print(prob_pred)

        #This can be deleted if not using Keras
        #For Keras turn cathegorical y back to normal y
        if y.ndim==2:
                if y.shape[0]!=1 and y.shape[1]!=1:
                        #Then we have a cathegorical vector
                        y = y[:,1]

        #making sure the inputs are row vectors
        y         = np.reshape(y,(1,y.shape[0]))
        prob_pred = np.reshape(prob_pred,(1,prob_pred.shape[0]))

        #Separate prob into particle and antiparticle samples
        prob_0    = prob_pred[np.logical_or.reduce([y==0])]
        prob_1    = prob_pred[np.logical_or.reduce([y==1])]
        #if __debug__:
                #print("Plot")
        p_AD_stat=stats.anderson_ksamp([prob_0,prob_1])
        print(p_AD_stat)
        p_AD=-p_AD_stat[2]
        return p_AD


if __name__ == "__main__":
	#This code is only executed when this function is called directly, not when it is imported
	print("Testing the p_value_getter function, which can be imported using 'import p_value_scoring_object' and used by typing 'p_KS=p_value_scoring_object.p_value_scoring_object(clf, X, y)'")
	from sklearn.datasets import load_iris
	from sklearn.svm import SVC
	#Load data set
	iris = load_iris()
	X = iris.data
	y = iris.target
	#To make sure there are only two classes and then shuffling
	data=np.c_[X[:100,:],y[:100]]
	np.random.shuffle(data)
	X = data[:,:-1]
	y = data[:,-1]	
	#print(X.shape)
	#print(y)
	clf= SVC(probability=True)
	clf.fit(X[:50,:],y[:50])
	
	print("KS p value : ",p_value_scoring_object(clf,X[50:,:],y[50:]))
	print("AD p value : ",p_value_scoring_object_AD(clf,X[50:,:],y[50:]))


