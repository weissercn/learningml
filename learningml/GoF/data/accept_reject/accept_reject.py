import random
import numpy as np
import matplotlib.pyplot as plt


#mode = "test"
mode = "Mike"


def bound_Dphi(phi1,phi2):
	d = (phi1+phi2) % (2*np.pi)
	if d <np.pi: bound_Dphi = d
	else: bound_Dphi = (2*np.pi - d)
	return bound_Dphi

def Mike_weight(theta1, theta2, Dphi):

        weight = 1 + (np.cos(theta1)* np.cos(theta2))**2 - ((np.sin(theta1)*np.sin(theta2))**2)*np.cos(2*Dphi)
        return weight

def test_fun(theta1, theta2):
	return (theta1-theta2)**2

ltheta1, ltheta2, lDphi = [], [], []

for i in range(1000000):
	while True:
		phi1, phi2     = random.uniform(0.,2*np.pi), random.uniform(0.,2*np.pi)
		theta1, theta2 = np.arccos(2.*np.random.rand()-1.), np.arccos(2.*np.random.rand()-1.)
		Dphi = bound_Dphi(phi1,phi2)
		if mode == "Mike":
			weight = Mike_weight(theta1,theta2,Dphi)
			weight_max = 2.0
		if mode == "test" :
			weight = test_fun(theta1, theta2)
			weight_max = np.pi**2
		assert weight < weight_max, "weight > weight_max"
		if random.random() < weight/float(weight_max) : break
	ltheta1.append(theta1)
	ltheta2.append(theta2)
	lDphi.append(Dphi)



if mode == "test":
	plt.hist2d(ltheta1,ltheta2)
	plt.colorbar()
	plt.xlabel("ltheta1")
	plt.ylabel("ltheta2")
	plt.savefig("2D_theta1_theta2_{}.png".format(mode))
	plt.clf()

elif mode == "Mike":

	plt.hist2d(lDphi,np.cos(ltheta1))
	plt.colorbar()
	plt.xlabel("Dphi")
	plt.ylabel("costheta1")
	plt.savefig("2D_Dphi_costheta1_{}.png".format(mode))
	plt.clf()

	plt.hist2d(lDphi,np.cos(ltheta2))
	plt.colorbar()
	plt.xlabel("Dphi")
	plt.ylabel("costheta2")
	plt.savefig("2D_Dphi_costheta2_{}.png".format(mode))
	plt.clf()      

        plt.hist2d(np.cos(ltheta1),np.cos(ltheta2))
        plt.colorbar()
        plt.xlabel("costheta1")
        plt.ylabel("costheta2")
        plt.savefig("2D_costheta1_costheta2_{}.png".format(mode))
        plt.clf()

