from __future__ import print_function
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

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

################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################



file0_name = "gauss_data/gaussian_same_projection_on_each_axis_2D_10000_0.0_1.0_1.0_0.txt"
#file0_name = "gauss_data/gaussian_same_projection_on_each_axis_rotated_perfectly_2D_10000_0.0_1.0_1.0_0.txt"
file1_name = "gauss_data/gaussian_same_projection_on_each_axis_2D_10000_0.0_varysigma2_0.9_1.0_0.txt"
#file1_name = "gauss_data/gaussian_same_projection_on_each_axis_rotated_perfectly_2D_10000_0.0_varysigma2_0.9_1.0_0.txt"

name = "gaussian_same_projection_on_each_axis_2D_10000_0.0_varysigma2_0.9_1.0_0"
#name = "gaussian_same_projection_on_each_axis_rotated_perfectly_2D_10000_0.0_varysigma2_0.9_1.0_0"

data0 = np.loadtxt(file0_name)
data1 = np.loadtxt(file1_name)

print("data0.shape : ",data0.shape)
#print("data0 : \n", data0[:10,0])

x1min = np.min( [np.min(data0[:,0]),np.min(data1[:,0])])
x1max = np.max( [np.max(data0[:,0]),np.max(data1[:,0])])
x2min = np.min( [np.min(data0[:,1]),np.min(data1[:,1])])
x2max = np.max( [np.max(data0[:,1]),np.max(data1[:,1])])

x1bins = np.linspace(x1min, x1max, 20)
x2bins = np.linspace(x2min, x2max, 20)

mu = 0.
sig = 1.

#print("data0 : \n", data0[:10,:])

plt.figure()
hist0, hist0_edges = np.histogram(data0[:,0], bins=x1bins)
hist1, hist1_edges = np.histogram(data1[:,0], bins=x1bins)
bin_middle = (hist0_edges[1:] + hist0_edges[:-1]) / 2
plt.errorbar(bin_middle, hist0, yerr=np.sqrt(hist0), marker='o', markersize=6, color= 'blue', label='F0')
plt.errorbar(bin_middle, hist1, yerr=np.sqrt(hist1), marker='s', markersize=6, color= 'red', label='F1')

#plt.hist(data0[:,0], bins=x1bins, alpha=0.5, color= 'blue', label='F0')
#plt.hist(data1[:,0], bins=x1bins, alpha=0.5, color= 'red', label='F1')
#plt.title("Gauss projection onto x1")
plt.xlim(-4,4)
#plt.ylim(-4,4)
#plt.plot(x1bins,10000*mlab.normpdf(x1bins, mu, sig),color='black',linewidth=2.)
plt.xlabel(r"$\mathit{x}_1$")
plt.ylabel('Number of events')
plt.legend(loc='best')
plt.savefig(name+"_1D_x1_proj_comp.png")
print("plotting "+name+"_1D_x1_proj_comp.png")


plt.figure()
hist0, hist0_edges = np.histogram(data0[:,1], bins=x2bins)
hist1, hist1_edges = np.histogram(data1[:,1], bins=x2bins)
bin_middle = (hist0_edges[1:] + hist0_edges[:-1]) / 2
plt.errorbar(bin_middle, hist0, yerr=np.sqrt(hist0), marker='o', markersize=6, color= 'blue', label='F0')
plt.errorbar(bin_middle, hist1, yerr=np.sqrt(hist1), marker='s', markersize=6, color= 'red', label='F1')

#plt.hist(data0[:,1], bins=x2bins, alpha=0.5, color= 'blue', label='F0')
#plt.hist(data1[:,1], bins=x2bins, alpha=0.5, color= 'red', label='F1')
#plt.title("Gauss projection onto x2")
plt.xlim(-4,4)
#plt.ylim(-4,4)
#plt.plot(x1bins,mlab.normpdf(x1bins, mu, sig),color='black',linewidth=2.)
plt.xlabel(r"$\mathit{x}_2$")
plt.ylabel('Number of events')
plt.legend(loc='best')
plt.savefig(name+"_1D_x2_proj_comp.png")
print("plotting "+name+"_1D_x2_proj_comp.png")


