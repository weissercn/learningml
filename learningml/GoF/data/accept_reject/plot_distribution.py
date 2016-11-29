
from __future__ import print_function
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.colors import LinearSegmentedColormap
import os


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

colors_red = [(1, 1, 1), (1, 0, 0), (0, 0, 0)]
colors_blue= [(1, 1, 1), (0, 0, 1), (0, 0, 0)]

cm_red = LinearSegmentedColormap.from_list("GoF_red", colors_red, N=20)
cm_blue= LinearSegmentedColormap.from_list("GoF_blue", colors_blue, N=20)
################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

print("We have to invert the sin problem x1, x2 -> x2, x1")

file0_name = os.environ['learningml']+"/GoF/data/accept_reject/sin1diff_data/data_sin1diff_5_and_5_periods2D_10000_sample_0.txt"
file1_name = os.environ['learningml']+"/GoF/data/accept_reject/sin1diff_data/data_sin1diff_5_and_6_periods2D_10000_sample_0.txt"


name = "data_sin1diff_5_and_6_periods2D_10000_sample_0" 

data0 = np.loadtxt(file0_name)
data1 = np.loadtxt(file1_name)


xedges = np.linspace(-1.,1.,51)
yedges = np.linspace(-1.,1.,51)
H, xedges, yedges = np.histogram2d(data0[:,0], data0[:,1], bins=(xedges, yedges))
fig = plt.figure()
ax = fig.add_axes([0.2,0.15,0.75,0.8])
ax.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cm_blue, aspect='auto')
ax.set_xlabel(r"$\theta_1$")
ax.set_ylabel(r"$\theta_2$")
fig.savefig(name+"_2Dhist_noCPV.pdf")
plt.close(fig)
print("plotting "+name+"_2Dhist_noCPV.pdf")

xedges = np.linspace(-1.,1.,51)
yedges = np.linspace(-1.,1.,51)
H, xedges, yedges = np.histogram2d(data1[:,0], data1[:,1], bins=(xedges, yedges))
fig = plt.figure()
ax = fig.add_axes([0.2,0.15,0.75,0.8])
ax.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cm_red, aspect='auto')
ax.set_xlabel(r"$\theta_1$")
ax.set_ylabel(r"$\theta_2$")
fig.savefig(name+"_2Dhist_CPV.pdf")
plt.close(fig)
print("plotting "+name+"_2Dhist_CPV.pdf")



print("data0.shape : ",data0.shape)
#print("data0 : \n", data0[:10,0])

x1min = min( [np.min(data0[:,1]),np.min(data1[:,1])])
x1max = max( [np.max(data0[:,1]),np.max(data1[:,1])])
x2min = min( [np.min(data0[:,0]),np.min(data1[:,0])])
x2max = max( [np.max(data0[:,0]),np.max(data1[:,0])])

xmin  = min(x1min, x2min)
xmax  = max(x1max, x2max)

x1bins = np.linspace(xmin, xmax, 51)
x2bins = np.linspace(xmin, xmax, 51)

x1widths = (x1bins[1:]-x1bins[:-1])/2.
x2widths = (x2bins[1:]-x2bins[:-1])/2.

fig = plt.figure()
ax = fig.add_axes([0.2,0.15,0.75,0.8])
hist0, hist0_edges = np.histogram(data0[:,0], bins=x1bins)
hist1, hist1_edges = np.histogram(data1[:,0], bins=x1bins)
print("x1 hist0 : ", hist0)
bin_middle = (hist0_edges[1:] + hist0_edges[:-1]) / 2
ax.errorbar(bin_middle, hist0, xerr=x1widths, yerr=np.sqrt(hist0), linestyle='', marker='*', markersize=15, color='blue', label=r'$f_1$')
ax.errorbar(bin_middle, hist1, xerr=x1widths, yerr=np.sqrt(hist1), linestyle='', marker='*', markersize=15, color='red',  label=r'$f_2$')
#plt.scatter(bin_middle,hist1, s=120, c='red', marker = '*', lw = 0)

#plt.hist(data0[:,0], bins=x1bins, alpha=0.5, color= 'blue', label='F0')
#plt.hist(data1[:,0], bins=x1bins, alpha=0.5, color= 'red', label='F1')
#plt.title("Gauss projection onto x1")
ax.set_xlim(-1,1)
ax.set_ylim(0,800)
#plt.plot(x1bins,10000*mlab.normpdf(x1bins, mu, sig),color='black',linewidth=2.)
ax.set_xlabel(r"$\theta_2$")
ax.set_ylabel('Number of events')
ax.legend(loc='upper right', frameon=False, numpoints=1)
fig.savefig(name+"_1D_theta2_proj_comp.pdf")
print("plotting "+name+"_1D_theta2_proj_comp.pdf")


fig = plt.figure()
ax = fig.add_axes([0.2,0.15,0.75,0.8])
hist0, hist0_edges = np.histogram(data0[:,1], bins=x2bins)
hist1, hist1_edges = np.histogram(data1[:,1], bins=x2bins)
print("x2 hist0 : ", hist0)
bin_middle = (hist0_edges[1:] + hist0_edges[:-1]) / 2
ax.errorbar(bin_middle, hist0, xerr=x2widths, yerr=np.sqrt(hist0), linestyle='', marker='*', markersize=15, color='blue', label=r'$f_1$')
ax.errorbar(bin_middle, hist1, xerr=x2widths, yerr=np.sqrt(hist1), linestyle='', marker='*', markersize=15, color='red',  label=r'$f_2$')

#plt.hist(data0[:,1], bins=x2bins, alpha=0.5, color= 'blue', label='F0')
#plt.hist(data1[:,1], bins=x2bins, alpha=0.5, color= 'red', label='F1')
#plt.title("Gauss projection onto x2")
ax.set_xlim(-1,1)
ax.set_ylim(0,800)
#plt.plot(x1bins,mlab.normpdf(x1bins, mu, sig),color='black',linewidth=2.)
ax.set_xlabel(r"$\theta_1$")
ax.set_ylabel('Number of events')
ax.legend(loc='upper right', frameon=False, numpoints=1)
fig.savefig(name+"_1D_theta1_proj_comp.pdf")
print("plotting "+name+"_1D_theta1_proj_comp.pdf")



