import adaptive_binning_chisquared_2sam
import os

systematics_fraction = 0.01 
dim_list = [1,2,3,4,5,6,7,8,9,10]
adaptive_binning=True
CPV  = True
PLOT = True


if CPV:	
	orig_name="chi2_gauss_0_95__0_95_CPV_not_redefined_syst_"+str(systematics_fraction).replace(".","_")+"_"
	orig_title="Gauss 0.95 0.95 syst{} adaptbin".format(systematics_fraction)
        #orig_name="chi2_gauss_0_95__0_95_CPV_not_redefined_syst_"+str(systematics_fraction).replace(".","_")+"_euclidean_plot_"
        #orig_title="Gauss 1.0 0.95 syst{} euclidean adaptbin".format(systematics_fraction)

else: 
	orig_name="chi2_gauss__1_0__1_0_noCPV_not_redefined_syst_"+str(systematics_fraction).replace(".","_")+"_"
	orig_title="Gauss 1.0 1.0 syst{} adaptbin".format(systematics_fraction)
        #orig_name="chi2_gauss__1_0__1_0_noCPV_not_redefined_syst_"+str(systematics_fraction).replace(".","_")+"_euclidean_"
        #orig_title="Gauss 1.0 1.0 syst{} euclidean adaptbin".format(systematics_fraction)



if PLOT:
	dim_list = [2,6,10]
	orig_name="plot_"+orig_name
	orig_title= "Plot "+orig_title
	sample_list_typical= [79, 74, 22]
	sample_list= [[item,item+1] for item in sample_list_typical]

else:
	sample_list = [range(100)]*len(dim_list)

comp_file_list_list = []
for dim_data_index, dim_data in enumerate(dim_list):
	comp_file_list=[]
	for i in sample_list[dim_data_index]:
		if CPV: 
			comp_file_list.append((os.environ['learningml']+"/GoF/data/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_{1}D_10000_0.0_1.0_1.0_{0}.txt".format(i,dim_data),os.environ['learningml']+"/GoF/data/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_{1}D_10000_0.0_0.95_0.95_{0}.txt".format(i,dim_data)))

			#comp_file_list.append((os.environ['learningml']+"/GoF/data/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_{1}D_10000_0.0_1.0_1.0_{0}_euclidean.txt".format(i,dim_data),os.environ['learningml']+"/GoF/data/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_{1}D_10000_0.0_0.95_0.95_{0}_euclidean.txt".format(i,dim_data)))

		else: 
			comp_file_list.append((os.environ['learningml']+"/GoF/data/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_{1}D_10000_0.0_1.0_1.0_{0}.txt".format(i,dim_data),os.environ['learningml']+"/GoF/data/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_{1}D_10000_0.0_1.0_1.0_1{0}.txt".format(str(i).zfill(2),dim_data)))

			#comp_file_list.append((os.environ['learningml']+"/GoF/data/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_{1}D_10000_0.0_1.0_1.0_{0}_euclidean.txt".format(i,dim_data),os.environ['learningml']+"/GoF/data/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_{1}D_10000_0.0_1.0_1.0_1{0}_euclidean.txt".format(str(i).zfill(2),dim_data)))

	comp_file_list_list.append(comp_file_list)

if adaptive_binning==True:
        if PLOT: number_of_splits_list = [3]
        else: number_of_splits_list = [1,2,3,4,5,6,7,8,9,10]
	adaptive_binning_chisquared_2sam.chi2_adaptive_binning_wrapper(orig_title, orig_name, dim_list, comp_file_list_list,number_of_splits_list,systematics_fraction)
else:
	single_no_bins_list=[2,3,5]
	adaptive_binning_chisquared_2sam.chi2_regular_binning_wrapper(orig_title, orig_name, dim_list, comp_file_list_list,single_no_bins_list,systematics_fraction)

