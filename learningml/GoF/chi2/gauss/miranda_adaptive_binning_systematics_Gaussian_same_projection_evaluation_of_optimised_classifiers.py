import adaptive_binning_chisquared_2sam
import os

systematics_fraction = 0.01 
dim_list = [2,3,4,5,6,7,8,9,10]
adaptive_binning=True
CPV = True

if CPV:	
	orig_name="gaussian_same_projection_redefined_p_value_distribution__1_0__0_9_CPV_miranda_systematics"
	orig_title="Gauss 0.1 0.9 syst0.01 adapt bin"
else: 
	orig_name="gaussian_same_projection_redefined_p_value_distribution__1_0__1_0_noCPV_miranda_systematics"
	orig_title="Gauss 0.1 0.1 syst0.01 adapt bin"

comp_file_list_list = []
for dim_data in dim_list:
	comp_file_list=[]
	for i in range(100):
		if CPV: 
			comp_file_list.append((os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_redefined_{1}D_1000_0.0_1.0_1.0_{0}.txt".format(i,dim_data),os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_redefined_{1}D_1000_0.0_1.0_0.9_{0}.txt".format(i,dim_data)))
		else: 
			comp_file_list.append((os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_redefined_{1}D_1000_0.0_1.0_1.0_{0}.txt".format(i,dim_data),os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_redefined_{1}D_1000_0.0_1.0_1.0_1{0}.txt".format(str(i).zfill(2),dim_data)))

	comp_file_list_list.append(comp_file_list)

if adaptive_binning==True:
	number_of_splits_list = [1,2,3,4,5,6,7,8,9,10]
	adaptive_binning_chisquared_2sam.chi2_adaptive_binning_wrapper(orig_title, orig_name, dim_list, comp_file_list_list,number_of_splits_list,systematics_fraction)
else:
	single_no_bins_list=[2,3,5]
	adaptive_binning_chisquared_2sam.chi2_regular_binning_wrapper(orig_title, orig_name, dim_list, comp_file_list_list,single_no_bins_list,systematics_fraction)

