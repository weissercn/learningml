import adaptive_binning_chisquared_2sam
import os

systematics_fraction = 0.01 
dim_list = [1,2,3,4,5,6,7,8,9,10]
adaptive_binning=True
CPV  = True
PLOT = True


if CPV:	
	orig_name="sin_5_6_10000_CPV_syst_"+str(systematics_fraction).replace(".","_")+"_"
	orig_title="sin 5 and 6 syst{}".format(systematics_fraction)

else: 
        orig_name="sin_5_5_10000_noCPV_syst_"+str(systematics_fraction).replace(".","_")+"_"
        orig_title="sin 5 and 5 syst{}".format(systematics_fraction)

if PLOT:
        dim_list = [2,6,10]
        orig_name="plot_"+orig_name
        orig_title= "Plot "+orig_title
        sample_list_typical= [47, 72, 34]
        sample_list= [[item,item+1] for item in sample_list_typical]

else:
        sample_list = [range(100)]*len(dim_list)


comp_file_list_list = []
for dim_data_index, dim_data in enumerate(dim_list):
	comp_file_list=[]
        for i in sample_list[dim_data_index]:
		if CPV: 
			comp_file_list.append((os.environ['learningml']+"/GoF/data/accept_reject/sin1diff_data/data_sin1diff_5_and_5_periods{0}D_10000_sample_{1}.txt".format(dim_data,i), os.environ['learningml']+"/GoF/data/accept_reject/sin1diff_data/data_sin1diff_5_and_6_periods{0}D_10000_sample_{1}.txt".format(dim_data,i)))
		else: 
			comp_file_list.append((os.environ['learningml']+"/GoF/data/accept_reject/sin1diff_data/data_sin1diff_5_and_5_periods{0}D_10000_sample_{1}.txt".format(dim_data,i), os.environ['learningml']+"/GoF/data/accept_reject/sin1diff_data/data_sin1diff_5_and_5_periods{0}D_10000_sample_1{1}.txt".format(dim_data,str(i).zfill(2))))

	comp_file_list_list.append(comp_file_list)

if adaptive_binning==True:
	if PLOT: number_of_splits_list = [3]
	else: number_of_splits_list = [1,2,3,4,5,6,7,8,9,10]
	adaptive_binning_chisquared_2sam.chi2_adaptive_binning_wrapper(orig_title, orig_name, dim_list, comp_file_list_list,number_of_splits_list,systematics_fraction)
else:
	single_no_bins_list=[2,3,5]
	adaptive_binning_chisquared_2sam.chi2_regular_binning_wrapper(orig_title, orig_name, dim_list, comp_file_list_list,single_no_bins_list,systematics_fraction)

