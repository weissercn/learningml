import adaptive_binning_chisquared_2sam
import os

systematics_fraction = 0.01 
#dim_list = [1,2,3,4,5,6,7,8,9,10]
#dim_list = [2,4,10]
adaptive_binning=True
CPV = True

#param_list = [0.130, 0.132,0.133,0.134,0.1345,0.135,0.1355,0.136,0.137,0.1375,0.138,0.139,0.14]
param_list = [0.130, 0.132,0.133,0.134,0.1345,0.135,0.1355,0.136,0.137,0.1375,0.138,0.139,0.14]
#param_list = [0.130,0.132,0.133,0.134,0.135,0.136,0.13625,0.13675, 0.137,0.14]
#param_list = [0.125,0.130,0.132,0.133,0.134,0.135,0.14]
#param_list = [0.1345,0.1355,0.136,0.137,0.138,0.139] 
param_name_list = ["alphaSvalue"]
dim_data = 8

if CPV:	
	orig_name="event_shapes_final_state_lower_level_scale_uniform_syst_"+str(systematics_fraction).replace(".","_")+"_"
	#orig_name="event_shapes_original_fill01_syst_"+str(systematics_fraction).replace(".","_")+"_"
	orig_title="Thrust syst{} adaptbin".format(systematics_fraction)

else: 
	orig_name="event_shapes_final_state_lower_level_scale_uniform_syst_"+str(systematics_fraction).replace(".","_")+"_"
	#orig_name="event_shapes_original_fill01_syst_"+str(systematics_fraction).replace(".","_")+"_"
	orig_title="Thrust syst{} adaptbin".format(systematics_fraction)
        param_list= [0.1365]

comp_file_list_list = []
for param in param_list:
	comp_file_list=[]
	for i in range(100):
		if CPV: 
			comp_file_list.append((os.environ['monash']+"/GoF_input_final_state/GoF_input_udsc_monash_final_state_lower_level_{1}.txt".format(dim_data,i), os.environ['monash']+"/GoF_input_final_state/GoF_input_udsc_"+str(param)+"_"+param_name_list[0]+"_final_state_lower_level_{1}.txt".format(dim_data,i)))


		else: 
			comp_file_list.append((os.environ['monash']+"/GoF_input_final_state/GoF_input_udsc_monash_final_state_lower_level_{1}.txt".format(dim_data,i), os.environ['monash']+"/GoF_input_final_state/GoF_input_udsc_monash_final_state_lower_level_1{1}.txt".format(dim_data,str(i).zfill(2))))

	comp_file_list_list.append(comp_file_list)

if adaptive_binning==True:
	number_of_splits_list = [1,2,3,4,5,6,7,8,9,10]
	#number_of_splits_list = [3]
	adaptive_binning_chisquared_2sam.chi2_adaptive_binning_wrapper(orig_title, orig_name, param_list, comp_file_list_list,number_of_splits_list,systematics_fraction)
else:
	single_no_bins_list=[2,3,5]
	adaptive_binning_chisquared_2sam.chi2_regular_binning_wrapper(orig_title, orig_name, param_list, comp_file_list_list,single_no_bins_list,systematics_fraction)
