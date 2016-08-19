import numpy as np
import math
import sys 
import os
sys.path.insert(0,os.environ['learningml']+'/GoF/')
import classifier_eval
from sklearn.svm import SVC

# Write a function like this called 'main'
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    #comp_file_list=[(os.environ['MLToolsDir']+"/Dalitz/dpmodel/data/data_optimisation.0.0.txt",os.environ['MLToolsDir']+"/Dalitz/dpmodel/data/data_optimisation.200.1.txt")]
    #comp_file_list=[(os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/accept_reject/legendre_data/data_sin1diff_5_and_5_periods10D_1000points_optimisation_sample_0.txt",os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/accept_reject/legendre_data/data_sin1diff_5_and_6_periods10D_1000points_optimisation_sample_0.txt")]
    #comp_file_list=[(os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_redefined_4D_1000_0.6_0.2_0.1_optimisation_0.txt",os.environ['MLToolsDir']+"/Dalitz/gaussian_samples/gaussian_same_projection_on_each_axis/gauss_data/gaussian_same_projection_on_each_axis_redefined_4D_1000_0.6_0.2_0.075_optimisation_0.txt")]    
    param_name = 'pTmin'
    param_dict = {'alphaSvalue': 0.125, 'pTmin': 0.8}
    comp_file_list=[(os.environ['monash']+"/GoF_input/GoF_input_udsc_monash_optimisation.txt" , os.environ['monash']+"/GoF_input/GoF_input_udsc_{}_{}_optimisation.txt".format(str(param_dict[param_name]),param_name))]

    name = "monash_"+param_name

    clf = SVC(C=params['aC'],gamma=params['agamma'],probability=True, cache_size=7000)

    result= classifier_eval.classifier_eval(name="svm_"+name,comp_file_list=comp_file_list,clf=clf,mode="spearmint_optimisation")

    with open("svm_optimisation_values_"+name+".txt", "a") as myfile:
        myfile.write(str(params['aC'][0])+"\t"+ str(params['agamma'][0])+"\t"+str(result)+"\n")
    return result
