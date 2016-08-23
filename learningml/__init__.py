from .text import testing_learningml
import sys
import os
os.environ["learningml"]=os.getcwd()
sys.path.insert(0, './GoF')
sys.path.insert(0, './GoF/data/gaussian_same_projection_on_each_axis')
import classifier_eval


