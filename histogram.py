import torch
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from math import radians
def save_histogram(path,result_path):
    dir_name = ['Base','Odin']
    for stype in dir_name:
        IN = np.loadtxt(path+'confidence_'+stype+'_In.txt')
        OUT = np.loadtxt(path+'confidence_'+stype+'_Out.txt')
        IN_mean= IN.mean()
        IN_var = IN.var()
        IN_normal = (IN-IN_mean)/IN_var
        OUT_normal = (OUT-IN_mean)/IN_var
        IN_min,IN_max = IN_normal.min(), IN_normal.max()
        OUT_min,OUT_max = OUT_normal.min(), OUT_normal.max()
        min_ = min(IN_min,OUT_min)
        max_ = max(IN_max,OUT_max)
        gap = (max_-min_)/100
        bins = np.arange(min_,max_+1,gap)
        IN_hist, bins = np.histogram(IN_normal,bins)
        OUT_hist, bins = np.histogram(OUT_normal,bins)
        plt.hist(IN_normal,bins)
        plt.hist(OUT_normal,bins)
        plt.axis([min_,max_,0,max(max(IN_hist),max(OUT_hist))])
        plt.savefig(result_path+stype+'.png')
        plt.show()