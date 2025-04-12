import numpy as np; import pandas as pd
import matplotlib.pyplot as plt; import seaborn as sbn
import numpy.linalg as la
from numpy import diag,dot
from scipy.linalg import svd
from util_CLESS import *; from save_fig import *
np.set_printoptions(suppress=True)

def class_center(X,y):
    """ compute the class centers: dim=(c,d)"""
    N, d = X.shape; nclass = len(set(y))
    CC = np.zeros([nclass,d])
    for c in range(nclass):
        CC[c] = np.mean(X[y==c],axis=0)
    return CC

def distance_table(CC):
    c,d = CC.shape
    Dtable = np.zeros([c,c])
    for i, j in np.ndindex((c,c)):
        Dtable[i,j] = la.norm(CC[i,:]-CC[j,:])
    return Dtable/np.sqrt(d)

def angle_table(CC):
    c,d = CC.shape
    #CC -= np.sum(CC,axis=0)/c
    dot_product = np.dot(CC,CC.T)
    norms = la.norm(CC,axis=1)
    norm_product = np.outer(norms,norms)
    cosine = dot_product/norm_product
    np.fill_diagonal(cosine,1)  #fix round error
    return np.arccos(cosine)*(180/np.pi)

def class_stat(X,y,d0,datafile):
    print('[1;37mclass_stat:[m')
    CC = class_center(X,y)
    c,d = CC.shape

    Dtable = distance_table(CC[:,:d0]); Atable = angle_table(CC[:,:d0])
    dist_angle_save(Dtable,Atable,datafile,
          'out/fort-'+datafile+'-mutial-dist-angle0.png')
    #if d>d0:
    #    Dtable = distance_table(CC); Atable = angle_table(CC)
    #    dist_angle_save(Dtable,Atable,datafile,
    #         'out/fort-'+datafile+'-mutial-dist-angle1.png')

