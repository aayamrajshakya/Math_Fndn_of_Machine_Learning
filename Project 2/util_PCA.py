import numpy as np; import numpy.linalg as la
import matplotlib.pyplot as plt; import seaborn as sbn
from numpy import diag,dot
from matplotlib.patches import Ellipse
from scipy.linalg import svd
import math
from SYNTH_SETTINGS import *; from util_fig import *;
import warnings; warnings.filterwarnings("ignore")

def class_center(X,y):
    """ compute the class centers: dim=(c,d)"""
    CC = []
    for c in range(len(set(y))):
        CC.append(list(np.mean(X[y==c],axis=0)))
    return np.array(CC)

def VT2angle(VT):
    return math.acos(np.abs(VT[0,0])) *(180/np.pi);

def aniso_dist2(Xc,s,VT,C):
    """ Measure anisotropic distances for data points in a class """
    AD2 = []
    N,d = Xc.shape
    for i in range(N):
         AD2.append( sum(((Xc[i]-C).dot(VT[j])/s[j])**2 for j in range(d)) )
    return np.array(AD2)

def dataCellipses(X,y,savename):
    N,d = X.shape; nclass = len(set(y))
    if d==2:  #for figures
        ELL = [];
        #fig, p = plt.subplots()  #Open new plots
        p = plt.subplot(111)      #Reuse instances
    for c in range(nclass):
        Xc = X[y==c]; CC = np.mean(Xc,axis=0)
        U, s, VT = svd(Xc-CC,full_matrices=False)
        AD2 = aniso_dist2(Xc,s,VT,CC)
        if d==2:  #for figures
           plt.scatter(Xc[:,0],Xc[:,1],s=15,c=COLOR[c],marker=MARKER[c])
           angle = VT2angle(VT);
           eta = np.sqrt(max(AD2))*2
           ELL.append(Ellipse(CC, s[0]*eta, s[1]*eta, angle=angle ) )

    if d==2:  #for figures
        for c, e in enumerate(ELL):
            e.set_clip_box(p.bbox); e.set_alpha(0.33)
            e.set_facecolor(COLOR[c])
            p.add_artist(e)
        #ymin,ymax = np.min(X[:,1]), np.max(X[:,1])
        #plt.ylim([int(ymin)-1,int(ymax)+1])
        plt.title('MVEE')
        myfigsave(savename)
        plt.show(block=False); plt.pause(5)

def data_denoise(X,y,portion=0.9):
    N,d = X.shape; nclass = len(set(y))
    for c in range(nclass):
        Xc = X[y==c]; y1=y[y==c]; CC = np.mean(Xc,axis=0)
        U, s, VT = svd(Xc-CC,full_matrices=False)
        AD2 = aniso_dist2(Xc,s,VT,CC)
        #-- Below, begin denoising ------------------------
        sort_index = np.argsort(AD2)
        #ADs = AD2[sort_index]; print(ADs)
        Xcs = Xc[sort_index]
        npick = round(Xc.shape[0]*portion)
        if c==0:
            Xout = Xcs[:npick]; yout = y1[:npick];
        else:
            Xout = np.concatenate((Xout,Xcs[:npick]))
            yout = np.concatenate((yout,y1[:npick]))
    return Xout, yout.reshape(len(Xout))
