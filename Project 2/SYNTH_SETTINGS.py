import numpy as np

N_D1 = 100
N_CLASS = 3

SCALE  = [[1,1],[1,2],[1.5,1],[0.8,1]]
THETA  = [0,-0.25*np.pi, 0, 0.2*np.pi]
#TRANS  = [[0,0],[5,0],[10,-1],[-5,1]]  #colinear
TRANS  = [[0,0],[6,0],[3,4],[-3,4]]
COLOR  = ['red','darkgreen','blue','cyan']; 
MARKER = ['.','s','+','*']

FORMAT = '%.3f','%.3f','%d'
LINESTYLE = [['r--','r-'],['b--','b-'],['c--','c-'],['k--','k-']]

DAT_FILENAME = 'fort-synthetic2.data'
FIG_FILENAME = 'fort-synthetic2.png'

