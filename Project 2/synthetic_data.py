import numpy as np
import matplotlib.pyplot as plt
from SYNTH_SETTINGS import *; from util_fig import *

def generate_data(n,scale,theta):
    # Normally distributed around the origin
    x = np.random.normal(0,1, n); y = np.random.normal(0,1, n)
    P = np.vstack((x, y)).T
    # Transform
    sx,sy = scale
    S = np.array([[sx,0],[0,sy]])
    c,s = np.cos(theta), np.sin(theta)
    R = np.array([[c,-s],[s,c]]).T  #T, due to right multiplication
    return P.dot(S).dot(R)

def synthetic_data():
    N=0
    plt.figure()
    for i in range(N_CLASS):
        scale = SCALE[i]; theta = THETA[i]; N+=N_D1
        D1 = generate_data(N_D1,scale,theta) +TRANS[i]
        D1 = np.column_stack((D1,i*np.ones([N_D1,1])))
        if i==0: DATA = D1
        else:    DATA = np.row_stack((DATA,D1))
        plt.scatter(D1[:,0],D1[:,1],s=15,c=COLOR[i],marker=MARKER[i])

    #xmin,xmax = np.min(DATA[:,0]), np.max(DATA[:,0])
    ymin,ymax = np.min(DATA[:,1]), np.max(DATA[:,1])
    plt.ylim([int(ymin)-1,int(ymax)+1])

    plt.title('Synthetic Data: N = '+str(N))
    np.savetxt(DAT_FILENAME,DATA,delimiter=',',fmt=FORMAT)
    print('   saved: %s' %(DAT_FILENAME))
    myfigsave(FIG_FILENAME)

    if __name__ == '__main__':
        plt.show(block=False); plt.pause(10)
    else:
        return DATA

if __name__ == '__main__':
    synthetic_data()
