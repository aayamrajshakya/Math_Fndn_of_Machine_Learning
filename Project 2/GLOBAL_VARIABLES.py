import numpy as np
import matplotlib.pyplot as plt

N_D1 = 100
FORMAT = '%.3f','%.3f','%d'

SCALE = [[1,1],[1,2],[1.5,1]]; TRANS = [[0,0],[6,0],[3,4]]
#SCALE = [[1,1],[1,1],[1,1]]; TRANS = [[0,0],[4,0],[8,0]]
THETA = [0,-0.25*np.pi, 0]
COLOR = ['r','b','c']
MARKER = ['.','s','+','*']
LINESTYLE = [['r--','r-'],['b--','b-'],['c--','c-']]

N_CLASS = len(SCALE)

DAT_FILENAME = 'synthetic.data'
FIG_FILENAME = 'synthetic-data.png'
FIG_INTERPRET = 'synthetic-data-interpret.png'

def myfigsave(figname):
    plt.savefig(figname,bbox_inches='tight', dpi=300)
    print(' saved: %s' %(figname))
