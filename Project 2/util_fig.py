import numpy as np
import matplotlib.pyplot as plt

def myfigsave(figname):
    plt.savefig(figname,bbox_inches='tight')
    print('   saved: %s' %(figname))

