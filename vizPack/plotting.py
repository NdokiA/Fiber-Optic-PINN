import matplotlib.pyplot as plt 
from matplotlib import cm 
import numpy as np
from ssfmPack import ssfm

def saveplot(path):
    plt.savefig(path,
                bbox_inches = 'tight',
                transparent = True,
                pad_inches = 0)
    
def plotPulseContour(Z, T, pulse, path, cutoff = -30):
    '''Contour plotting of the pulse propagation'''
    fig, ax = plt.subplots()
    ax.set_title('Pulse Evolution (dB scale)')
    
    P = ssfm.getPower(pulse)
    P[P<1e-100] = 1e-100
    P = 10*np.log10(P)
    P[P<cutoff] = cutoff
    
    surf = ax.contourf(Z,T,P, levels = 40)
    ax.set_ylabel('Time [ps]')
    ax.set_xlabel('Distance [km]')
    cbar = fig.colorbar(surf, ax = ax)
    saveplot(path)
    