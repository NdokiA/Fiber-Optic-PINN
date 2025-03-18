import matplotlib.pyplot as plt 
from matplotlib import cm 
import numpy as np
import json

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
    
    P = ssfm.convertdB(pulse, cutoff)
    
    surf = ax.contourf(Z,T,P, levels = 40)
    ax.set_ylabel('Time [ps]')
    ax.set_xlabel('Distance [km]')
    cbar = fig.colorbar(surf, ax = ax)
    saveplot(path)

def updateJSON(model_name, data):
    with open("result/report.json", "r") as file:
        try:
            loaded = json.load(file)
        except json.JSONDecodeError:
            loaded = {}  # Reinitialize as empty dictionary

        loaded[model_name] = data
    with open("result/report.json", "w") as file:
        json.dump(loaded, file, indent = 4)
    print('Data Updated!')