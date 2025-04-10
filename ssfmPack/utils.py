import matplotlib.pyplot as plt 
from matplotlib import cm 
import numpy as np
import json

def convertdB(pulse, cutoff):
    P = np.abs(pulse)**2
    P[P<1e-100] = 1e-100 
    P = 10*np.log10(P) 
    P[P<cutoff] = cutoff 
    
    return P
def saveplot(path):
    plt.savefig(path,
                bbox_inches = 'tight',
                transparent = True,
                pad_inches = 0)

def plotPulseContour(Z, T, pulse, path, cutoff = -30):
    '''Contour plotting of the pulse propagation'''
    fig, ax = plt.subplots(figsize = (8,4))
    ax.set_title('Pulse Evolution (dB scale)')
    
    P = convertdB(pulse, cutoff)
    
    surf = ax.contourf(Z,T,P, levels = 40)
    ax.set_ylabel('Time [ps]')
    ax.set_xlabel('Distance [km]')
    cbar = fig.colorbar(surf, ax = ax)
    if path:
        saveplot(path)

def plotLine(loss_records, time_records, path):
    '''Plot the loss and the runtime during model training'''
    fig, ax = plt.subplots(2,1)
    ax[0].plot(loss_records)
    ax[0].set_title(f'Recorded Training Loss')
    ax[0].set_yscale('log') 
    
    ax[1].plot(time_records[1:])
    ax[1].set_title(f'Recorded Training Time')
    plt.tight_layout()
    if path:
        saveplot(path)
    
def updateJSON(model_name, data, path):
    with open(path, "r") as file:
        try:
            loaded = json.load(file)
        except json.JSONDecodeError:
            loaded = {}  # Reinitialize as empty dictionary

        loaded[model_name] = data
    with open(path, "w") as file:
        json.dump(loaded, file, indent = 4)
    print('Data Updated!')
    
