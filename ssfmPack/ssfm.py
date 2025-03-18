import numpy as np
from scipy.fftpack import fft, ifft, fftshift, ifftshift, fftfreq

def getFreqRange(time: np.ndarray) -> np.ndarray:
    """
    Get frequency range from given time range

    Args:
        time (np.ndarray): time array

    Returns:
        np.ndarray: freq array
    """    
    d = time[1]-time[0]
    return fftshift(fftfreq(len(time), d = d))

def getTimeRange(f: np.ndarray) -> np.ndarray:
    """
    Get time range from given frequency range

    Args:
        f (np.ndarray): freq array

    Returns:
        np.ndarray: time array
    """     
    d = f[1]-f[0]
    return fftshift(fftfreq(len(f), d = d))

def getPower(pulse: np.ndarray) -> np.ndarray:
    """
    Get power of the pulse by taking the absolute square

    Args:
        pulse (np.ndarray): pulse array

    Returns:
        np.ndarray: power array
    """    
    return np.abs(pulse)**2

def getEnergy(time: np.ndarray, pulse: np.ndarray) -> float:
    """
    Returns the energy of the pulse via trapezoid integration. 

    Args:
        time (np.ndarray): time array
        pulse (np.ndarray): pulse array

    Returns:
        float: energy of the pulse
    """    
    return np.trapz(getPower(pulse), time)

def getSpectrum(time: np.ndarray, pulse: np.ndarray) -> np.ndarray:
    """
    Do Fourier Transformation to change the pulse (time domain) into spectrum (frequency domain)
    Asserts error if energy is not conserved up to a certain error during transformation

    Args:
        time (np.ndarray): time array
        pulse (np.ndarray): pulse array

    Returns:
        np.ndarray: spectrum array
    """    
    dt = time[1]-time[0]
    f = getFreqRange(time)
    spectrum = fftshift(fft(pulse))*dt
    
    pulseEnergy = getEnergy(time, pulse)
    spectrumEnergy = getEnergy(f, spectrum)
    err = np.abs((pulseEnergy/spectrumEnergy-1))
    assert(err < 1e-6), f'Energy unconserved error: {err:.3e}'
    
    return spectrum

def getPulse(f: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
    """
    Do Inverse Fourier Transformation to change the spectrum (freq. domain) into pulse (time domain)
    Asserts error if energy is not conserved up to a certain error during transformation

    Args:
        f (np.ndarray): frequency array
        spectrum (np.ndarray): spectrum array

    Returns:
        np.ndarray: pulse array
    """    
    time = getTimeRange(f)
    dt = time[1]-time[0] 
    pulse = ifft(ifftshift(spectrum))/dt 
    
    pulseEnergy = getEnergy(time, pulse)
    spectrumEnergy = getEnergy(f, spectrum)
    err = np.abs(pulseEnergy/spectrumEnergy-1)
    assert(err < 1e-6), f'Energy unconserved error: {err:.3e}'
    
    return pulse

def convertdB(pulse, cutoff = -30): 
    
    pulse = getPower(pulse)
    pulse[pulse < 1e-100] = 1e-100 
    pulse = 10*np.log10(pulse) 
    pulse[pulse < cutoff] = cutoff
    
    return pulse

def SSFM(t_array: np.ndarray, z_array: np.ndarray, init_pulse: np.ndarray, 
         alpha: float, beta2: float, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Split-Step Fourier Method (SSFM) to solve the Nonlinear Schrodinger Equation (NLSE)

    Args:
        t_array (np.ndarray): time array
        z_array (np.ndarray): spatial array
        init_pulse (np.ndarray): initial pulse
        alpha (float): nonlinear refractive index
        beta2 (float): fiber optic's dispersion coefficient
        gamma (float): fiber optic's attenuation coefficient

    Returns:
        tuple[np.ndarray, np.ndarray]: pulse matrix, spectrum matrix
    """    
    #Initialize pulse-spectrum matrix
    f = getFreqRange(t_array)
    dz = z_array[1]-z_array[0]
    
    z_dim = len(z_array)
    t_dim = len(t_array)
    pulseMatrix = np.zeros((z_dim, t_dim), dtype = complex)
    spectrumMatrix = np.zeros((z_dim+1, t_dim), dtype = complex)
    
    pulse = init_pulse.astype(complex)
    pulseMatrix[0] = pulse 
    spectrumMatrix[0] = getSpectrum(t_array, init_pulse)
    
    #Calculating SSFM
    omega = 2*np.pi*f
    dispersionLoss = np.exp((1j*beta2/2*(omega)**2 - alpha/2)*dz)
    nonlinearity = 1j*gamma*dz 
    
    for n in range(1,z_dim):
        pulse = pulse * np.exp(nonlinearity*getPower(pulse))
        spectrum = getSpectrum(t_array, pulse)*dispersionLoss 
        pulse = getPulse(f, spectrum)
        
        pulseMatrix[n,:] = pulse
        spectrumMatrix[n,:] = spectrum
    
    return pulseMatrix, spectrumMatrix
    