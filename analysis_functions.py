import numpy as np
from lmfit.lineshapes import gaussian, dho, lorentzian

def convolve_peaks(x, y, sigma=1, func=gaussian):
    """Convolve data based on normalized peak function.
    This is slow, but handles non-uniform x well.
    """
    yout = np.zeros_like(y)
    for xval, yval in zip(x, y):
        yout += func(x, amplitude=yval, center=xval, sigma=sigma)

    yout = yout*np.sum(y)/np.sum(yout)
    return yout


def bose(x, kBT):
    """Return a 1-dimensinoal Bose factor function
    kBT should be in the same units as x
    bose(x, kBT) = 1 / (1 - exp(-x / kBT) )
    n.b. kB = 8.617e-5 eV/K
    """
    return np.real(1./ (1 - np.exp(-x / (kBT)) +0.001*1j ))



def paramagnon(x, amplitude=1, center=0, sigma=.1, res=.1, kBT=.1):
    """Damped harmonic oscillator convolved with resolution
    Parameters
    ----------
    x : array
        independent variable
    amplitude: float
        peak height
    center : float
        Plot of osciallator -- not the same as the peak
    sigma : float
        damping
        This corresponds to HWHM when sigma<<center
    res : float
        Resolution -- sigma parameter of Gaussian resolution used
        for convolution. FWHM of a gaussian is 2*np.sqrt(2*np.log(2))=2.355
    kBT : float
        Temperature for Bose factor.
        kBT should be in the same units as x
        n.b. kB = 8.617e-5 eV/K
    Form of equation from https://journals.aps.org/prb/pdf/10.1103/PhysRevB.93.214513
    (eq 4)
    """
    chi = (2*x*amplitude*sigma*center /
           ((x**2 - center**2)**2 + (x*sigma)**2 ))

    return convolve_peaks(x, chi*bose(x, kBT), sigma=res, func=gaussian)


def lorz(x, amplitude=1, center=0, sigma=.1, res=.1, kBT=.1):
    """Antisymmetrized lorentzian
    Parameters
    ----------
    x : array
        independent variable
    amplitude: float
        peak height
    center : float
        Plot of osciallator -- not the same as the peak
    sigma : float
        damping
        This corresponds to HWHM when sigma<<center
    res : float
        Resolution -- sigma parameter of Gaussian resolution used
        for convolution. FWHM of a gaussian is 2*np.sqrt(2*np.log(2))=2.355
    kBT : float
        Temperature for Bose factor.
        kBT should be in the same units as x
        n.b. kB = 8.617e-5 eV/K
    Form of equation from https://journals.aps.org/prb/pdf/10.1103/PhysRevB.93.214513
    (eq 4)
    """
    chi = (lorentzian(x, amplitude=amplitude, center=center, sigma=sigma)
           - lorentzian(x, amplitude=amplitude, center=-center, sigma=sigma))

    return convolve_peaks(x, chi*bose(x, kBT), sigma=res, func=gaussian)
