from scipy.interpolate import interp1d
from numpy import *

qfloor = -30.
ffloor = 1e-18
# mass1 = 1.4 # link to conf!!

# import plots
    
# routines for neutrino loss computation
# uses formulae from Beaudet et al 1967

def gfun(l):
    #   Eq. (27b)
    return 1. - 13.04 * l**2 + 133.5 * l**4 + 1534.0 * l**6 + 918.6 * l**8

def ffun(xi, l, coeff):
    # general expression for the "f" factor (Eq. 26)
    a0,a1,a2,b1,b2,b3,c = coeff
    return maximum((a0 + a1 * xi + a2 * xi**2) / (xi**3 + b1/l + b2/l**2 + b3/l**3) * exp(-c*xi), ffloor)

def Qnu(rho, u, me = 0.85, separate = True, mass1 = 1.4):
    # general expression for volume losses,Eq. 27a

    tempfac = 0.00656608 # conversion from energy density to kT/me c^2
    l = tempfac * (u/mass1)**0.25
    
    # rhofac = 1.93474e-14 # rhoscale in 1e9 g/cm**3 units
    # r9 = (rho * rhofac * mass1) / me # rho/me in 1e9g/cm**3 units
    rhofac = 1.93474e-5 # rhoscale in g/cm**3 units
    r1 = (rho * rhofac * mass1) / me # rho/me in 1e9g/cm**3 units

    xi = (rho/me*mass1)**(1./3.) / l * 2.68457e-05
    
    lnqscale = 49.6156 - 2. * log(mass1) # ln(3.53e21)
    
    # plasma neutrino:
    qpl = ffun(xi, l, (2.32e-7, 8.449e-8, 1.787e-8, 2.581e-2, 1.734e-2, 6.99e-4, 0.56457))
    qpl = log(qpl) + 3. * log(r1)
    w = (qpl > qfloor)
    w0 =(qpl <= qfloor)
    if (w.sum() > 0):
        qpl[w] = exp(qpl[w] - lnqscale)
    if (w0.sum() > 0):
        qpl[w0] = 0.
        
    # photoneutrino:
    qph = ffun(xi, l, (4.886e10, 7.58e10, 6.023e10, 6.29e-3, 7.483e-3, 3.061e-4, 1.5654))
    qph = log(qph) + log(r1) + 5. * log(l)
    w = (qph > qfloor)
    w0 =(qph <= qfloor)
    if (w.sum() > 0):
        qph[w] = exp(qph[w] - lnqscale)
    if (w0.sum() > 0):
        qph[w0] = 0.
        
    # annihilation:
    qpa = ffun(xi, l, (6.002e19, 2.084e20, 1.872e21, 9.383e-1, -4.141e-1, 5.829e-2, 5.5924))
    qpa = log(qpa) + log(gfun(l)) - 2./l
    w = (qpa > qfloor)
    w0 =(qpa <= qfloor)
    if (w.sum() > 0):
        qpa[w] = exp(qpa[w] - lnqscale)
    if (w0.sum() > 0):
        qpa[w0] = 0.
        
    if separate:
        return qpa, qph, qpl
    else:
        return qpa+qph+qpl
       
