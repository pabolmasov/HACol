from scipy.interpolate import interp1d
from numpy import *

import plots
import neu

# aimed to reproduce Fig 3 from Beaudet et al 1967
        
def demo():
    
    mass1 = 1.4
    
    temps = [1e10, 1e9, 3.85e8, 1e8] # K
        
    rho1 = [1e6,1e3,1e2, 1] ; rho2 = [1e14, 1e11, 1e10, 1e8] # g/cm^3
    nrho = 100
    
    ymin = [1e17, 1e8, 1e3, 0.01]
    ymax = [1e28, 1e19, 1e15, 1e9]

    rhoscale = 1.93474e-5

    for k in arange(size(temps)):
        rho = (rho2[k]/rho1[k])**(arange(nrho)/double(nrho-1))*rho1[k]

        rhonorm = rho / rhoscale / mass1
        u = (temps[k]/3.89e7)**4. * mass1

        qa, qh, ql = neu.Qnu(rhonorm, u, me = 0.85, separate = True)
    
        qa *= 3.53e21 / mass1**2
        qh *= 3.53e21 / mass1**2
        ql *= 3.53e21 / mass1**2
        
        plots.someplots(rho/0.85, [qa, qh, ql], name='beaudettest{:01d}'.format(k), formatsequence=['k-', 'r--', 'g:'], xtitle = r'$\rho,\, {\rm g\, cm^{-3}}$', ytitle =  r'$Q, \ {\rm erg\, cm^{-3}\, s^{-1}}$', multix = False, inchsize=[5,5], xlog = True, ylog=True, yrange=[ymin[k], ymax[k]])
