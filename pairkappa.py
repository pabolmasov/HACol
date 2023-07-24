from numpy import *

rhodconst = 4.333e+11 # degeneracy density normalized according to globals.conf
# sqrt(2./pi**3)*(mc/hbar)**3*GMsun * sigmaT/c**2
mecc = 510.994 # electron mass in keV

def multiplicity(temp, rho):
    # gives the number of positrons in annihilation equilibrium of a mildly relativistic gas; temp shd be in me c^2 units
    # see Landau-Lifshitz V (stat physics), p. 344
    return sqrt(1. + 4. * (rhodconst/rho)**(2) * temp**3 * exp(-2./temp))

def kappafac(urad, rho):
    # cross-section increase due to pair production;
    # input urad and rho shd be normalized by mass (urad/mass1), (rho/mass1)
    
    temp = (urad)**(0.25) * 3.35523 # keV
    prfac = multiplicity(temp / mecc, rho)
    
    return prfac

'''
import matplotlib
from matplotlib import rc
from matplotlib import axes
from matplotlib import interactive, use
from matplotlib import ticker
from pylab import *

x = (10./0.01)**(arange(100)/double(99))*0.01
y = multiplicity(x,5000.)

print(y)

clf()
plot(x*mecc, y,'k-')
xscale('log') ; yscale('log')
xlabel(r'$T$, keV') ; ylabel(r'$\varkappa/\varkappa_{\rm T}$')
savefig('pairkappa.png')
'''
