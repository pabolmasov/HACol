import numpy.random
from numpy.random import rand
from numpy import *
from scipy.optimize import fsolve
from scipy.special import expn

def fxis(x, gamma, eta, n):
    return 1.+exp(gamma*x)*(x*expn(2,gamma)-expn(2,gamma*x)) - eta * gamma**0.25 * x**((n+0.5)/4.)

def xis(gamma, eta, n=3):
    '''
    solves equation (34) from Basko&Sunyaev (1976)
    arguments: 
    gamma = c R_{NS}^3/(kappa dot{M} A_\perp * afac**2) \simeq (RNS/Rsph) * (RNS**2/across)
    eta = (8/21 * u_0 d_0 kappa / \sqrt{2GMR} c )**0.25, u_0 = B^2/8/pi
    (gamma = rstar**2/mdot/across[0]/afac**2)
    (eta = (8/21/sqrt(2)))**0.25 (umag*sqrt(rstar)*d0**2)**0.25, where d0 = (across/4./pi/rstar/sin(theta))[0]
    '''
    if((eta*gamma**0.25)<1.) | (gamma>100.):
        return nan
    x = fsolve(fxis, 2., args=(gamma, eta, n))
    #    print(fxis(x, gamma, eta, n))
    return x
