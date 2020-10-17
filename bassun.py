import numpy.random
from numpy.random import rand
from numpy import *
from scipy.optimize import fsolve
from scipy.special import expn
from scipy.integrate import simps

def fxis(x, gamma, eta, n):
    return 1.+exp(gamma*x)*(x*expn(2,gamma)-expn(2,gamma*x)) - eta * gamma**0.25 * x**((n+0.5)/4.)

def xis(gamma, eta, n=3, x0=20., ifbeta = False):
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
    x = fsolve(fxis, x0, args=(gamma, eta, n), maxfev = 1000, xtol=1e-10)
    #    print(fxis(x, gamma, eta, n))
    if ifbeta:
        beta = 1.-gamma*exp(gamma)*(expn(1,gamma)-expn(1, gamma*x))
        return x, beta
    else:
        return x

def dtint(gamma, xs, beta = None):
    '''
    calculates the (normalized) time for a sound wave to travel from the surface to the shock front (or back)
    input: BS gamma, BS beta, position of the shock in rstar units
    '''
    nxs = size(xs)

    if nxs <= 1:
        nx = 10000
        x = (xs-1.)*arange(nx)/double(nx-1)+1.

        if beta is None:
            beta = 1.-gamma*exp(gamma)*(expn(1,gamma)-expn(1, gamma*xs))
    
        csq = 1./3. * exp(gamma * x) * (expn(2,gamma*x)/x + beta * exp(-gamma) - expn(2,gamma)) / x**3

        dt = simps(1./sqrt(csq), x=x)
    else:
        dt = zeros(nxs)
        for k in arange(nxs):
            dt[k] = dtint(gamma, xs[k], beta = beta)
        
    return dt
