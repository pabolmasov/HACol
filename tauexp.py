from scipy.interpolate import interp1d
from numpy import *

# routines for accurate and stable representation of 1-e^-tau, (1-e^-tau)/tau

# smooth factor for optical depth
def taufun(tau, taumin, taumax):
    '''
    calculates 1-exp(-x) in a reasonably smooth way trying to avoid round-off errors for small and large x
    '''
    wtrans = where(tau<taumin)
    wopaq = where(tau>taumax)
    wmed = where((tau>=taumin) & (tau<=taumax))
    tt = copy(tau)
    if(size(wtrans)>0):
        tt[wtrans] = (tau[wtrans]+abs(tau[wtrans]))/2.
    if(size(wopaq)>0):
        tt[wopaq] = 1.
    if(size(wmed)>0):
        tt[wmed] = 1. - exp(-tau[wmed])
    return tt

def tratfac(x, taumin, taumax):
    '''
    a smooth and accurate version of (1-e^{-x})/x
    '''
    xmin = taumin ; xmax = taumax # limits the same as for optical depth
    nx = size(x)
    tt = copy(x)
    if nx>1:
        w1 = where(x<= xmin) ;  w2 = where(x>= xmax) ; wmed = where((x < xmax) & (x > xmin))
        if(size(w1)>0):
            tt[w1] = 1.
        if(size(w2)>0):
            tt[w2] = 1./x[w2]
        if(size(wmed)>0):
            tt[wmed] = (1.-exp(-x[wmed]))/x[wmed]
        wnan=where(isnan(x))
        if(size(wnan)>0):
            tt[wnan] = 0.
            print("x = "+str(x[wnan]))
            ip = input('trat')
        return tt
    else:
        if x <= xmin:
            return 1.
        else:
            if x>=xmax:
                return 1./x
            else:
                return (1.-exp(-x))/x
            
# smoothed step for gravpotential
def sstep(x, xmin, xmax):
    '''
    calculates (1-exp(-x))^2 with smooth limits
    '''
    wlow = where(x<xmin)
    whigh = where(x>xmax)
    wmed = where((x>=xmin) & (x<=xmax))
    xx = copy(x)
    if(size(wlow)>0):
        xx[wlow] = x[wlow]**2 * (x[wlow] > 0.)
    if(size(whigh)>0):
        xx[whigh] = 1.
    if(size(wmed)>0):
        xx[wmed] = (1.-exp(-x[wmed]))**2
    return xx

def scalarstep(x, xmin, xmax):
    '''
    calculates 1-exp(-x) with smooth limits for a scalar angument
    '''
    if x<0.:
        return 0.
    if x<xmin:
        return x
    if x>xmax:
        return 1.
    return 1.-exp(-x)
