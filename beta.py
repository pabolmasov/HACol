from scipy.interpolate import interp1d
from numpy import *

# TODO: merge these values with the config file
ufloor = 1e-15
rhofloor = 1e-15

# speed of sound multiplier (see Chandrasekhar 1967 or Johnson 2008):
def Gamma1(gamma, beta):
    g1 = gamma - 1.
    return beta + 9. * g1 * (beta-4./3.)**2/(beta+12.*g1 * (1.-beta))

# calculating beta = pgas / p as a function of rho and P, and back
def Fbeta(rho, u, betacoeff):
    '''
    calculates a function of 
    beta = pg/p from rho and u (dimensionless units)
    F(beta) itself is F = beta / (1-beta)**0.25 / (1-beta/2)**0.75
    '''
    nx = size(rho)
    if nx <= 1:
        if (u*rho) > 0.:
            beta = betacoeff * rho / u**0.75
        else:
            beta = 1.
    else:
        beta = rho*0.+1.
        wpos=where(u>ufloor)
        if(size(wpos)>0):
            beta[wpos]=betacoeff * rho[wpos] / u[wpos]**0.75
    return beta 

def Fbeta_press(rho, press, betacoeff):
    '''
    calculates a function of 
    beta = pg/p from rho and pressure (dimensionless units)
    F(beta) itself is F = beta / (1-beta)**0.25
    '''
    beta = rho*0.+1.
    wpos=where(press>ufloor)
    if(size(wpos)>0):
        beta[wpos]=betacoeff * rho[wpos] / (press[wpos]*3.)**0.75 
    return beta 

def betafun_define():
    '''
    defines the function to calculate beta as a function of rho/u**0.75
    '''
    bepsilon = 1e-8 ; nb = 1e4
    b1 = 0. ; b2 = 1.-bepsilon
    b = (b2-b1)*arange(nb+1)/double(nb)+b1
    fb = b / (1.-b)**0.25 / (1.-b/2.)**0.75
    bfun = interp1d(fb, b, kind='linear', bounds_error=False, fill_value=1.)
    return bfun

def betafun_press_define():
    '''
    defines the function to calculate beta as a function of rho/p**0.75
    '''
    bepsilon = 1e-8 ; nb = 1e4
    b1 = 0. ; b2 = 1.-bepsilon
    b = (b2-b1)*arange(nb+1)/double(nb)+b1
    fb = b / (1.-b)**0.25
    bfun = interp1d(fb, b, kind='linear', bounds_error=False, fill_value=1.)
    return bfun
