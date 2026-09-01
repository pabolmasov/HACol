import matplotlib
from matplotlib import rc
from matplotlib import axes
from matplotlib import interactive, use
from matplotlib import ticker
from numpy import *
import numpy.ma as ma
from pylab import *
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.interpolate import interp1d
from scipy.optimize import minimize, root, root_scalar
import glob
import re
import os

import time
import g_units as units

#Uncomment the following if you want to use LaTeX in figures 
# rc('font',**{'family':'serif'})
# rc('mathtext',fontset='cm')
# rc('mathtext',rm='stix')
# rc('text', usetex=True)
# #add amsmath to the preamble
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"]

close('all')
ioff()
use('Agg')

import configparser as cp
conffile = 'globals.conf'
config = cp.ConfigParser(inline_comment_prefixes="#")
config.read(conffile)

ifplot =  config['DEFAULT'].getboolean('ifplot')

if ifplot:
    import plots
    
formatsequence = ['k-', 'g:', 'b--', 'r-.']

# K = 0.5

omega = 0.0

# f0 = 10.0

import bassun as bs
import geometry as geo


def dtint(x, b):
    '''
    calculates the integral needed for the sound travel time
    b = Re/Rstar * beta - 1/sin^2\theta0
    '''
    print("x = ", x[0], " to ", x[-1]) # shd be from 0 to cos(theta0)
    print("b = ", b)
    y = (1.+3.*x**2) * (1.-x**2)/(1. - b * (1.-x**2))
    print (y, (y<=0.).sum())
    
    dt = simpson(sqrt(y), x=x)

    return dt

def lowk(theta, theta0, rrat, beta, umag):
    ufac = 1.+(1./tan(theta)**2-1./tan(theta0)**2)*rrat/beta
    ufac = minimum((ufac/ufac[-1])**(4.)*umag/umag[-1], 3.) 
    return ufac

def fcfun(x0, K):
    nx = 1e6
    x = x0 * arange(nx)/double(nx)
    return 1.5 * simpson(exp(K * x * (1.+x**2)) * x / (1.-x**2)**2,x=x) #, initial = 0)

def intfun(x, f0, K):
    # formal solution for f = normalized (u/rho). f0 is the value at x = cos(theta) = cos(theta_out) \simeq 0
    return exp(-K * x * (1.+x**2)) * (f0 + 1.5 * cumulative_trapezoid(exp(K * x * (1.+x**2)) * x / (1.-x**2)**2,x=x, initial = 0.))

def luintfun(f, x):
    # int I from 0 to x. lnu = I(x0) - I(x)
    return- 3. * cumulative_trapezoid(x / f * (3.*omega**2*(1.-x**2)**2 - 2./(1.-x**2)**2), x=x, initial=0.)

def uint(theta0, fout, K, firstpoint=False, theta_out = pi/2.):

    nth = 10000
    
    theta = (theta_out-theta0) * arange(nth)/double(nth-1)+ theta0
    x = cos(theta[::-1])
    # theta = theta[::-1]

    fint = intfun(x, fout, K)
    # fint = maximum(fint, 0.)

    luint = luintfun(fint,x)[::-1]

    luint = luint-luint[-1] # so that the function (ln u) is 0 @ theta_out
    
    if firstpoint:
        return theta[0], fint[-1], luint[0] # theta0, f at the surface, ln u at the surface (should be =3)
    else:
        return theta, fint[::-1], luint

def fzero_solution(conf = 'SUBSO', snapshot = None, thsnapshots = None, iffit = False):
    '''
    thsnapshots are thprofile.dat files produced by acompace. If they are set, plotting snapshot is suppressed
    '''

    # reading the data:
    outdir = config[conf].get('outdir')
    rstar = config[conf].getfloat('rstar')
    m1 = config[conf].getfloat('m1')
    mu30 = config[conf].getfloat('mu30')
    mdot = config[conf].getfloat('mdot') * 4. * pi # in code units
    xifac = config[conf].getfloat('xifac')
    xirad = config[conf].getfloat('xirad')
    r_e = config[conf].getfloat('r_e_coeff') * (mu30**2/mdot)**(2./7.)*m1**(-10./7.) * xifac # magnetosphere radius
    afac = config[conf].getfloat('afac')
    drrat = config[conf].getfloat('drrat')
    tscale = config[conf].getfloat('tscale')
    # Dthick = config[conf].getfloat('Dthick')

    theta0 = arcsin(sqrt(rstar/r_e)) # polar cap radius
    theta_out = arcsin(1./sqrt(1.+drrat**2))
    k = afac / drrat * r_e / (mdot/4./pi) # k parameter

    print("mdot = ", mdot / 4./pi)
    
    print("r_e = ", r_e, " = ", r_e/rstar, "R*")
    
    print("theta0 = ", theta0)
    print("k = ", k)
    #    ii = input('K')
    # print("expected fout = ", 0.75 * Dthick**2)
    
    if snapshot is not None:
        linesT = loadtxt(snapshot) # 'vcomp/tireoutT.dat'
        rT = linesT[:,0]  ;  uT = linesT[:,3] ; rhoT = linesT[:,1]
        thetaT = arcsin(sqrt(rT*rstar/r_e))
        vT = linesT[:,2] ; rhoT = linesT[:,1]
        mdotT = -(vT * rhoT)[-1]*4.*pi*r_e**2*drrat # units?
        fT = uT/rhoT * rT[-1]
        print("measured mdot = ", mdotT/4./pi)
        print("Re = ", rT[-1]*rstar)
        print("fT = ",fT)
        #mdot = mdotT
        #k = afac / drrat * r_e / m1 / mdot # k parameter
        #print("internal k = ", k)

        print("theta_min in the data = ", thetaT.min())
        theta0 = thetaT.min()
        theta_out = thetaT.max()

    if thsnapshots is not None:
        nsnaps = size(thsnapshots)

        alias = 100

        tharlist = [] ; farlist = [] ; uarlist = [] ; dfarlist = [] ; duarlist = []

        for k in arange(nsnaps):
            linesT = loadtxt(thsnapshots[k])
            tharlist.append(linesT[::alias,0]) ; farlist.append(linesT[::alias,1]) ; uarlist.append(linesT[::alias,2]) ;  dfarlist.append(linesT[::alias,3]) ; duarlist.append(linesT[::alias,4])
            print("fout = ", linesT[-1,1])
            print("theta(data) = ", tharlist)
            f0 = linesT[-1,1]
            
    # logarithmic bracketing
    # minimal fout should be 3/4 of the int, because we do not want f_surface to change sign

    # ii = input('f0')
        
    umagrat0 = -12. * log(sin(theta0)) + log(1.+3.*cos(theta0)**2) # + log(3.)
    umagrat_out = -12. * log(sin(theta_out)) + log(1.+3.*cos(theta_out)**2) # + log(3.)

    if (iffit):

        lf1 = -1.0 ; lf2 = 1.0 ; tol = 1e-10

        theta, fint1, u1 = uint(theta0, 10.**lf1, k, theta_out = theta_out, firstpoint = True)
        theta, fint2, u2 = uint(theta0, 10.**lf2, k, theta_out = theta_out, firstpoint = True)
        # (1.+3.*cos(theta)**2)/(1.+3.*cos(theta0)**2)*(sin(theta0)/sin(theta))**6
        
        # ii = input("theta")
        
        print("fout = ", 10.**lf1, ": f(0) = ", fint1, "; u[-1] = ", u1)
        print("fout = ", 10.**lf2, ": f(0) = ", fint2, "; u[-1] = ", u2)

        ucrit1 = u1-umagrat0+umagrat_out -log(3.)
        ucrit2 = u2-umagrat0+umagrat_out - log(3.)
        print("fout = ", 10.**lf1, ": f(0) = ", fint1, "; u[-1] = ", ucrit1)
        print("fout = ", 10.**lf2, ": f(0) = ", fint2, "; u[-1] = ", ucrit2)
        
        ii = input("theta")
        # same sign is not expected
        if (ucrit1*ucrit2 >= 0.):
            return 0.
        
        while (abs(lf2-lf1) >  tol ):
            lf = (lf1+lf2)/2.
            theta, fint, u = uint(theta0, 10.**lf, k, theta_out = theta_out, firstpoint = True)
            ucrit = u-umagrat0+umagrat_out - log(3.)
            print("fout = ", 10.**lf, ": f(0) = ", fint)
            if ((ucrit*ucrit1) >= 0.):
                lf1 = lf
            else:
                lf2 = lf

        f0 = 10.**lf2

    # f0 = 0.75
        
    theta, fint, u = uint(theta0, f0, k, theta_out = theta_out)
    umagrat = -12. * log(sin(theta)/sin(theta_out)) + log(1.+3.*cos(theta)**2) - log(1.+3.*cos(theta_out)**2) # + log(3.)

    umag = (1.+3.*cos(theta)**2)/(1.+3.*cos(theta0)**2)*(sin(theta0)/sin(theta))**12
    
    u = exp(u-umagrat)
    
    print("experimental beta = ", 4./3. * fint[0] * sin(theta0)**2)
    beta = 4./3. * fint[0] * sin(theta0)**2
    print("theta = ", theta)
    print("sin^-2(theta0) = ", 1./sin(theta0)**2)
    print("f = ", 0.75 /sin(theta0)**2 * beta + 0.75 * (1./sin(theta)**2-1./sin(theta0)**2))
    print("f = ", fint[-1] + 0.75 * (1./sin(theta)**2-1./sin(theta_out)**2))
    print("fout = ", fint[-1])
    
    if snapshot is not None:
        umagsnap = (1.+3.*cos(thetaT)**2)/(1.+3.*cos(theta0)**2)*(sin(theta0)/sin(thetaT))**12 
        umagsnap0 = (1.+3.*cos(thetaT)**2)/sin(thetaT)**12 

    print("U/Umag = ",u)

    fan = 0.75  + 0.75 * (1./sin(theta)**2-1./sin(theta_out)**2)
    #    fan = fint[0] +  0.75 * (1./sin(theta)**2-1./sin(theta0)**2)
    
    unorm_lowk = (fan/fan[0])**(4.) * umag[0] / umag
    
    if thsnapshots is not None:
        # print(len(tharlist))
        # ii = input('far')
        plots.subfint(theta, fint, u, tharlist, uarlist, farlist, duTnorm = duarlist, dfT = dfarlist, unorm_lowk = unorm_lowk)
    else:
        plots.subfint(theta, fint, u, thetaT, uT, fT * umagsnap0/umagsnap[0], unorm_lowk = unorm_lowk)
    
    # ASCII output:
    fout = open('uint.dat', 'w+')
    fout.write('# theta f  u/umag\n')
    nx = size(theta)
    for k in arange(nx):
        fout.write(str(theta[k])+' '+str(fint[k])+' '+str(u[k])+'\n')
        
    fout.flush()
    fout.close()
    fout = open('uint_theta.dat', 'w+')
    fout.write('# theta f  u/umag\n')
    nx = size(theta)
    for k in arange(nx):
        xi = (sin(theta[k])/sin(theta0))**2
        fout.write(str(theta[k])+' '+str(fint[k])+' '+str((u)[k])+'\n')
        
    fout.flush()
    fout.close()

    # calculating the sound travel time
    geofile = outdir+"/geo.dat"
    gr, gtheta, alpha, across, l, delta = geo.gread(geofile)
    perimeter = 2. * (across/delta + 2.*delta)
    BSgamma = (across/delta**2)[0]/mdot*rstar / (xirad/1.5)
    # umag is magnetic pressure
    b12 = 2.*mu30*(rstar*m1/6.8)**(-3) # dipolar magnetic field on the pole, 1e12Gs units
    umag = b12**2*2.29e6*m1
    
    BSeta = (8./21./sqrt(2.)*umag*3. * (xirad/1.5))**0.25*sqrt(delta[0])/(rstar)**0.125
    print("BSgamma = "+str(BSgamma))
    print("BSeta = "+str(BSeta))
    xs, BSbeta = bs.xis(BSgamma, BSeta, x0=r_e/rstar, ifbeta = True)
    print("BSbeta = ", BSbeta)
    beta = 1. # no radiation losses
    dtt = dtint(cos(theta)[::-1], (1.-beta)/sin(theta0)**2)
    print("integral = ", dtt)
    dtt = simpson(sqrt(1.+3.*cos(theta)**2)/sqrt(fint), x = -cos(theta))
    print("integral = ", dtt)
    print("Re = ", r_e)
    # t = sqrt(3.) * r_e**1.5/sqrt(m1) * dtt * tscale * m1 # sound time ? needs checking
    # print("T = ", t, "s")#?

    print("fosc = ", 1./t, "Hz")
    
    # return lf

# usage:
# fzero_solution(conf='ASOL_slowT4', snapshot='vcomp/tireoutT.dat')
# fzero_solution(conf = 'SUBSO', snapshot = None, thsnapshots = ['out_subso/thprofile.dat', 'out_bottom/thprofile.dat', 'out_zero/thprofile.dat'])

# 
