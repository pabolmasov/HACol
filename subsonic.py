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

#Uncomment the following if you want to use LaTeX in figures 
rc('font',**{'family':'serif'})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
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


formatsequence = ['k-', 'g:', 'b--', 'r-.']

# K = 0.5

omega = 0.0

# f0 = 10.0

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
    return -3. * cumulative_trapezoid(x / f * (3.*omega**2*(1.-x**2)**2 - 2./(1.-x**2)**2), x=x, initial=0.)

def uint(theta0, f0, K, firstpoint=False, theta_out = pi/2.):

    nth = 10000
    
    theta = (theta_out-theta0) * arange(nth)/double(nth-1)+ theta0
    x = cos(theta[::-1])
    # theta = theta[::-1]

    fint = intfun(x, f0, K)
    fint = maximum(fint, 0.)

    luint = luintfun(fint,x)[::-1]

    luint = luint-luint[-1] # so that the function (ln u) is 0 @ theta_out
    
    if firstpoint:
        return theta[0], fint[-1], luint[0] # theta0, f at the surface, ln u at the surface (should be =3)
    else:
        return theta, fint[::-1], luint

def fzero_solution(conf = 'ASOL_slowT4', snapshot = None):
    # solve for f(theta0) = 0

    # reading the data:
    rstar = config[conf].getfloat('rstar')
    m1 = config[conf].getfloat('m1')
    mu30 = config[conf].getfloat('mu30')
    mdot = config[conf].getfloat('mdot') * 4.*pi # now it is in G M / c kappa
    xifac = config[conf].getfloat('xifac')
    r_e = config[conf].getfloat('r_e_coeff') * (mu30**2/mdot)**(2./7.)*m1**(-10./7.) * xifac # magnetosphere radius
    afac = config[conf].getfloat('afac')
    drrat = config[conf].getfloat('drrat')
    Dthick = config[conf].getfloat('Dthick')

    theta0 = sqrt(rstar/r_e) # polar cap radius
    theta_out = arcsin(1./sqrt(1.+drrat**2))
    k = afac / drrat * r_e / m1 / mdot # k parameter
    
    print("r_e = ", r_e, " = ", r_e/rstar, "R*")
    
    print("theta0 = ", theta0)
    print("k = ", k)
    print("expected f0 = ", 0.75 * Dthick**2)
    
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
    
    # we want to find the f0 that produces f(theta0)=0
    # logarithmic bracketing
    # minimal f0 should be 3/4 of the int, because we do not want f_surface to change sign
    f0 = fcfun(cos(theta0), k)
    print("f0min = ", f0)
    # ii = input('f0')
    
    lf1 = log10(f0)-2.0 ; lf2 = log10(f0)+2.0 ; tol = 1e-10

    theta, fint1, u1 = uint(theta0, 10.**lf1,k, theta_out = theta_out, firstpoint = True)
    theta, fint2, u2 = uint(theta0, 10.**lf2,k, theta_out = theta_out, firstpoint = True)

    umagrat = -12. * log(sin(theta0)) + log(1.+3.*cos(theta0)**2) # + log(3.)
    ucrit1 = u1-umagrat-log(3.) ; ucrit2 = u2-umagrat - log(3.)
    # (1.+3.*cos(theta)**2)/(1.+3.*cos(theta0)**2)*(sin(theta0)/sin(theta))**6

    print(umagrat)
    # ii = input("theta")

    print("f0 = ", 10.**lf1, ": f(0) = ", fint1, "; u[-1] = ", ucrit1)
    print("f0 = ", 10.**lf2, ": f(0) = ", fint2, "; u[-1] = ", ucrit2)

    # same sign is not expected
    if (ucrit1*ucrit2 >= 0.):
        return 0.

    while (abs(lf2-lf1) >  tol ):
        lf = (lf1+lf2)/2.
        theta, fint, u = uint(theta0, 10.**lf, k, theta_out = theta_out, firstpoint = True)
        ucrit = u-umagrat - log(3.)
        print("f0 = ", 10.**lf, ": f(0) = ", fint)
        if ((ucrit*ucrit1) >= 0.):
            lf1 = lf
        else:
            lf2 = lf

    theta, fint, u = uint(theta0, 10.**lf2,k, theta_out = theta_out)
    u = exp(u-umagrat)
    
    print("beta = ", 0.75 * fint[0] * sin(theta0)**2)
    beta = 0.75 * fint[0] * sin(theta0)**2
    print("f0 = ", fint[-1])
    
    umag = (1.+3.*cos(theta)**2)/(1.+3.*cos(theta0)**2)*(sin(theta0)/sin(theta))**12 
    if snapshot is not None:
        umagsnap = (1.+3.*cos(thetaT)**2)/(1.+3.*cos(theta0)**2)*(sin(theta0)/sin(thetaT))**12 
        umagsnap0 = (1.+3.*cos(thetaT)**2)/sin(thetaT)**12 

    print("U/Umag_out = ",(u/umag))

    clf()
    fig = figure()
    subplot(211)
    plot(theta, fint, 'k-')
    # plot(theta, fint[-1]*exp(k*cos(theta)*(1.+cos(theta)**2)), 'r:')
    plot(theta, fint[-1] + 0.75/tan(theta)**2, 'g-.')
    plot(thetaT, fT*umagsnap0, 'b--')
    xlabel(r'$\theta$')
    ylabel(r'$f(\theta)$')
    yscale('log')
    subplot(212)
    plot(theta, u/umag, 'k-')
    plot(theta, u*0.+3., 'r:')
    plot(theta, u*0.+1., 'r:')
    plot(thetaT, uT, 'b--')
    plot(theta, lowk(theta, theta0, rstar/r_e, beta, umag), 'g-.')
    #    plot(theta, lowk(theta, theta0, rstar/r_e, 0.5, umag), 'g-.')
    # plot(theta, lowk(theta, theta0, rstar/r_e, 1.0, umag), 'g-.')
    #    plot(theta, 3. * (1. + beta* (rstar/r_e) * ( 1./tan(theta0)**2-1./tan(theta)**2))**(-4.), 'k-.')
    #    plot(theta, 3. * (1. + 1.0* (rstar/r_e) * ( 1./tan(theta0)**2-1./tan(theta)**2))**(-4.), 'k-.')
    xlabel(r'$\theta$')
    ylabel(r'$u(\theta)/u_{\rm mag}(\theta)$')
    ylim(1e-1,20.)
    yscale('log')
    fig.set_size_inches(4.,6.)
    fig.tight_layout()
    savefig('uint0.png')
    savefig('uint0.pdf')

    # ASCII output:
    fout = open('uint.dat', 'w+')
    fout.write('# theta f  u/umag\n')
    nx = size(theta)
    for k in arange(nx):
        fout.write(str(theta[k])+' '+str(fint[k])+' '+str((u/umag)[k])+'\n')
        
    fout.flush()
    fout.close()
    fout = open('uint_xi.dat', 'w+')
    fout.write('# theta f  u/umag\n')
    nx = size(theta)
    for k in arange(nx):
        xi = (sin(theta[k])/sin(theta0))**2
        fout.write(str(xi)+' '+str(fint[k])+' '+str((u/umag)[k])+'\n')
        
    fout.flush()
    fout.close()

    return lf

# usage:
# fzero_solution(conf='ASOL_slowT4', snapshot='vcomp/tireoutT.dat')

