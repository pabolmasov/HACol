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

def fcfun(x0, K):
    nx = 1e6
    x = x0 * arange(nx)/double(nx)
    return 0.75 * simpson(exp(-K * x * (1.+x**2)) * x / (1.-x**2)**2,x=x) #, initial = 0)

def intfun(x, f0, K):
    # formal solution for f = normalized (u/rho). f0 is the value at x = cos(theta) = 0
    return exp(K * x * (1.+x**2)) * (f0 - 0.75 * cumulative_trapezoid(exp(-K * x * (1.+x**2)) * x / (1.-x**2)**2,x=x, initial = 0))

def luintfun(f, x):
    # int I from 0 to x. lnu = I(x0) - I(x)
    return -3. * cumulative_trapezoid(x / f * (3.*omega**2*(1.-x**2)**2 - 2./(1.-x**2)**2), x=x, initial=0.)

def uint(theta0, f0, K, firstpoint=False):

    nth = 10000
    theta = (pi/2.-theta0) * arange(nth)/double(nth-1)+ theta0
    # theta = theta[::-1]

    fint = intfun(cos(theta[::-1]), f0, K)[::-1]

    luint = luintfun(fint,cos(theta)[::-1])[::-1]

    luint = luint-luint[0]
    
    if firstpoint:
        return theta[0], fint[0], luint[-1]
    else:
        return theta, fint, exp(luint)
    
def fcritshow():

    theta0 = [0.03,0.1,0.3]
    nth0 = size(theta0)
    # nth = 1000
    # theta = (pi/2.-theta0) * (arange(nth)+0.5)/double(nth)+ theta0

    nk = 100 ;  kmin = 0.001 ; kmax = 10.
    
    kk = (kmax/kmin)**(arange(nk)/double(nk-1)) * kmin
    
    fc = zeros(nk)
    fc0 = zeros(nk)
    
    nx = 1000
    # print(cos(theta0))
    
    clf()
    for i in arange(nth0):
        for j in arange(nk):
        # print("k = ", kk[j])
            fc[j] = fcfun(cos(theta0[i]), kk[j]) # -intfun(cos(theta)[::-1], 0., k[j])[-1]
            print("k = ", kk[j], " ;  fc = ", fc[j], " ~ ", fc0)

        plot(kk, fc, formatsequence[i], label=r'$\theta_0 = {:g}$'.format(theta0[i]), linewidth = 2)
        
        a0 = (cos(theta0[i])/sin(theta0[i]))**2/2.
        a1 = -(1.+sin(theta0[i])**2)/sin(theta0[i])**2 * cos(theta0[i]) + log((1.-cos(theta0[i]))/(1.+cos(theta0[i])))
        plot(kk, (a0+a1*kk) * 1.5, 'r:')
        
    fc0 = 1./kk**2 * 1.5

    plot(kk, fc0, 'k:')

    ylim(kk.max()**(-2.), (cos(theta0)/sin(theta0)).max()**2)

    xlabel(r'$k$')
    ylabel(r'$f_{\rm c}$')
    yscale('log')
    xscale('log')
    legend()
    savefig('fc.png')
    savefig('fc.pdf')

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

    theta0 = sqrt(rstar/r_e) # polar cap radius
    k = afac / drrat * r_e / m1 / mdot # k parameter

    print("r_e = ", r_e, " = ", r_e/rstar, "R*")
    
    print("theta0 = ", theta0)
    print("k = ", k)
    
    if snapshot is not None:
        linesT = loadtxt(snapshot) # 'vcomp/tireoutT.dat'
        rT = linesT[:,0]  ;  uT = linesT[:,3]
        thetaT = arcsin(sqrt(rT*rstar/r_e))
    
    # we want to find the f0 that produces f(theta0)=0
    # logarithmic bracketing
    lf1 = -1.; lf2 = 3. ; tol = 1e-7

    theta, fint1, u1 = uint(theta0, 10.**lf1,k, firstpoint=True)
    theta, fint2, u2 = uint(theta0, 10.**lf2,k, firstpoint=True)

    umagrat = -12. * log(sin(theta0)) + log(1.+3.*cos(theta0)**2) + log(3.)
    u1 += umagrat ; u2 += umagrat
    # (1.+3.*cos(theta)**2)/(1.+3.*cos(theta0)**2)*(sin(theta0)/sin(theta))**6

    print(umagrat)
    # ii = input("theta")

    print("f0 = ", 10.**lf1, ": f(0) = ", fint1, "; u[-1] = ", u1)
    print("f0 = ", 10.**lf2, ": f(0) = ", fint2, "; u[-1] = ", u2)

    # same sign is not expected
    if (u1*u2 >= 0.):
        return 0.

    while (abs(lf2-lf1) >  tol ):
        lf = (lf1+lf2)/2.
        theta, fint, u = uint(theta0, 10.**lf,k, firstpoint=True)
        u += umagrat 
        print("f0 = ", 10.**lf, ": f(0) = ", fint)
        if ((u*u1) >= 0.):
            lf1 = lf
        else:
            lf2 = lf

    theta, fint, u = uint(theta0, 10.**lf2,k)

    print("beta = ", 0.75 * fint[0] * sin(theta0)**2)
    print("f0 = ", fint[-1])

    umag = (1.+3.*cos(theta)**2)/(1.+3.*cos(theta0)**2)*(sin(theta0)/sin(theta))**12 / 3.
    if snapshot is not None:
        umagsnap = (1.+3.*cos(thetaT)**2)/(1.+3.*cos(theta0)**2)*(sin(theta0)/sin(thetaT))**12 / 3.

    clf()
    subplot(211)
    plot(theta, fint, 'k-')
    plot(theta, fint[-1]*exp(k*cos(theta)*(1.+cos(theta)**2)), 'r:')
    xlabel(r'$\theta$')
    ylabel(r'$f(\theta)$')
    # yscale('log')
    subplot(212)
    plot(theta, u/umag, 'k-')
    plot(theta, u*0.+3., 'r:')
    plot(theta, u*0.+1., 'r:')
    plot(thetaT, uT, 'b--')
    xlabel(r'$\theta$')
    ylabel(r'$u(\theta)/u_{\rm mag}(\theta)$')
    yscale('log')
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

def uishow():

    theta0 = 0.03
    f0 = 301.0
    k = 0.01
    
    theta, fint, u = uint(theta0, f0,k)

    print("f0 = ", fint[0])
    
    fint_appro = exp(k * cos(theta)) * (f0 - 0.75 * (1./tan(theta)**2+((1.+sin(theta)**2)/sin(theta)**2*cos(theta)+log((1.-cos(theta))/(1.+cos(theta)))) * k ))
    
    clf()
    subplot(211)
    plot(theta, fint, 'k.')
    plot(theta, fint_appro, 'rx')
    xlabel(r'$\theta$')
    ylabel(r'$f(\theta)$')
    yscale('log')
    subplot(212)
    plot(theta, u, 'k.')
    plot(theta, (1.+3.*cos(theta)**2)/(1.+3.*cos(theta0)**2)*(sin(theta0)/sin(theta))**6, 'r:')
    xlabel(r'$\theta$')
    ylabel(r'$u(\theta)/u_0$')
    yscale('log')
    savefig('uint.png')



def fplot():
    clf()
    plot(theta, intfun(cos(theta)), 'k.')
    plot(theta, exp(K * cos(theta) * (1.+cos(theta)**2))*f0, 'g:')
    plot(theta, 1./sin(theta)**2, 'r-')
    xlabel(r'$\theta$')
    ylabel(r'$f(\theta)$')
    yscale('log')
    savefig('fint.png')
