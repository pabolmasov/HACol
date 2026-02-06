from numpy import *
import numpy.ma as ma

from matplotlib import gridspec

import os
import sys
import glob

import h5py

# from scipy.optimize import root_scalar
# from scipy.integrate import simpson
# from scipy.integrate import cumulative_trapezoid as cumtrapz

from os.path import exists

import matplotlib
from matplotlib.pyplot import *

import subsonic as sub
import bassun as BS

cmap = 'viridis'

mass1 = 1.5 # NS mass in Solar masses
xim = 0.5 # magnetosphere radius in RA
rstar = 6.77159 / mass1 * 10./10. # NS radius in GM/c^2 for physical R = 10km
realxirad = 1.5

afac = 0.25
drrat = 0.25

# mdot is mass accretion rate in Eddington units LEdd/c^2, assuming kappa = 0.35cm^2/g
# mu30 is magnetic moment in 10^{30}G cm^3 units

def RAlf(mdot, mu30):
    # Alfven radius in GM/c^2 units
    return 1577.74 * mu30**(4./7.)/mdot**(2./7.)/mass1**(10./7.)

def Rsph(mdot):
    return 1.5 * mdot

def theta0(mdot, mu30):
    return arcsin((sqrt(rstar / RAlf(mdot, mu30)/xim)))

def lcor(mdot, mu30):
    # luminosity of the subsonic flow in Eddington units
    cth = cos(theta0(mdot, mu30))
    return afac / drrat * (2. * log((1.+cth)/(1.-cth))-3.*cth)


def shooting_beta(th0, k, verbose = True):
    '''
    restores inner f by choosing proper fout
    '''
    
    theta_out = pi/2.
    
    lf1 = -2.0 ; lf2 = 2.0 ; tol = 1e-10 # brackets and tolerance for lg(fout)
    
    theta, fint1, u1 = sub.uint(th0, 10.**lf1, k, theta_out = theta_out, firstpoint = True)
    theta, fint2, u2 = sub.uint(th0, 10.**lf2, k, theta_out = theta_out, firstpoint = True)

    if verbose:
        print("fout = ", 10.**lf1, ": f(0) = ", fint1, "; u[-1] = ", u1)
        print("fout = ", 10.**lf2, ": f(0) = ", fint2, "; u[-1] = ", u2)

    # magnetic energy density ratio logarithm
    umagrat0 = -12. * log(sin(th0)) + log(1.+3.*cos(th0)**2)
    umagrat_out = -12. * log(sin(theta_out)) + log(1.+3.*cos(theta_out)**2)
    umagrat = umagrat0 - umagrat_out
    
    ucrit1 = u1- umagrat -log(3.)
    ucrit2 = u2 - umagrat - log(3.)

    while (abs(lf2-lf1) >  tol ):
        lf = (lf1+lf2)/2.
        theta, fint, u = sub.uint(th0, 10.**lf, k, theta_out = theta_out, firstpoint = True)
        ucrit = u - umagrat - log(3.)
        if verbose:
            print("fout = ", 10.**lf, ": f(0) = ", fint)
        if ((ucrit*ucrit1) >= 0.):
            lf1 = lf
        else:
            lf2 = lf

    fout = 10.**((lf1+lf2)/2.)
            
    theta, fint, u = sub.uint(th0, 10.**lf, k, theta_out = theta_out, firstpoint = True)
            
    return fint / 0.75 * sin(th0)**2, fout

def maincycle():

    mdot1 = 10.0 ; mdot2 = 1000.0 ; nmdot = 122
    mdot = (mdot2/mdot1)**(arange(nmdot)/double(nmdot-1))*mdot1
    muar1 = 0.003 ; muar2 = 30. ; nmu = 121
    mu = (muar2/muar1)**(arange(nmu)/double(nmu-1))*muar1
    
    mdotar, muar = meshgrid(mdot, mu)

    karr = afac / drrat * RAlf(mdotar, muar) * xim / mdotar

    beta_fint = copy(mdotar) * 0.
    beta_BS =  copy(mdotar) * 0.

    fint = copy(mdotar)
    xis = copy(mdotar)
    
    print(shape(beta_fint))
    
    for kmdot in arange(nmdot):
        for kmu in arange(nmu):
            th0 =  theta0(mdot[kmdot], mu[kmu])
            delta0 = sin(th0)/sqrt(1.+3.*cos(th0)**2)*rstar*drrat
            umag0 = (1.+3.*cos(th0)**2)/(rstar*mass1)**6 * 5.545e12 * mu[kmu]**2 # code units
            beta_tmp, fint_tmp = shooting_beta(th0, karr[kmu, kmdot], verbose  = False)
            beta_fint[kmu, kmdot] = beta_tmp
            fint[kmu, kmdot] = fint_tmp
            # BSgamma = (4.*pi*afac/drrat) * sqrt(1.+3.*cos(th0)**2)/mdot[kmdot] / (realxirad/1.5) # Across/\delta^2 / mdot *rstar /(realxirad/1.5) rstar??
            BSgamma = (afac/drrat) * sqrt(1.+3.*cos(th0)**2)/mdot[kmdot] / (realxirad/1.5) * rstar # / mass1 
            BSeta = (8./21./sqrt(2.) * umag0 * 3. * (realxirad/1.5))**0.25*sqrt(delta0)/(rstar)**0.125
            print("theta0 = ", th0)
            print("BSgamma = ", BSgamma)
            print("BSeta = ", BSeta)
            # print( BS.xis(BSgamma, BSeta)/(xim*RAlf(mdot[kmdot], mu[kmu])))
            # print("x0 = ",  maximum(1.1,(xim*RAlf(mdot[kmdot], mu[kmu])/rstar)))
            # xistmp, beta_tmp = squeeze(BS.xis(BSgamma, BSeta, ifbeta = True))
            # print(xistmp, beta_tmp)
            xis[kmu, kmdot], beta_BS[kmu,kmdot] = squeeze(BS.xis(BSgamma, BSeta, x0 = maximum(2.,(xim*RAlf(mdot[kmdot], mu[kmu])/rstar)), ifbeta = True))
            # /(xim*RAlf(mdot[kmdot], mu[kmu])/rstar)
            print("xi = ", BS.xis(BSgamma, BSeta, x0 = sqrt(xim*RAlf(mdot[kmdot], mu[kmu])/rstar)), "xie = ", xim*RAlf(mdot[kmdot], mu[kmu])/rstar)
            # ii = input('beta')

    betarad_approx = 1.-lcor(mdotar, muar)/mdotar
    betarad = beta_fint

    excluded = betarad < 1.-rstar/RAlf(mdotar, muar)/xim

    betarad_masked = ma.masked_array(betarad, mask = (betarad > 1.) | (betarad < 0.))

    clf()
    fig, ax = subplots()
    pc = ax.pcolormesh(muar, mdotar, log10(xis/(xim*RAlf(mdotar, muar)/rstar)))
    cb = colorbar(pc)
    cb.set_label(r'$\log_{10}R_{\rm shock}/R_{\rm m}$')
    contour(muar, mdotar, xis, [0.0], colors = 'k', linewidths = 4.)
    ax.set_xlabel(r'$\mu$, $10^{30}$G cm$^3$') ; ax.set_ylabel(r'$\dot{M}c^2/L_{\rm Edd}$')
    ax.set_xscale('log') ; ax.set_yscale('log')
    fig.set_size_inches(10.,8.)
    savefig('blimits_xis.png')
    
    clf()
    fig, ax = subplots()
    pc = ax.pcolormesh(muar, mdotar, log10(fint), vmin = -2., vmax = 2.0)
    cb = colorbar(pc)
    cb.set_label(r'$\log_{10}f_{\rm out}$')
    contour(muar, mdotar, fint, [1.0], colors = 'k', linewidths = 4.)
    ax.set_xlabel(r'$\mu$, $10^{30}$G cm$^3$') ; ax.set_ylabel(r'$\dot{M}c^2/L_{\rm Edd}$')
    ax.set_xscale('log') ; ax.set_yscale('log')
    fig.set_size_inches(10.,8.)
    savefig('blimits_fint.png')
    
    
    clf()
    fig, ax = subplots()
    pc = ax.pcolormesh(muar, mdotar, log10(karr), vmin = -2., vmax = 2.0)
    cb = colorbar(pc)
    cb.set_label(r'$\log_{10}k$')
    contour(muar, mdotar, karr, [1.0], colors = 'k', linewidths = 4.)
    ax.set_xlabel(r'$\mu$, $10^{30}$G cm$^3$') ; ax.set_ylabel(r'$\dot{M}c^2/L_{\rm Edd}$')
    ax.set_xscale('log') ; ax.set_yscale('log')
    fig.set_size_inches(10.,8.)
    savefig('blimits_karr.png')
    
    
    clf()
    fig, ax = subplots()
    pc = ax.pcolormesh(muar, mdotar, betarad, vmin = 0., vmax = 1.0)
    cb = colorbar(pc, ax = ax, orientation='horizontal', location = 'top', aspect = 50)
    cb.set_label(r'$\beta$', fontsize=14)
    ax.contour(muar, mdotar, betarad, [2./3.], colors = 'k', linewidths = 4.)
    c = ax.contour(muar, mdotar, betarad, [0.5, 0.75, 0.9, 0.99], colors = 'k', linewidths = 1.)
    # c0 = ax.contour(muar, mdotar, betarad_approx, [0.5, 0.75, 0.9, 0.99], colors = 'k', linewidths = 1., linestyles=':')
    c0 = ax.contour(muar, mdotar, beta_BS, [0.3, 0.4, 0.5, 0.75, 0.9, 0.99], colors = 'k', linewidths = 1., linestyles=':')
    c06 = ax.contour(muar, mdotar, beta_BS, [2./3.], colors = 'k', linestyles=':', linewidths = 4.)
    
    csph = ax.contour(muar, mdotar, Rsph(mdotar)/(RAlf(mdotar, muar)*0.5), levels = [1], colors = 'r')
    csphs = ax.contour(muar, mdotar, Rsph(mdotar)/sqrt(RAlf(mdotar, muar)*0.5), levels = [1], colors = 'r', linestyles = 'dotted')
    
    formatstring = ['0.5', '0.75', '0.9', '0.99']
    fmt = {}
    for l, s in zip(c.levels, formatstring):
        fmt[l] = s
        
    ax.clabel(c, c.levels, inline_spacing = 0, fmt = fmt,fontsize=14, rightside_up = True)
    ax.clabel(c0, c0.levels, inline_spacing = 0, fmt = fmt,fontsize=14, rightside_up = True)
    cs1 = ax.contourf(muar, mdotar, excluded, [-1, 0., 1., 2.], linestyles = 'solid', hatches=['', '//'], alpha = 1.0)
    cs1.set_facecolor('none')
    cs1.set_edgecolor('w')
    # bar.set_edgecolor('k')
    # cs2 = ax.contourf(muar, mdotar, betarad, [-10., 0., 1.], hatches=['\\', '', '||'], alpha = 1.0)
    cs2 = ax.contourf(muar, mdotar, fint, [-10., 0., 1., 1000.], hatches=['\\', '', '--', '||'], alpha = 1.0)
    cs2.set_facecolor('none')
    cs2.set_edgecolor('w')
    cs3 = ax.contour(muar, mdotar, log10(xis/(xim*RAlf(mdotar, muar)/rstar)), [0.], colors='w', linewidths=3, linestyles='dashed') #, hatches=['||', '', '--'], alpha = 1.0)
    # cs3.set_facecolor('none')
    # cs3.set_edgecolor('w')
    cs4 = ax.contourf(muar, mdotar, RAlf(mdotar, muar)*0.5/rstar, [-1., 1.], colors='w', linewidths=3, hatches=['||', '', '--'], alpha = 1.0)
    cs4.set_facecolor('none')
    cs4.set_edgecolor('k')
    ax.set_xlabel(r'$\mu$, $10^{30}$G cm$^3$',fontsize=16) ; ax.set_ylabel(r'$\dot{M}c^2/L_{\rm Edd}$', fontsize=16)
    ax.set_xscale('log') ; ax.set_yscale('log')
    secax = ax.secondary_yaxis('right', functions = (lambda x: 3.78269e-09 * x, lambda x: x/3.78269e-09))
    secax.set_ylabel(r'$\dot{M}$, $M_\odot \, \rm yr^{-1}$', fontsize = 16, loc='center')
    plot([0.1], [300.], '*k', markersize = 20.0, mfc = 'none')
    tick_params(labelsize=14, length=3, width=1., which='major')
    tick_params(labelsize=14, length=1, width=1., which='minor')
    fig.set_size_inches(12.,8.)
    savefig('blimits.png')
    
    
    clf()
    fig = figure()
    scatter(betarad[fint<1.0], karr[fint<1.], c = log10(muar[fint<1.0]))
    scatter(betarad[fint>1.0], karr[fint>1.0], c = log10(muar[fint>1.0]), facecolors='none')
    cb = colorbar()
    cb.set_label(r'$\log_{10}\mu_{30}$')
    yscale('log')
    ylabel(r'$k$') ; xlabel(r'$\beta$')
    xlim(0.,1.)
    fig.set_size_inches(8.,6.)
    savefig('blimits_skar.png')

# compare two solutions, BS and sobso
def solcompare(solmdot = 30., solmu = 0.1):

    theta_out = pi/2.
    
    solk = afac / drrat * RAlf(solmdot, solmu) * xim / solmdot
    th0 =  theta0(solmdot, solmu)
    delta0 = sin(th0)/sqrt(1.+3.*cos(th0)**2)*rstar*drrat
    umag0 = (1.+3.*cos(th0)**2)/(rstar*mass1)**6 * 5.545e12 * solmu**2 # code units
    beta_tmp, fint_tmp = shooting_beta(th0, solk, verbose  = False)
    
    BSgamma = (afac/drrat) * sqrt(1.+3.*cos(th0)**2)/solmdot / (realxirad/1.5) * rstar 
    BSeta = (8./21./sqrt(2.) * umag0 * 3. * (realxirad/1.5))**0.25*sqrt(delta0)/(rstar)**0.125

    xis, beta_BS = squeeze(BS.xis(BSgamma, BSeta, x0 = maximum(2.,(xim*RAlf(solmdot, solmu)/rstar)), ifbeta = True))

    print("subsonic beta = ", beta_tmp)
    print("BS beta = ", beta_BS)

    xBS, vBS, uBS = BS.BSsolution(BSgamma, BSeta)

    theta, fint, usu = sub.uint(th0, fint_tmp, solk, theta_out = theta_out, firstpoint = False)
    xsu = (sin(theta)/sin(th0))**2

    umagsu = exp(-12. * log(sin(theta)/sin(th0)) + log(1.+3.*cos(theta)**2)-log(1.+3.*cos(th0)**2))
    umagBS = 1./xBS**6
    
    usu = exp(usu - usu[0]) #  -12. * log(sin(theta)/sin(th0)) + log(1.+3.*cos(theta)**2)-log(1.+3.*cos(th0)**2))
    print(usu)
    
    clf()
    plot(xsu, usu/umagsu, 'k-', label = 'subsonic')
    plot(xBS, uBS/umagBS, 'r:', label = 'BS solution')
    legend()
    xlabel(r'$R/R_*$')   ;  ylabel(r'$u/u_{\rm mag}$')
    yscale('log')
    savefig('solc.png')
