import matplotlib
from matplotlib import rc
from matplotlib import axes
from numpy import *
from pylab import *
from scipy.integrate import cumtrapz
import glob

#Uncomment the following if you want to use LaTeX in figures 
rc('font',**{'family':'serif'})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
# #add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 

import hdfoutput as hdf
from globals import *

close('all')
ioff()

#############################################################
# Plotting block 
def uplot(r, u, rho, sth, v, name='outplot'):
    '''
    energy u supplemented by rest-mass energy rho c^2
    '''
    ioff()
    clf()
    fig=figure()
    plot(r, u, 'k', label='$u$',linewidth=2)
    plot(r, rho, 'r', label=r'$\rho c^2$')
    plot(r, rho*v**2/2., 'm', label=r'$\frac{1}{2}\rho v^2$')
    plot(r, rho/r, 'r', label=r'$\rho/r$', linestyle='dotted')
    plot(r, rho*0.5*(r*omega*sth)**2, 'r', label=r'$\frac{1}{2}\rho (\Omega R \sin\theta)^2$', linestyle='dashed')
    plot(r, umag*(rstar/r)**6, 'b', label=r'$u_{\rm mag}$')
    B=u*4./3.+rho*(-1./r-0.5*(r*omega*sth)**2+v**2/2.)
    plot(r, B, 'g', label='$B$', linestyle='dotted')
    plot(r, -B, 'g', label='$-B$')
    #    plot(x, y0, 'b')
    #    xscale('log')
    #    ylim(umag*((rstar/r)**6).min(), umag)
    ylim(u[u>0.].min(), u.max())
    xlabel('$r$, $GM/c^2$ units')
    yscale('log')
    xscale('log')    
    legend()
    fig.set_size_inches(4, 8)
    savefig(name+'.png')
    close()

def vplot(x, v, cs, name='outplot'):
    '''
    velocity, speed of sound, and virial velocity
    '''
    ioff()
    clf()
    plot(x, v*0., 'k:')
    plot(x, v, 'k', label='$v/c$',linewidth=2)
    plot(x, cs, 'g', label=r'$\pm c_{\rm s}/c$')
    plot(x, -cs, 'g')
    plot(x, x*0.+1./sqrt(x), 'r', label=r'$\pm v_{\rm vir}/c$')
    plot(x, x*0.-1./sqrt(x), 'r')
    xlabel('$r$, $GM/c^2$ units')
    #    yscale('log')
    xscale('log')
    ylim(-0.2,0.2)
    legend()
    savefig(name+'.png')
    close()

def splot(x, y, name='outplot'):
    '''
    so far plots some quantity S(R) 
    '''
    clf()
    plot(x, y, 'k')
    xscale('log') ; yscale('log')
    xlabel(r'$r$') ; ylabel(r'$S(R)$')
    savefig(name+'.png')
    
#########################################################################
# post-processing PDS plot:

def pdsplot(freq, pds, outfile='pds'):
    clf()
    plot(freq[1:], pds[1:], 'k')
    xscale('log') ; yscale('log')
    ylabel('PDS') ; xlabel('$f$, Hz')
    savefig(outfile+'.png')
    close()

def binplot(freq, dfreq, pds, dpds, outfile='binnedpds'):
    
    clf()
    errorbar(freq, pds, xerr=dfreq, yerr=dpds, fmt='.k')
    xscale('log') ; yscale('log')
    ylabel('PDS') ; xlabel('$f$, Hz')
    savefig(outfile+'.png')
    close()
    
def dynspec(t2,binfreq2, pds2, outfile='flux_dyns', nbin=None):

    nbin0=2
    
    lpds=log10(pds2)
    lmin=lpds[nbin>nbin0].min() ; lmax=lpds[nbin>nbin0].max()
    binfreqc=(binfreq2[1:,1:]+binfreq2[1:,:-1])/2.
    fmin=binfreqc[nbin>nbin0].min()
    fmax=binfreqc[nbin>nbin0].max()
    clf()
    pcolor(t2, binfreq2, lpds, cmap='jet', vmin=lmin, vmax=lmax)
    colorbar()
    xlim(t2.min(), t2.max())
    ylim(fmin, fmax)
    yscale('log')
    xlabel(r'$t$, s')
    ylabel('$f$, Hz')
    savefig(outfile+'.png')
    close()

#############################################
def quasi2d(hname, n1, n2):
    '''
    makes quasi-2D Rt plots
    '''
    nt=n2-n1
    # first frame
    entryname, t, l, r, sth, rho, u, v = hdf.read(hname, n1)
    nr=size(r)
    var = zeros([nt, nr], dtype=double)
    tar = zeros(nt, dtype=double)
    var[0,:] = v[:] ; tar[0] = t
    for k in arange(n2-n1-1):
        entryname, t, l, r, sth, rho, u, v = hdf.read(hname, n1+k+1)
        var[k+1, :] = v[:]
        tar[k+1] = t
    nv=30
    vlev=linspace(var.min(), var.max(), nv)
    # velocity
    clf()
    fig=figure()
    contourf(r, tar*tscale, var, levels=vlev,cmap='hot')
    colorbar()
    xscale('log') ;  xlabel(r'$r$') ; ylabel(r'$t$')
    fig.set_size_inches(4, 6)
    savefig('q2d_v.png')
    close('all')
    # internal energy density
    umagtar = umag * (1.+3.*(1.-sth**2))/4. * (rstar/r)**6
    lurel = log10(u/umagtar)
    lulev = linspace(lurel.min(), lurel.max(), 20)
    clf()
    fig=figure()
    contourf(r, tar*tscale, lurel, levels=lulev,cmap='hot')
    colorbar()
    xscale('log') ;  xlabel(r'$r$') ; ylabel(r'$t$')
    fig.set_size_inches(4, 6)
    savefig('q2d_u.png')
    close('all')

def postplot(hname, nentry):
    '''
    reading and plotting a single snapshot number "nentry" 
    taken from the HDF output "hname"
    '''
    entryname, t, l, r, sth, rho, u, v = hdf.read(hname, nentry)
    uplot(r, u, rho, sth, v, name=hname+"_"+entryname+'_u')
    vplot(r, v, sqrt(4./3.*u/rho), name=hname+"_"+entryname+'_v')
    
def multiplots(hname, n1, n2):
    '''
    invoking postplot for a number of frames
    '''
    for k in arange(n2-n1)+n1:
        postplot(hname, k)

#####################################
def curvestack(n1, n2, step, prefix = "tireout", postfix = ".dat"):
    '''
    plots a series of U/Umag curves from the ascii output
    '''
    clf()
    for k in arange(n1,n2,step):
        fname = prefix + hdf.entryname(k, ndig=5) + postfix
        print(fname)
        lines = loadtxt(fname, comments="#")
        print(shape(lines))
        r = lines[:,0] ; urel = lines[:,3]
        plot(r, urel, label = str(k))
    plot(r, (r/r.max())**(-10./3.+6.), ':k')
    legend()
    xscale('log') ; yscale('log')
    xlabel(r'$R/R_*$') ; ylabel(r'$U/U_{\rm mag}$')
    savefig("curvestack.png")
    close('all')

#########################################
def energytest(fluxfile='flux', totfile='totals'):
    lines = loadtxt(fluxfile+".dat", comments="#", delimiter=" ", unpack=False)
    tflux = lines[:,0]/tscale ; flu=lines[:,1]
    lines = loadtxt(totfile+".dat", comments="#", delimiter=" ", unpack=False)
    tene = lines[:,0]/tscale ; ene=lines[:,2] ; mass=lines[:,1]
    enlost = cumtrapz(flu, x=tflux, initial=0.)
    Neff=1./rstar
    clf()
    plot(tflux*tscale, enlost, color='k')
    plot(tene*tscale, ene, color='r')
    plot(tene*tscale, (mass-mass[0])*Neff, color='g')
    plot(tene*tscale, tene*mdot*Neff, color='g', linestyle='dotted')
    ylim(ene.min(), (ene+enlost).max())
    xlabel('t')
    ylabel('energy')
    savefig('energytest.png')
    close()
