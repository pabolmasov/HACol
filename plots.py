import matplotlib
from matplotlib import rc
from matplotlib import axes
from numpy import *
from pylab import *
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import glob
import re
import os

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
def uplot(r, u, rho, sth, v, name='outplot', umagtar = None, ueq = None):
    '''
    energy u supplemented by rest-mass energy rho c^2
    '''
    if(umag is None):
        umagtar = umag*(rstar/r)**6
    ioff()
    clf()
    fig=figure()
    plot(r, u/umagtar, 'k', label='$u$',linewidth=2)
    if(ueq is not None):
        plot(r, ueq/umagtar, 'k', label=r'$u_{\rm eq}$',linewidth=2, linestyle = 'dotted')
    plot(r, rho/umagtar, 'r', label=r'$\rho c^2$')
    plot(r, rho*v**2/2./umagtar, 'm', label=r'$\frac{1}{2}\rho v^2$')
    plot(r, rho/r /umagtar, 'r', label=r'$\rho/r$', linestyle='dotted')
    plot(r, rho*0.5*(r*omega*sth)**2/umagtar, 'r', label=r'$\frac{1}{2}\rho (\Omega R \sin\theta)^2$', linestyle='dashed')
    plot(r, umagtar/umagtar, 'b', label=r'$u_{\rm mag}$')
    B=u*4./3.+rho*(-1./r-0.5*(r*omega*sth)**2+v**2/2.)
    plot(r, B/umagtar, 'g', label='$B$', linestyle='dotted')
    plot(r, -B/umagtar, 'g', label='$-B$')
    #    plot(x, y0, 'b')
    #    xscale('log')
    #    ylim(umag*((rstar/r)**6).min(), umag)
    ylim(((u/umagtar)[u>0.]).min(), (u/umagtar).max())
    xlabel('$r$, $GM/c^2$ units')
    ylabel(r'$U/U_{\rm mag}$')
    yscale('log')
    xscale('log')    
    legend()
    fig.tight_layout()
    fig.set_size_inches(6, 5)
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
    ylim(-0.5,0.5)
    legend()
    savefig(name+'.png')
    close()

def splot(x, y, name='outplot', fmt='-k', xtitle=r'$r$', ytitle=r'$S(R)$'):
    '''
    so far plots some quantity S(R) 
    '''
    clf()
    plot(x, y, fmt)
    xscale('log') ; yscale('log')
    xlabel(xtitle) ; ylabel(ytitle)
    savefig(name+'.png')
    close('all')
    
def someplots(x, ys, name='outplot', ylog = False, xlog = True, xtitle=r'$r$', ytitle=''):
    '''
    plots a series of curves  
    '''
    ny=shape(ys)[0]
    colorsequence = ['k', 'r', 'b']
    sc=size(colorsequence)

    clf()
    for k in arange(ny):
        plot(x, ys[k], 'k', color=colorsequence[k % sc])
    if(xlog):
        xscale('log')
    if(ylog):
        yscale('log')
    xlabel(xtitle) ; ylabel(ytitle)
    savefig(name+'.png')
    close('all')
  
#########################################################################
# post-processing PDS plot:

def pdsplot(freq, pds, outfile='pds'):
    clf()
    plot(freq[1:], pds[1:], 'k')
    xscale('log') ; yscale('log')
    ylabel('PDS') ; xlabel('$f$, Hz')
    savefig(outfile+'.png')
    close()

def binplot_short(freq, dfreq, pds, dpds, outfile='binnedpds'):

    nf=1000
    ftmp=(freq.max()/freq.min())**(arange(nf+1)/double(nf))*freq.min()
    w=where(pds>dpds)
    clf()
    errorbar(freq, pds, xerr=dfreq, yerr=dpds, fmt='.k')
    plot(ftmp, 3./sqrt(1.+ftmp**2)/ftmp, color='r')
    xscale('log') ; yscale('log')
    ylim(((pds-dpds)[w]).min(), ((pds+dpds)[w]).max())
    ylabel('PDS') ; xlabel('$f$, Hz')
    savefig(outfile+'.png')
    close()
    
def dynspec(t2,binfreq2, pds2, outfile='flux_dyns', nbin=None, omega=None):

    nbin0=2
    
    lpds=log10(pds2)
    lmin=lpds[nbin>nbin0].min() ; lmax=lpds[nbin>nbin0].max()
    binfreqc=(binfreq2[1:,1:]+binfreq2[1:,:-1])/2.
    fmin=binfreqc[nbin>nbin0].min()
    fmax=binfreqc[nbin>nbin0].max()
    clf()
    pcolormesh(t2, binfreq2, pds2, cmap='hot_r') #, vmin=10.**lmin, vmax=10.**lmax)
    colorbar()
    if omega != None:
        plot([t2.min(), t2.max()], [omega/2./pi, omega/2./pi], color='k')
    xlim(t2.min(), t2.max())
    ylim(fmin, fmax)
    yscale('log')
    xlabel(r'$t$, s')
    ylabel('$f$, Hz')
    savefig(outfile+'.png')
    savefig(outfile+'.eps')
    close()

#############################################
def quasi2d(hname, n1, n2):
    '''
    makes quasi-2D Rt plots
    '''
    outdir = os.path.dirname(hname)
    
    nt=n2-n1
    # first frame
    entryname, t, l, r, sth, rho, u, v = hdf.read(hname, n1)
    nr=size(r)
    nrnew = 500 # radial mesh interpolated to nrnew
    rnew = (r.max()/r.min())**(arange(nrnew)/double(nrnew-1))*r.min()
    sthfun = interp1d(r, sth)
    sthnew = sthfun(rnew)
    var = zeros([nt, nrnew], dtype=double)
    uar = zeros([nt, nrnew], dtype=double)
    lurel = zeros([nt, nrnew], dtype=double)
    tar = zeros(nt, dtype=double)
    #    var[0,:] = v[:] ; uar[0,:] = u[:] ; tar[0] = t
    for k in arange(n2-n1):
        entryname, t, l, r, sth, rho, u, v = hdf.read(hname, n1+k)
        vfun = interp1d(r, v, kind = 'linear')
        var[k, :] = vfun(rnew)
        ufun = interp1d(r, u, kind = 'linear')
        uar[k, :] = ufun(rnew)
        tar[k] = t
    nv=30
    vlev=linspace(var.min(), var.max(), nv, endpoint=True)
    print(var.min())
    print(var.max())
    varmean = var.mean(axis=0)
    varstd = var.std(axis=0)
    # velocity
    clf()
    fig=figure()
    pcolormesh(rnew, tar*tscale, var, vmin=var.min(), vmax=var.max(),cmap='hot')
    colorbar()
    contour(rnew, tar*tscale, var, levels=[0.], colors='k')
    xscale('log') ;  xlabel(r'$R/R_{\rm NS}$', fontsize=14) ; ylabel(r'$t$, s', fontsize=14)
    fig.set_size_inches(4, 6)
    fig.tight_layout()
    savefig(outdir+'/q2d_v.png')
    savefig(outdir+'/q2d_v.eps')
    close('all')
    clf()
    fig=figure()
    plot(rnew, -sqrt(1./rstar/rnew), ':k')
    plot(rnew, rnew*0., '--k')
    plot(rnew, varmean, '-k')
    plot(rnew, varmean+varstd, color='gray')
    plot(rnew, varmean-varstd, color='gray')
    ylim((varmean-varstd*2.).min(), (varmean+varstd*2.).max())
    xscale('log') ;  xlabel(r'$R/R_{\rm NS}$', fontsize=14)
    ylabel(r'$\langle v\rangle /c$', fontsize=14)
    fig.set_size_inches(3.35, 2.)
    fig.tight_layout()
    savefig(outdir+'/q2d_vmean.png')
    savefig(outdir+'/q2d_vmean.eps')
    close('all')
    
    # internal energy density
    umagtar = umag * (1.+3.*(1.-sthnew**2))/4. / (rnew)**6 # r is already in rstar units
    #    print(umag)
    for k in arange(nrnew):
        lurel[:,k] = log10(uar[:,k]/umagtar[k])
    lulev = linspace(lurel[uar>0.].min(), lurel[uar>0.].max(), 20, endpoint=True)
    print(lulev)
    clf()
    fig=figure()
    contourf(rnew, tar*tscale, lurel, cmap='hot_r', levels=lulev)
    colorbar()
    contour(rnew, tar*tscale, lurel, levels=[0.], colors='k')
    xscale('log') ;  xlabel(r'$R/R_{\rm NS}$', fontsize=14) ; ylabel(r'$t$, s', fontsize=14)
    fig.set_size_inches(4, 6)
    savefig(outdir+'/q2d_u.png')
    savefig(outdir+'/q2d_u.eps')
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
def curvestack(n1, n2, step, prefix = "out/tireout", postfix = ".dat"):
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
    plot(r, r*0.+1., linestyle='dotted', color='gray')
    plot(r, (r/r.max())**(-10./3.+6.), ':k')
    legend()
    xscale('log') ; yscale('log')
    xlabel(r'$R/R_*$') ; ylabel(r'$U/U_{\rm mag}$')
    savefig("curvestack.png")
    close('all')

def Vcurvestack(n1, n2, step, prefix = "out/tireout", postfix = ".dat", plot2d=False):
    '''
    plots a series of velocity curves from the ascii output
    '''
    kctr = 0
    vmin=0. ; vmax=0.
    clf()
    for k in arange(n1,n2,step):
        fname = prefix + hdf.entryname(k, ndig=5) + postfix
        print(fname)
        lines = loadtxt(fname, comments="#")
        print(shape(lines))
        r = lines[:,0] ; v = lines[:,2]
        plot(r, v, label = str(k))
        if(v.min()<vmin):
            vmin = v.min()
        if(v.max()>vmax):
            vmax = v.max()
        if plot2d & (kctr==0):
            nt = int(floor((n2-n1)/step)) ; nr = size(r)
            tar = zeros(nt, dtype=double)
            v2 = zeros([nt, nr], dtype=double)
        if(plot2d):
            ff=open(fname)
            stime = ff.readline()
            ff.close()
            tar[kctr] = double(''.join(re.findall("\d+[.]\d+e-\d+|\d+[.]\d+", stime)))
            print(stime+": "+str(tar[kctr]))
            v2[kctr,:] = v[:]
            kctr += 1
    plot(r, -1./sqrt(r*rstar), '--k', label='virial')
    plot(r, -1./7./sqrt(r*rstar), ':k', label=r'$\frac{1}{7}$ virial')
    legend()
    xscale('log')
    ylim(vmin, vmax)
    xlabel(r'$R/R_*$') ; ylabel(r'$v/c$')
    savefig("Vcurvestack.png")
    if(plot2d):
        nv=20
        clf()
        fig=figure()
        pcolormesh(r, tar, v2, cmap='hot', vmin=vmin, vmax=vmax) #, levels=(arange(nv+1)-0.5)/double(nv)*(vmax-vmin)+vmin, cmap='hot')
        colorbar()
        xscale('log')
        xlabel(r'$R/R_*$') ; ylabel(r'$t$, s')
        ylim(tar.min(), tar.max())
        fig.set_size_inches(4, 6)
        fig.tight_layout()
        savefig("Vcurvestack_2d.png")
    close('all')
        

#########################################
def energytest(fluxfile='out/flux', totfile='out/totals'):
    lines = loadtxt(fluxfile+".dat", comments="#", delimiter=" ", unpack=False)
    tflux = lines[:,0]/tscale ; flu=lines[:,1]
    lines = loadtxt(totfile+".dat", comments="#", delimiter=" ", unpack=False)
    tene = lines[:,0]/tscale ; ene=lines[:,2] ; mass=lines[:,1]
    enlost = cumtrapz(flu, x=tflux, initial=0.)
    Neff=1./rstar
    clf()
    plot(tflux*tscale, enlost, color='k', label="radiated")
    plot(tene*tscale, ene-ene[0], color='r', label="gain")
    plot(tene*tscale, (mass-mass[0])*Neff, color='g', label="gravitational")
    plot(tene*tscale, (tene-tene[0])*mdot*Neff, color='g', linestyle='dotted', label="gravitational, est.")
    legend()
    ylim(((mass-mass[0])*Neff).min(), ((mass-mass[0])*Neff).max())
    xlabel('t') 
    ylabel('energy')
    savefig('energytest.png')
    close()

###########################
def binplot(xe, f, df, fname = "binplot", fit = 0):

    xc = (xe[1:]+xe[:-1])/2. ; xs = abs(-xe[1:]+xe[:-1])/2.
    clf()
    fig=figure()
    errorbar(xc, f*xc, xerr=xs, yerr=df, fmt='.-',
             linestyle='None', mec='k', mfc='k')
    if size(fit)>0 :
        plot(xe, fit*xe, color='r')
    xscale('log') ; yscale('log') ; xlabel('$L/L^*$', fontsize=14)
    ylabel('$LdN/dL$')
    ylim(((f*xc)[f>df]).min(), (f*xc).max()*1.5)
    fig.set_size_inches(4, 3)
    fig.tight_layout()
    savefig(fname+".png")
    savefig(fname+".eps")
    close("all")
