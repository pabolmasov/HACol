import matplotlib
from matplotlib import rc
from matplotlib import axes
from matplotlib import interactive, use
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
import geometry as geo
from globals import *
from beta import *

close('all')
ioff()
use('Agg')

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

def somemap(x, y, q, name='map', xlog=True, ylog=False, xtitle='$r$, $GM/c^2$ units', ytitle='$t$, s', levels = None):
    '''
    plots a 2dmap
    '''
    clf()
    fig=figure()
    if(levels is not None):
        pcolormesh(x, y, q, cmap='hot', vmin = levels.min(), vmax=levels.max())
    else:
        pcolormesh(x, y, q, cmap='hot')
    colorbar()
    #    contour(x, y, q, levels=[1.], colors='k')
    if(xlog):
        xscale('log')
    if(ylog):
        yscale('log')
    xlabel(xtitle) ; ylabel(ytitle)
    fig.tight_layout()
    savefig(name)
    close()
    
def plot_somemap(fname):
    lines = loadtxt(fname, comments="#", delimiter=" ", unpack=False)
    x=lines[:,1] ; y=lines[:,0] ; q=lines[:,2]
    xun = unique(x) ; yun = unique(y)
    nx = size(xun) ; ny = size(yun)
    x=reshape(x, [ny,nx]) ; y=reshape(y, [ny,nx]) ; q=reshape(q, [ny,nx])
    #    x = transpose(x) ; y=transpose(y) ; q=transpose(q)
    somemap(x, y, -q/mdot, name=fname+".png", levels=arange(50)/30.,
            xlog=False, xtitle='$r/R_*$')
    
def someplots(x, ys, name='outplot', ylog = False, xlog = True, xtitle=r'$r$', ytitle='', formatsequence = None, vertical = None):
    '''
    plots a series of curves  
    '''
    ny=shape(ys)[0]

    if formatsequence is None:
        formatsequence = ["." for x in range(ny)]

    clf()
    fig = figure()
    for k in arange(ny):
        if vertical is not None:
            plot([vertical, vertical], [ys[k].min(), ys[k].max()], 'r-')
        plot(x, ys[k], formatsequence[k])
    if(xlog):
        xscale('log')
    if(ylog):
        yscale('log')
    xlabel(xtitle, fontsize=14) ; ylabel(ytitle, fontsize=14)
    plt.tick_params(labelsize=12, length=1, width=1., which='minor')
    plt.tick_params(labelsize=12, length=3, width=1., which='major')
    fig.set_size_inches(4, 4)
    fig.tight_layout()
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
    
def plot_dynspec(t2,binfreq2, pds2, outfile='flux_dyns', nbin=None, omega=None):

    nbin0=2
    
    lpds=log10(pds2)
    lmin=lpds[nbin>nbin0].min() ; lmax=lpds[nbin>nbin0].max()
    binfreqc=(binfreq2[1:,1:]+binfreq2[1:,:-1])/2.
    fmin=binfreqc[nbin>nbin0].min()
    fmax=binfreqc[nbin>nbin0].max()
    clf()
    pcolormesh(t2, binfreq2, pds2, cmap='hot') #, vmin=10.**lmin, vmax=10.**lmax)
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
    
    betafun = betafun_define() # defines the interpolated function for beta

    nt=n2-n1
    # first frame
    entryname, t, l, r, sth, rho, u, v, qloss = hdf.read(hname, n1)
    nr=size(r)
    nrnew = 500 # radial mesh interpolated to nrnew
    rnew = (r.max()/r.min())**(arange(nrnew)/double(nrnew-1))*r.min()
    sthfun = interp1d(r, sth)
    sthnew = sthfun(rnew)
    var = zeros([nt, nrnew], dtype=double)
    uar = zeros([nt, nrnew], dtype=double)
    par = zeros([nt, nrnew], dtype=double)
    qar = zeros([nt, nrnew], dtype=double)
    lurel = zeros([nt, nrnew], dtype=double)
    tar = zeros(nt, dtype=double)
    #    var[0,:] = v[:] ; uar[0,:] = u[:] ; tar[0] = t
    for k in arange(n2-n1):
        entryname, t, l, r, sth, rho, u, v, qloss = hdf.read(hname, n1+k)
        vfun = interp1d(r, v, kind = 'linear')
        var[k, :] = vfun(rnew)
        qfun = interp1d(r, qloss, kind = 'linear')
        qar[k, :] = qfun(rnew)
        ufun = interp1d(r, u, kind = 'linear')
        uar[k, :] = ufun(rnew)
        beta = betafun(Fbeta(rho, u))
        press = u/3./(1.-beta/2.)
        pfun = interp1d(r, press, kind = 'linear')
        par[k, :] = pfun(rnew)
        tar[k] = t
    nv=30
    vmin = round(var.min(),2)
    vmax = round(var.max(),2)
    vlev=linspace(vmin, vmax, nv, endpoint=True)
    print(var.min())
    print(var.max())
    varmean = var.mean(axis=0)
    varstd = var.std(axis=0)
    # velocity
    clf()
    fig=figure()
    contourf(rnew, tar*tscale, var, vlev, cmap='hot')
    #    pcolormesh(rnew, tar*tscale, var, vmin=vmin, vmax=vmax,cmap='hot')
    colorbar()
#    contour(rnew, tar*tscale, var, levels=[0.], colors='k')
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
    umin = round(lurel[uar>0.].min(),2)
    umax = round(lurel[uar>0.].max(),2)
    lulev = linspace(umin, umax, nv, endpoint=True)
    print(lulev)
    clf()
    fig=figure()
    contourf(rnew, tar*tscale, lurel, cmap='hot', levels=lulev)
    colorbar()
    contour(rnew, tar*tscale, par/umagtar, levels=[0.9], colors='k')
    xscale('log') ;  xlabel(r'$R/R_{\rm NS}$', fontsize=14) ; ylabel(r'$t$, s', fontsize=14)
    fig.set_size_inches(4, 6)
    fig.tight_layout()
    savefig(outdir+'/q2d_u.png')
    savefig(outdir+'/q2d_u.eps')
    close('all')
    # Q-:
    clf()
    fig=figure()
    contourf(rnew, tar*tscale, log10(qar), cmap='hot')
    #    pcolormesh(rnew, tar*tscale, var, vmin=vmin, vmax=vmax,cmap='hot')
    colorbar()
#    contour(rnew, tar*tscale, var, levels=[0.], colors='k')
    xscale('log') ;  xlabel(r'$R/R_{\rm NS}$', fontsize=14) ; ylabel(r'$t$, s', fontsize=14)
    fig.set_size_inches(4, 6)
    fig.tight_layout()
    savefig(outdir+'/q2d_q.png')
    savefig(outdir+'/q2d_q.eps')
    close('all')
    

def postplot(hname, nentry, ifdat = True):
    '''
    reading and plotting a single snapshot number "nentry" 
    taken from the HDF output "hname"
    '''
    geofile = os.path.dirname(hname)+"/geo.dat"
    print(geofile)
    r, theta, alpha, across, l, delta = geo.gread(geofile) 
    if(ifdat):
        fname = hname + hdf.entryname(nentry, ndig=5) + ".dat"
        entryname = hdf.entryname(nentry, ndig=5)
        print(fname)
        lines = loadtxt(fname, comments="#")
        r = lines[:,0] ; rho = lines[:,1] ; v = lines[:,2] ; u = lines[:,3]
    else:
        entryname, t, l, r, sth, rho, u, v, qloss = hdf.read(hname, nentry)
    #    uplot(r, u, rho, sth, v, name=hname+"_"+entryname+'_u')
    #    vplot(r, v, sqrt(4./3.*u/rho), name=hname+"_"+entryname+'_v')
    someplots(r, [-v*rho*across, v*rho*across], name=hname+entryname+"_mdot", ytitle="$\dot{m}$", ylog=True, formatsequence = ['.k', '.r'])
    someplots(r, [-u*v*(r/r.min())**4], name=hname+entryname+"_g", ytitle=r"$uv \left( R/R_{\rm NS}\right)^4$", ylog=True)
    
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
    print(Neff)
    enlostfun = interp1d(tflux, enlost, kind = 'linear')
    clf()
    #    plot(tflux*tscale, enlost, color='k', label="radiated")
    plot(tene*tscale, ene-ene[0]+enlostfun(tene)/2., color='k', label="budget")    
    plot(tene*tscale, -ene+ene[0], color='r', label="gain")
    plot(tene*tscale, (mass-mass[0])*Neff, color='g', label="gravitational")
    plot(tene*tscale, (tene-tene[0])*mdot*Neff, color='g', linestyle='dotted', label="gravitational, est.")
    legend()
    #    ylim(((mass-mass[0])*Neff).min(), ((mass-mass[0])*Neff).max())
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
######################################################
def multishock_plot(fluxfile, frontfile):
    '''
    plots the position of the shock as a function of time and total flux
    '''
    fluxlines = loadtxt(fluxfile+'.dat', comments="#", delimiter=" ", unpack=False)
    frontlines = loadtxt(frontfile+'.dat', comments="#", delimiter=" ", unpack=False)
    frontinfo = loadtxt(frontfile+'glo.dat', comments="#", delimiter=" ", unpack=False)
    tf=fluxlines[:,0] ; f=fluxlines[:,1]
    ts=frontlines[1:,0] ; s=frontlines[1:,1] ; ds=frontlines[1:,2]
    eqlum = frontinfo[0] ; rs = frontinfo[1] ; rcool = frontinfo[2]

    f /= 4.*pi # ; eqlum /= 4.*pi
    
    # interpolate!
    fint = interp1d(tf, f, bounds_error=False)
    
    someplots(ts, [s, s*0. + rs, s*0. + rcool], name = frontfile + "_frontcurve", xtitle=r'$t$, s', ytitle=r'$R_{\rm shock}/R_*$', xlog=False, formatsequence = ['k-', 'r-', 'b-'])
    someplots(fint(ts), [s, s*0. + rs, s*0. + rcool], name = frontfile + "_fluxfront", xtitle=r'$L/L_{\rm Edd}$', ytitle=r'$R_{\rm shock}/R_*$', xlog=False, ylog=False, formatsequence = ['k-', 'r-', 'b-'], vertical = eqlum)
    someplots(tf, [f, eqlum], name = frontfile+"_flux", xtitle=r'$t$, s', ytitle=r'$L/L_{\rm Edd}$', xlog=False, ylog=False)
    
def twomultishock_plot(fluxfile1, frontfile1, fluxfile2, frontfile2):
    '''
    plotting two shock fronts together
    '''
    # twomultishock_plot("titania_fidu/flux", "titania_fidu/sfront", "titania_rot/flux", "titania_rot/sfront")
    fluxlines1 = loadtxt(fluxfile1+'.dat', comments="#", delimiter=" ", unpack=False)
    frontlines1 = loadtxt(frontfile1+'.dat', comments="#", delimiter=" ", unpack=False)
    frontinfo1 = loadtxt(frontfile1+'glo.dat', comments="#", delimiter=" ", unpack=False)
    fluxlines2 = loadtxt(fluxfile2+'.dat', comments="#", delimiter=" ", unpack=False)
    frontlines2 = loadtxt(frontfile2+'.dat', comments="#", delimiter=" ", unpack=False)
    frontinfo2 = loadtxt(frontfile2+'glo.dat', comments="#", delimiter=" ", unpack=False)
    tf1=fluxlines1[:,0] ; f1=fluxlines1[:,1]
    ts1=frontlines1[1:,0] ; s1=frontlines1[1:,1] ; ds1=frontlines1[1:,2]
    tf2=fluxlines2[:,0] ; f2=fluxlines2[:,1]
    ts2=frontlines2[1:,0] ; s2=frontlines2[1:,1] ; ds2=frontlines2[1:,2]
    eqlum = frontinfo1[0] ; rs = frontinfo1[1]

    f1 /= 4.*pi ; f2 /= 4.*pi  # ; eqlum /= 4.*pi
    
    # interpolate!
    fint1 = interp1d(tf1, f1, bounds_error=False)
    fint2 = interp1d(tf2, f2, bounds_error=False)
    sint = interp1d(ts2, s2, bounds_error=False) # second shock position

    clf()
    plot(fint1(ts1), s1, 'g--', linewidth = 2)
    plot(fint2(ts2), s2, 'k-')
    plot([minimum(f1.min(), f2.min()), maximum(f1.max(), f2.max())], [rs, rs], 'r-')
    plot([eqlum, eqlum], [minimum(s1.min(), s2.min()), maximum(s1.max(), s2.max())], 'r-')
    xlabel(r'$L/L_{\rm Edd}$', fontsize=14) ; ylabel(r'$R_{\rm shock}/R_*$', fontsize=14)
    savefig("twofluxfronts.png") ;   savefig("twofluxfronts.eps")
    close('all')
    
    #    someplots(fint1(ts1), [s1, sint(ts1), s1*0. + rs], name = "twofluxfronts", xtitle=r'Flux', ytitle=r'$R_{\rm shock}/R_*$', xlog=False, ylog=False, formatsequence = ['k-', 'r-', 'b-'], vertical = eqlum)
    
#############################################################
def allfluxes():
    dirs = [ 'titania_fidu', 'titania_rot', 'titania_irr']
    labels = ['F', 'R', 'I']
    fmtseq = ['-k', '--r', 'g:', '-.b']

    eta = 0.206
    
    clf()
    fig = figure()
    plot([0., 0.4], [10.*eta,10.*eta], 'gray')
    for k in arange(size(dirs)):
        fluxlines = loadtxt(dirs[k]+'/flux.dat', comments="#", delimiter=" ", unpack=False)
        t = fluxlines[:,0] ; f = fluxlines[:,1]
        plot(t, f/4./pi, fmtseq[k], label=labels[k])
        
        #    legend()
    #    yscale('log') ; ylim(1.,20.);
    xlim(0.001,0.1)  ; ylim(1.5,3.) ; xscale('log')
    xlabel(r'$t$, s', fontsize=18) ; ylabel(r'$L/L_{\rm Edd}$', fontsize=18)
    plt.tick_params(labelsize=14, length=1, width=1., which='minor')
    plt.tick_params(labelsize=14, length=3, width=1., which='major')
    fig.set_size_inches(4, 4)
    fig.tight_layout()
    savefig('allfluxes.png')
    savefig('allfluxes.eps')
    close('all')
