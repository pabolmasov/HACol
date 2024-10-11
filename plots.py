import matplotlib
from matplotlib import rc
from matplotlib import axes
from matplotlib import interactive, use
from matplotlib import ticker
from numpy import *
import numpy.ma as ma
from pylab import *
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import glob
import re
import os

#Uncomment the following if you want to use LaTeX in figures 
# rc('font',**{'family':'serif'})
# rc('mathtext',fontset='cm')
# rc('mathtext',rm='stix')
# rc('text', usetex=True)
# #add amsmath to the preamble
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 

from hdfoutput import read, entryname
import geometry as geo
from beta import *
from tauexp import *
import configparser as cp
conffile = 'globals.conf'
config = cp.ConfigParser(inline_comment_prefixes="#")
config.read(conffile)

close('all')
ioff()
use('Agg')

formatsequence = ['k-', 'g:', 'b--', 'r-.']

def qloss_separate(rho, v, u, g, conf):
    '''
    standalone estimate for flux distribution
    '''

    betacoeff = conf.getfloat('betacoeff')
    xirad = conf.getfloat('xirad')
    ifthin = conf.getboolean('ifthin')
    cooltwosides = conf.getboolean('cooltwosides')
    betafun = betafun_define()
    #  tau = rho * g.delta
    #  tauphi = rho * g.across / g.delta / 2. # optical depth in azimuthal direction
    #     taueff = copy(1./(1./tau + 1./tauphi))
    taumin = conf.getfloat('taumin')
    taumax = conf.getfloat('taumax')
    if cooltwosides:
        taueff = rho * g.delta 
    else:
        taueff = rho / (1. / g.delta + 2. * g.delta /  g.across) 
    #    taufac =  1.-exp(-tau)
    beta = betafun(Fbeta(rho, u, betacoeff))
    urad = copy(u * (1.-beta)/(1.-beta/2.))
    urad = (urad+abs(urad))/2.    
    if ifthin:
        taufactor = tratfac(taueff, taumin, taumax) / xirad
    else:
        taufactor = taufun(taueff, taumin, taumax) / (xirad*taueff+1.)
    if cooltwosides:
        qloss = copy(2.*urad*(g.across/g.delta) * taufactor)  # diffusion approximation; energy lost from 2 sides
    else:
        qloss = copy(2.*urad*(g.across/g.delta+2.*g.delta) * taufactor)  # diffusion approximation; energy lost from 4 sides
    return qloss


#############################################################
# Plotting block 
def uplot(r, u, rho, sth, v, name='outplot', umagtar = None, ueq = None, configactual = None, unorm = True, time = None):
    '''
    energy u supplemented by rest-mass energy rho c^2
    '''
    if configactual is None:
        umag = 1. # no normalization
        omega = 0.
        rstar = r.min()
    else:
        xifac = configactual.getfloat('xifac')
        m1 = configactual.getfloat('m1')
        rstar = configactual.getfloat('rstar')
        mu30 = configactual.getfloat('mu30')
        b12 = 2.*mu30*(rstar*m1/6.8)**(-3) # dipolar magnetic field on the pole, 1e12Gs units
        mdot = configactual.getfloat('mdot') * 4. *pi # internal units
        r_e = configactual.getfloat('r_e_coeff') * (mu30**2/mdot)**(2./7.)*m1**(-10./7.) * xifac # magnetosphere radius
        umag = b12**2*2.29e6*m1
        #        umag = configactual.getfloat('umag')
        omega = configactual.getfloat('omegafactor')*r_e**(-1.5)
        # omega = configactual.getfloat('omega')
        #        print("umagtar = "+str(umagtar))
    if umagtar is None:
        umagtar = umag*(rstar/r)**6 * (1.+3.*(1.-sth**2))/4.
    if unorm:
        unormfactor = umagtar
    else:
        unormfactor = r*0.+1.
    ioff()
    clf()
    fig=figure()
    plot(r, u/unormfactor, 'k', label='$u$',linewidth=2)
    if(ueq is not None):
        plot(r, ueq/unormfactor, 'k', label=r'$u_{\rm eq}$',linewidth=2, linestyle = 'dotted')
    plot(r, rho/umagtar, 'r', label=r'$\rho c^2$')
    plot(r, rho*v**2/2./umagtar, 'm', label=r'$\frac{1}{2}\rho v^2$')
    plot(r, rho/r /umagtar, 'r', label=r'$\rho/r$', linestyle='dotted')
    plot(r, rho*0.5*(r*omega*sth)**2/umagtar, 'r', label=r'$\frac{1}{2}\rho (\Omega R \sin\theta)^2$', linestyle='dashed')
    plot(r, r*0.+1., 'b', label=r'$u_{\rm mag}$')
    B=u*4./3./unormfactor+rho*(-1./r-0.5*(r*omega*sth)**2+v**2/2.)/umagtar
    plot(r, B, 'g', label='$B$', linestyle='dotted')
    plot(r, -B, 'g', label='$-B$')
    if time is not None:
        if time > 0.:
            lt = int(ceil(2. - log10(time)))
        else:
            lt = 0
        title(r't = '+str(round(time, lt))+'s', fontsize = 20)
    ylim((maximum(u/unormfactor, 1.e-3)[u>0.]).min(), maximum((u/unormfactor).max(), 10.))
    xlabel('$r$, $GM/c^2$ units', fontsize = 20)
    ylabel(r'$U/U_{\rm mag}$', fontsize = 20)
    plt.tick_params(labelsize=16, length=1, width=1., which='minor', direction = "in")
    plt.tick_params(labelsize=16, length=3, width=1., which='major', direction = "in")
    yscale('log')
    xscale('log')    
    legend(fontsize = 14)
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

def somemap(x, y, q, name='map', xlog=True, ylog=False, xtitle=r'$R/R_*$', ytitle='$t$, s', levels = None, inchsize = None, cbtitle = None, addcontour = None, transpose = False, xrange=None, yrange = None):
    '''
    plots a 2dmap
    '''
    if transpose:
        x, y = y, x
        if size(shape(x)) > 1:
            x=x.transpose()
        if size(shape(y)) > 1:
            y=y.transpose()
        q = q.transpose()
        xlog, ylog = ylog, xlog
    
    clf()
    fig=figure()
    if(levels is not None):
        pcolormesh(x, y, q, cmap='hot', vmin = levels.min(), vmax=levels.max())
    else:
        pcolormesh(x, y, q, cmap='hot')
    cb = colorbar()
    if cbtitle is not None:
        cb.set_label(r' '+cbtitle, fontsize=14)
    cb.ax.tick_params(labelsize=12, length=3, width=1., which='major', direction ='in')
    tick_params(labelsize=14, length=3, width=1., which='minor', direction='in')
    tick_params(labelsize=14, length=6, width=1., which='major', direction='in')

    if addcontour is not None:
        ld = size(shape(addcontour)) # equal to 2 if there is a single contour
        if ld == 2:
            if transpose:
                addcontour = addcontour.transpose()
            contour(x, y, addcontour, levels=[1.], colors='k')
        else:
            for kd in arange(ld):
                if transpose:
                    contour(x, y, addcontour[kd].transpose(), levels=[1.], colors='k')
                else:
                    contour(x, y, addcontour[kd], levels=[1.], colors='k')

    if(xlog):
        xscale('log')
    if(ylog):
        yscale('log')
    if xrange is not None:
        xlim(xrange)
    if yrange is not None:
        ylim(yrange)
    if transpose:
        xlabel(ytitle, fontsize=18) ; ylabel(xtitle, fontsize=18)
    else:
        xlabel(xtitle, fontsize=18) ; ylabel(ytitle, fontsize=18)
    if inchsize is not None:
        fig.set_size_inches(inchsize[0], inchsize[1])
        if transpose:
            fig.set_size_inches(inchsize[1], inchsize[0])

    fig.tight_layout()
    savefig(name+'.png')
    savefig(name+'.eps')
    close()
    
def plot_somemap(fname, ncol = -1, xlog = True):
    lines = loadtxt(fname, comments="#", delimiter=" ", unpack=False)
    print(shape(lines))
    x=lines[:,1] ; y=lines[:,0] ; q=lines[:,ncol]
    xun = unique(x) ; yun = unique(y)
    nx = size(xun) ; ny = size(yun)
    x=reshape(x, [ny,nx]) ; y=reshape(y, [ny,nx]) ; q=reshape(q, [ny,nx])
    #    x = transpose(x) ; y=transpose(y) ; q=transpose(q)
    lev1 = quantile(q[x>median(x)], 0.1)
    lev2 = quantile(q[x>median(x)], 0.9)
    nl = 10
    lev1 = -1. ; lev2 = 2.
    levs = (lev2-lev1) * arange(nl)/double(nl-1)+lev1
    somemap(x, y, q, name=fname, xlog=xlog, xtitle=r'$r/R_*$', ytitle = r'$t$, s', transpose=True, levels = levs, inchsize = [3,10])
    
def someplots(x, ys, name='outplot', ylog = False, xlog = True, xtitle=r'$r$', ytitle='', formatsequence = None, legendsequence = None, vertical = None, verticalformatsequence = None, multix = False, yrange = None, xrange = None, inchsize = None, dys = None, linewidthsequence = None, secaxfunpair = None):
    '''
    plots a series of curves  
    if multix is off, we assume that the independent variable is the same for all the data 
    '''
    if isinstance(ys, list):
        ny = len(ys)
    elif isinstance(ys, ndarray):
        ny=shape(ys)[0]
    else:
        ny = 1 # presumably, scalar

    if multix:
        # nx = shape(x)[0]
        if isinstance(x, list):
            nx = len(x)
        elif isinstance(x, ndarray):
            nx=shape(x)[0]
        else:
            nx = 1 # presumably, scalar
        if nx != ny:
            print("X and Y arrays do not match")
            return 0

    if formatsequence is None:
        formatsequence = ["." for x in range(ny)]
    if legendsequence is None:
        legendsequence = ["" for x in range(ny)]
    if linewidthsequence is None:
        linewidthsequence = [1 for x in range(ny)]
        
    clf()
    fig, ax = subplots()
    for k in arange(ny):
        if vertical is not None:
            if verticalformatsequence is None:
                verticalformatsequence = formatsequence[-1]
            nv = size(vertical)
            if nv <= 1:
                plot([vertical, vertical], [ys[k].min(), ys[k].max()], verticalformatsequence)
            else:
                for kv in arange(nv):
                    plot([vertical[kv], vertical[kv]], [ys[k].min(), ys[k].max()], verticalformatsequence)
        if multix:
            plot(x[k], ys[k], formatsequence[k], linewidth = linewidthsequence[k], label = legendsequence[k])
        else:
            plot(x, ys[k], formatsequence[k], linewidth = linewidthsequence[k])
            
        if secaxfunpair is not None:
            secax = ax.secondary_xaxis('top', functions=secaxfunpair)
            secax.set_xlabel(r'$\tau$', fontsize = 14)
            secax.set_xticks([-10., -5., 0., 1., 2., 3., 4.])

            plot(x, ys[k], formatsequence[k], linewidth = linewidthsequence[k], label = legendsequence[k])
    if dys is not None:
        if multix:
            errorbar(x[0], ys[0], fmt = formatsequence[0], yerr = dys)
        else:
            errorbar(x, ys[0], fmt = formatsequence[0], yerr = dys)
    if(xlog):
        xscale('log')
    if(ylog):
        yscale('log')
    if yrange is not None:
        ylim(yrange[0], yrange[1])
    if xrange is not None:
        xlim(xrange[0], xrange[1])
    xlabel(xtitle, fontsize=14) ; ylabel(ytitle, fontsize=14)
    plt.tick_params(labelsize=12, length=1, width=1., which='minor', direction='in')
    plt.tick_params(labelsize=12, length=3, width=1., which='major', direction='in')
    if inchsize is not None:
        fig.set_size_inches(inchsize[0], inchsize[1])
    else:
        fig.set_size_inches(5, 4)
    if legendsequence is not None:
        fig.legend(loc='lower right', borderaxespad=0.,)    
        #loc='upper center',ncol=4,
               #fancybox=True,   bbox_to_anchor=(0.9, 0.5),
                
    fig.tight_layout()
    savefig(name+'.png')
    savefig(name+'.pdf')
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

def errorplot(x, dx, y, dy, outfile = 'errorplot', xtitle = None, ytitle = None, fit = None,
              xrange = None, yrange = None, addline = None, xlog = False, ylog = False, pointlabels = None, lticks = None):
    '''
    addline should be a tuple and contain [x,y]
    '''
    clf()
    fig, ax = subplots()
    if xrange is not None:
        w = where(((x+dx)<xrange[1]) * ((x-dx)> xrange[0]))
        if yrange is not None:
            w = where(((x+dx/2.)<xrange[1]) * ((x-dx/2.)> xrange[0]) * ((y+dy/2.)<yrange[1]) * ((y-dy/2.)> yrange[0]))
        ax.errorbar(x[w], y[w], xerr=dx[w], yerr=dy[w], fmt='.k')
    else:
        ax.errorbar(x, y, xerr=dx, yerr=dy, fmt='.k')
    if addline is not None:
        ax.plot(addline[0], addline[1], 'r:')
    if fit is not None:
        xtmp = linspace(x.min(), x.max(), 100)
        ax.plot(xtmp, exp(log(xtmp)*fit[0]+fit[1]), 'r-')
        ax.plot(xtmp, 1e3/(xtmp/5.5)**3.5, 'b:')
        ax.set_ylim((y-dy).min(), (y+dy).max())
    if pointlabels:
        for k in arange(size(x)):
            ax.text(x[k], y[k], pointlabels[k])
    if xtitle is not None:
        ax.set_xlabel(xtitle, fontsize=16)
    if ytitle is not None:
        ax.set_ylabel(ytitle, fontsize=16)
    if yrange is not None:
        ax.set_ylim(yrange[0], yrange[1])
    if xrange is not None:
        ax.set_xlim(xrange[0], xrange[1])
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    ax.tick_params(labelsize=14, length=1, width=1., which='minor', direction = "in")
    ax.tick_params(labelsize=14, length=3, width=1., which='major', direction = "in")
    if lticks is not None:
        ax.set_xticks(lticks)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    fig.set_size_inches(4., 7.)
    fig.tight_layout()
    savefig(outfile+'.png')
    close()
    
def plot_dynspec(t2,binfreq2, pds2, outfile='flux_dyns', nbin=None, omega=None, logscale=True):

    nbin0=2
    
    if logscale:
        lpds=log10(pds2)
    else:
        lpds = pds2
    # print(lpds)
    lmin=lpds[nbin>=nbin0].min() ; lmax=lpds[nbin>=nbin0].max()
    binfreqc=(binfreq2[1:,1:]+binfreq2[1:,:-1])/2.
    fmin=binfreqc[nbin>=nbin0].min()
    fmax=binfreqc[nbin>=nbin0].max()
    clf()
    fig = figure()
    pcolormesh(t2, binfreq2, lpds, cmap='hot') #, vmin=lmin-0.1, vmax=lmax+0.1)
    cbar = colorbar()
    cbar.ax.tick_params(labelsize=10, length=3, width=1., which='major', direction ='in')
    if logscale:
        cbar.set_label(r'$\log_{10}f^2 PDS$, relative units', fontsize=12)
    else:
        cbar.set_label(r'$f^2 PDS$, relative units', fontsize=12)
    if omega != None:
        plot([t2.min(), t2.max()], [omega/2./pi, omega/2./pi], color='k')
    xlim(t2.min(), t2.max())
    ylim(fmin, fmax)
    #    yscale('log')
    xlabel(r'$t$, s', fontsize = 16)
    ylabel('$f$, Hz', fontsize = 16)
    plt.tick_params(labelsize=14, length=1, width=1., which='minor', direction = "in")
    plt.tick_params(labelsize=14, length=3, width=1., which='major', direction = "in")
    fig.set_size_inches(7., 7.)
    fig.tight_layout()
    savefig(outfile+'.png')
    savefig(outfile+'.eps')
    close()
    return [fmin, fmax] # outputting the frequency range
    
def postplot(hname, nentry, ifdat = False, conf = 'DEFAULT'):
    '''
    reading and plotting a single snapshot number "nentry" 
    taken from the HDF output "hname"
    '''
    outdir = os.path.dirname(hname)
    geofile = outdir+"/geo.dat"
    print(geofile)
    r, theta, alpha, across, l, delta = geo.gread(geofile) 
    if(ifdat):
        fname = hname + entryname(nentry, ndig=5) + ".dat"
        entry = entryname(nentry, ndig=5)
        print(fname)
        lines = loadtxt(fname, comments="#")
        r = lines[:,0] ; rho = lines[:,1] ; v = lines[:,2] ; u = lines[:,3]
        sth = sin(theta)
    else:
        entry, t, l, r, sth, rho, u, v, qloss, glo = read(hname, nentry)
    # reading configure
    configactual = config[conf]
    m1 = configactual.getfloat('m1')
    rstar = configactual.getfloat('rstar')
    mu30 = configactual.getfloat('mu30')
    b12 = 2.*mu30*(rstar*m1/6.8)**(-3) # dipolar magnetic field on the pole, 1e12Gs units
    xifac = configactual.getfloat('xifac')
    mdot = configactual.getfloat('mdot') * 4. *pi # internal units
    r_e = configactual.getfloat('r_e_coeff') * (mu30**2/mdot)**(2./7.)*m1**(-10./7.) * xifac # magnetosphere radius
    umag = b12**2*2.29e6*m1
    #        umag = configactual.getfloat('umag')
    omega = configactual.getfloat('omegafactor')*r_e**(-1.5)
    umagtar = umag*(1./r)**6 * (1.+3.*(1.-sth**2))/4.  
    
    uplot(r*rstar, u*umagtar, rho, sth, v, name=hname+entry+'_u', umagtar = umagtar, unorm = True)
    vplot(r*rstar, v, sqrt(4./3.*u*umagtar/rho), name=hname+entry+'_v')
    
    g = geo.geometry_initialize(r, r.max(), r.max(), writeout=None, afac = 1.)
    g.r = r ; g.theta = theta ; g.alpha = alpha ; g.across = across ; g.l = l ; g.delta = delta # temporary: need a proper way to restore geometry from the output
    
    someplots(r, [-v*rho*across/4./pi, v*0.+mdot/4./pi], name=hname+entry+"_mdot", ytitle="$\dot{m}$", ylog=True, formatsequence = ['.k', 'r-'])
    someplots(r, [-u*v*(r/r.min())**4], name=hname+entry+"_g", ytitle=r"$uv \left( R/R_{\rm *}\right)^4$", ylog=True)
    if ifdat:
        q = qloss_separate(rho, v, u*umagtar, g, config[conf])
    else:
        q = qloss_separate(rho, v, u, g, config[conf])
    perimeter = 2.*(delta*2.+across/delta)
    someplots(r, [q, -rho*v*across/(r*rstar)**2], name=hname+entry+"_q",
              ytitle=r'$\frac{{\rm d}^2 E}{{\rm d}l {\rm d} t}$', ylog=True, xlog = True,
              formatsequence = ['k-', 'r-'])
    print("ltot = "+str(trapezoid(q, x=l)/4./pi))
    print("energy release = "+str(mdot /4./pi/ rstar))
    print("heat from outside = "+str((-u * v * across)[-1] /4./pi))
    
    
def multiplots(hname, n1, n2):
    '''
    invoking postplot for a number of frames
    '''
    for k in arange(n2-n1)+n1:
        postplot(hname, k)

#####################################
def curvestack(n1, n2, step, prefix = "out/tireout", postfix = ".dat", conf = 'DEFAULT'):
    '''
    plots a series of U/Umag curves from the ascii output
    '''
    clf()
    fig = figure()
    for k in arange(n1,n2,step):
        fname = prefix + entryname(k, ndig=5) + postfix
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
    fig.set_size_inches(5, 4)
    fig.tight_layout()
    savefig("curvestack.png")
    close('all')

def rhocurvestack(n1, n2, step, prefix = "out/tireout", postfix = ".dat", conf = 'DEFAULT'):
    '''
    plots a series of U/Umag curves from the ascii output
    '''

    tscale = 4.92594e-06 *1.4 # temporary! need to read it!
    
    rho0 = [] ; rho1 = [] ; vl1 = [] ; vl0 = []
    
    for k in arange(n1,n2,step):
        fname = prefix + entryname(k, ndig=5) + postfix
        print(fname)
        lines = loadtxt(fname, comments="#")
        print(shape(lines))
        r = lines[:,0] ; rho = lines[:,1] ; v = lines[:,2]
        rho0.append(rho[0])   ;     rho1.append(rho[1])
        vl1.append(v[1])
        vl0.append(v[0])
    print(r[0])
    dr = (r[1]-r[0])
    v1 = abs(asarray(vl1))
    v0 = abs(asarray(vl0))
    rho0 = asarray(rho0)
    rho1 = asarray(rho1)
    print(v)
    outdir = os.path.dirname(prefix)
    fluxlines = loadtxt(outdir+'/flux.dat', comments="#", delimiter=" ", unpack=False)
    t = fluxlines[:,0] ; f = fluxlines[:,1]
    t =t[arange(0,n2-n1,step)]

    rho_acc = cumulative_trapezoid((v1*rho1+v0*rho0)/2., x=t/tscale, initial = rho0[0])
    
    someplots(t, [rho0, rho1, rho_acc], name = "rho0", xtitle=r'$t$, s', ytitle=r'$\rho_0$', xlog=False, ylog=True, formatsequence = ['k-', 'r:', 'b--'])

def Vcurvestack(n1, n2, step, prefix = "out/tireout", postfix = ".dat", plot2d=False, conf = 'DEFAULT', rmax = None, mdotshow = False):
    '''
    plots a series of velocity curves from the ascii output
    '''
    rstar = config[conf].getfloat('rstar')
    mdot = config[conf].getfloat('mdot')
    tscale = config[conf].getfloat('tscale')
    m1 = config[conf].getfloat('m1')
    tscale *= m1

    outdir = os.path.dirname(prefix)
    geofile = outdir+"/geo.dat"
    print("mdot = "+str(mdot))
    ii = input("M")
    r, theta, alpha, across, l, delta = geo.gread(geofile) 

    kctr = 0
    vmin=0. ; vmax=0.
    nt = int(floor((n2-n1)/step)) # ; nr = size(r)

    close("all")
    clf()
    fig = figure()
    for k in arange(nt)*step+n1:
        fname = prefix + entryname(k, ndig=5) + postfix
        print(fname)
        lines = loadtxt(fname, comments="#")
        print(shape(lines))
        r = lines[:,0] ; v = lines[:,2] ; rho = lines[:,1]
        nr = size(r)
        if mdotshow:
            plot(r, v*rho*across, label = str(k))
        else:
            plot(r, v, label = str(k))
        if rmax is not None:
            vmincurrent = v[r<rmax].min()
            vmaxcurrent = v[r<rmax].max()
        else:
            vmincurrent = v.min()
            vmaxcurrent = v.max()           
        if(vmincurrent<vmin):
            vmin = vmincurrent
        if(vmaxcurrent>vmax):
            vmax = vmaxcurrent
        if kctr == 0:
            tar = zeros(nt, dtype=double)
        if plot2d & (kctr==0):           
            v2 = zeros([nt, nr], dtype=double)
        if mdotshow & (kctr==0):
            m2 = zeros([nt, nr], dtype=double)
            rho2 = zeros([nt, nr], dtype=double)
        ff=open(fname)
        stime = ff.readline()
        ff.close()
        tar[kctr] = double(''.join(re.findall("\d+[.]\d+e-\d+|\d+[.]\d+", stime)))
        if(plot2d):            
            print(stime+": "+str(tar[kctr]))
            v2[kctr,:] = v[:]
        if mdotshow:
            m2[kctr,:] = (rho*v*across)[:]
            rho2[kctr,:] = (rho)[:]                
        kctr += 1
    if mdotshow:
        plot(r, r*0.-mdot*4.*pi, '--k', label=r'$\dot{M}$')
    else:
        plot(r, -1./sqrt(r*rstar), '--k', label='virial')
        plot(r, -1./7./sqrt(r*rstar), ':k', label=r'$\frac{1}{7}$ virial')
    if nt < 10:
        legend()
    xscale('log')

    if not(mdotshow):
        vmin = maximum(vmin, -1.) ; vmax = minimum(vmax, 1.)
        ylim(vmin, vmax)
        ylabel(r'$v/c$')
    else:
        ylabel(r'$\dot{M}$')
    if rmax is not None:
        xlim(1., rmax)
        ylim(m2.min(), m2.max())
    xlabel(r'$R/R_*$')
    fig.set_size_inches(5, 4)
    fig.tight_layout()
    savefig("Vcurvestack.png")
    if(plot2d):
        nv=20
        clf()
        fig=figure()
        if mdotshow:
            pcolormesh(r, tar, -m2/mdot/4./pi, cmap='hot', vmin=-1., vmax=2.)
        else:
            pcolormesh(r, tar, v2, cmap='hot', vmin=vmin, vmax=vmax)
        colorbar()
        xlabel(r'$R/R_*$') ; ylabel(r'$t$, s')
        if rmax is not None:
            xlim(1., rmax)
        else:
            if r.max()/r.min() > 3.:
                xscale('log')
        # print(tar.min())
        ylim(tar.min(), tar.max())
        fig.set_size_inches(4, 6)
        fig.tight_layout()
        savefig("Vcurvestack_2d.png")
        if mdotshow:
            kslice = 11
            clf()
            plot(tar, rho2[:, kslice], 'k-')
            plot(tar, rho2[:, kslice*2], 'b--')
            #            plot(tar, rho2[:, -kslice], 'r:')
            # yscale('log')
            xlabel(r'$t$, s')
            ylabel(r'$\rho$')            
            fig.set_size_inches(4, 6)
            fig.tight_layout()
            savefig("Vcurvestack_drho.png")
    
    if mdotshow:
        kslice1 = 2
        kslice2 = 3
        clf()
        plot(tar, -m2[:, kslice1]/mdot/4./pi, 'k-')
        plot(tar, -m2[:, kslice2]/mdot/4./pi, 'b--')
        plot(tar, -m2[:, -10]/mdot/4./pi, 'r:')
        #  plot(tar, -m2[:, -10]/mdot, 'r:')
        # ylim(-1., 3.)
        xlabel(r'$t$, s')
        ylabel(r'$s/\dot{M}$')
        savefig("Vcurvestack_mdslice.png")
        
    close('all')
        

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
def multishock_plot(frontfile, trange = None):
    '''
    plots the position of the shock as a function of time and total flux
    '''
    # fluxlines = loadtxt(fluxfile+'.dat', comments="#", delimiter=" ", unpack=False)
    frontlines = loadtxt(frontfile+'.dat', comments="#", delimiter=" ", unpack=False)
    frontinfo = loadtxt(frontfile+'glo.dat', comments="#", delimiter=" ", unpack=False)
    # tf=fluxlines[:,0] ; f=fluxlines[:,1]
    ts=frontlines[1:,0] ; s=frontlines[1:,1] ; ds=frontlines[1:,2]
    # eqlum = frontinfo[0] ;
    rs = frontinfo[1] ; xs = frontinfo[0]
    tf = ts
    f = ds=frontlines[1:,5]

    #     f /= 4.*pi # ; eqlum /= 4.*pi
    
    # interpolate!
    # fint = interp1d(tf, f, bounds_error=False)
    
    someplots(ts, [s, s*0. + rs], name = frontfile + "_frontcurve", xtitle=r'$t$, s', ytitle=r'$R_{\rm shock}/R_*$', xlog=False, formatsequence = ['k-', 'r-', 'b-'],
        xrange = trange)
    someplots(f, [s, s*0. + rs], name = frontfile + "_fluxfront", xtitle=r'$L_{\rm X}/L_{\rm Edd}$', ytitle=r'$R_{\rm shock}/R_*$', xlog=False, ylog=False, formatsequence = ['k-', 'r-', 'b-'], vertical = xs, verticalformatsequence = 'r-')
    # someplots(tf, [f, eqlum], name = frontfile+"_flux", xtitle=r'$t$, s', ytitle=r'$L/L_{\rm Edd}$', xlog=False, ylog=False)
    
def multimultishock_plot(prefices, parflux = True, sfilter = 1., smax = None, trange = None):
        # fluxfile1, frontfile1, fluxfile2, frontfile2):
    '''
    plotting many shock fronts together
    '''
    # multimultishock_plot(["titania_fidu", "titania_v5", "titania_v30"])

    nf = size(prefices)
    tlist = [] ;   flist = [];   slist = [] ;  dslist = []

    maxs = 0.

    for k in arange(nf):
        if not parflux:
            fluxfile = prefices[k]+'/flux'
            fluxlines = loadtxt(fluxfile+'.dat', comments="#", delimiter=" ", unpack=False)
            tf=fluxlines[:,0] # ; f=fluxlines[:,1]

        frontfile = prefices[k]+'/sfront'
        frontlines = loadtxt(frontfile+'.dat', comments="#", delimiter=" ", unpack=False)
        frontinfo = loadtxt(frontfile+'glo.dat', comments="#", delimiter=" ", unpack=False)
        
        if parflux:
            f = frontlines[1:,5]
        else:
            f = fluxlines[:,-1]
        ts=frontlines[1:,0] ; s=frontlines[1:,1] ; ds=frontlines[1:,2]
        # tlist.append(ts)
        if parflux:
            f1 = f # / 4./pi
        else:
            fint = interp1d(tf, f, bounds_error=False)
            f1 = fint(ts)/4./pi
        if smax is None:
            maxs = s.max()
        else:
            maxs = smax
        slist.append(s[(s>sfilter)&(s<maxs)]); flist.append(f1[(s>sfilter)&(s<maxs)])
        tlist.append(ts[(s>sfilter)&(s<maxs)])

        if k == 0:
            eqlum = frontinfo[0] ; rs = frontinfo[1]
        
    clf()
    fig = figure()
    for k in arange(nf):
        if trange is not None:
            w = where((tlist[k]>trange[0]) * (tlist[k]<trange[1]))
            print(size(w))
            flist[k] = (flist[k])[w]
            slist[k] = (slist[k])[w]
            tlist[k] = (tlist[k])[w]
        plot(flist[k], slist[k], formatsequence[k])

    plot([minimum(flist[0].min(), flist[0].min()), maximum(flist[0].max(), flist[0].max())], [rs, rs], 'r-')
    plot([eqlum, eqlum], [minimum(slist[0].min(), slist[0].min()), maximum(slist[0].max(), slist[0].max())], 'r-')
    xlabel(r'$L/L_{\rm Edd}$', fontsize=14) ; ylabel(r'$R_{\rm shock}/R_*$', fontsize=14)
    plt.tick_params(labelsize=12, length=1, width=1., which='minor', direction='in')
    plt.tick_params(labelsize=12, length=3, width=1., which='major', direction='in')
    fig.set_size_inches(6, 4)
    fig.tight_layout()
    savefig("manyfluxfronts.png") ;   savefig("manyfluxfronts.pdf")
    close('all')
    
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

def plot_dts(n, prefix = 'out/tireout', postfix = '.dat', conf = 'DEFAULT'):
    configactual = config[conf]
    CFL = configactual.getfloat('CFL')
    Cth = configactual.getfloat('Cth')
    Cdiff = configactual.getfloat('Cdiff')
    rstar = configactual.getfloat('rstar')
    mu30 = configactual.getfloat('mu30')
    m1 = configactual.getfloat('m1')
    b12 = 2.*mu30*(rstar*m1/6.8)**(-3)
    
    geofile = os.path.dirname(prefix)+"/geo.dat"
    print(geofile)
    r, theta, alpha, across, l, delta = geo.gread(geofile)
    g = geo.geometry()
    g.r = r ; g.theta = theta ; g.alpha = alpha ; g.l = l ; g.delta = delta
    g.across = across ; g.cth = cos(theta)
    dl = l[1:]-l[:-1]
    dl = concatenate([dl, [dl[-1]]])

    umag = b12**2*2.29e6*m1
    umagtar = umag * ((1.+3.*g.cth**2)/4. * (rstar/g.r)**6)
    
    fname = prefix + entryname(n, ndig=5) + postfix
    print(fname)
    lines = loadtxt(fname, comments="#")
    print(shape(lines))
    r1 = lines[:,0] ; rho = lines[:,1] ; v = lines[:,2] ; u = lines[:,3] * umagtar
    print(shape(r1))
    print((r1-r).std())
    csq = 4./3.*u/rho
    dt_CFL = CFL * dl / (sqrt(csq)+abs(v))
    
    qloss = qloss_separate(rho, v, u, g, configactual)

    dt_thermal = Cth * u * g.across / qloss

    dt_diff = Cdiff * dl**2 * rho * 3.
    someplots(r, [dl], ylog = True, ytitle=r'$\Delta l$', name = 'dl', formatsequence = ['k-'])

    someplots(r, [dt_CFL, dt_thermal, dt_diff], ylog = True, ytitle=r'$\Delta t$', name = 'dts', formatsequence = ['k-', 'b:', 'r--'])

