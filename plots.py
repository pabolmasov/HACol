import matplotlib
from matplotlib import rc
from matplotlib import axes
from numpy import *
from pylab import *

#Uncomment the following if you want to use LaTeX in figures 
rc('font',**{'family':'serif'})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
# #add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 

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
    xlabel('$r$, $GM/c^2$ units')
    yscale('log')
    xscale('log')    
    legend()
    savefig(name+'.png')

def vplot(x, v, cs, name='outplot'):
    '''
    velocity, speed of sound, and virial velocity
    '''
    ioff()
    clf()
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
    errorbar(freq, pds, xerr=dfreq, yerr=dpds, marker='.', mec='k', mfc='k', c='k')
    xscale('log') ; yscale('log')
    ylabel('PDS') ; xlabel('$f$, Hz')
    savefig(outfile+'.png')
    close()
    
def dynspec(t2,binfreq2, pds2, outfile='flux_dyns'):
    lpds=(pds2)
    lmin=lpds[pds2>0.].min() ; lmax=lpds.max()
    clf()
    pcolor(t2, binfreq2, (lpds), cmap='jet', vmin=lmin, vmax=lmax)
    colorbar()
    xlim(t2.min(), t2.max())
    ylim(binfreq2.min(), binfreq2.max())
    yscale('log')
    xlabel(r'$t$, s')
    ylabel('$f$, Hz')
    savefig(outfile+'.png')
    close()
