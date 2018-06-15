import matplotlib
from matplotlib import rc
from matplotlib import axes
from numpy import *
from pylab import *
from scipy.integrate import *
from scipy.interpolate import *

#Uncomment the following if you want to use LaTeX in figures 
rc('font',**{'family':'serif'})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
# #add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 

import numpy.random
# import time
import os
import os.path

# let us assume GM=1, c=1, kappa=1; this implies Ledd=4.*pi

nx=1000 # the actual number of points in use
nx0=nx*20 # first we make a finer mesh for interpolation

b12=10.
m1=1.4
mdot=10000./4./pi # mass accretion rate
rstar=6. # GM/c**2 units
# vout=-0.5/sqrt(re) # initial poloidal velocity at the outer boundary 
eta=0.5 # self-illumination efficiency 
mfloor=1e-25  # crash floor for mass per unit length
rhofloor=1e-25 # crash floor for density
ufloor=1e-25 # crash floor for energy density
afac=0.5 # part of the longitudes subtended by the flow
re = 123. * ((b12*rstar**3)**2/mdot)**(2./7.)*m1**(-4./7.) # magnetospheric radius
dre=minimum(mdot, re*0.5) # radial extent of the flow at re
print("magnetospheric radius re = "+str(re))
print("Delta re = "+str(dre))
tscale=4.92594e-06*m1

omega=0.95*re**(-1.5) # in Keplerian units on the outer rim
umag=b12**2*3.2e6 # magnetic energy density at the surface, for a 1.4Msun accretor
pmagout=umag*(rstar/re)**6 # magnetic field pressure at the outer rim of the disc
vout=-0.5*pmagout*4.*pi*re*dre/mdot # initial poloidal velocity at the outer boundary 

xirad=1. # radiation loss scaling

#############################################################
# Plotting block (to be moved to a separate file)
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
    B=u*4./3.+rho(-1./r+0.5*(r*omega*sth)**2+v**2/2.)
    plot(r, B, 'g', label='$B$', linestyle='dotted')
    plot(r, B, 'g', label='$-B$')
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


###########################################################################################
def geometry(r, writeout=None):
    '''
    computes all the geometrical quantities. Sufficient to run once before the start of the simulation.
    Output: sin(theta), cos(theta), sin(alpha), cos(alpha), tangential cross-section area, 
    and l (zero at the surface, growing with radius)
    adding nontrivial writeout key allows to write the geometry to an ascii file 
    '''
    #    theta=arcsin(sqrt(r/re))
    #    sth=sin(theta) ; cth=cos(theta)
    sth=sqrt(r/re) ; cth=sqrt(1.-r/re) # OK
    across=4.*pi*afac*dre*re*sth**3/sqrt(1.+3.*cth**2) # OK 
    alpha=arctan((cth**2-1./3.)/sth/cth) # Galja's formula (3)
    sina=sin(alpha) ; cosa=cos(alpha)
    l=cumtrapz(sqrt(1.+3.*cth**2)/2./cth, x=r, initial=0.) # coordinate along the field line
    # dl diverges near R=Re, hence the maximal radius should be smaller than Re
    # ascii output:
    if(writeout != None):
        theta=arctan(sth/cth)
        fgeo=open(writeout, 'w')
        fgeo.write('# format: r -- theta -- alpha -- across -- l\n')
        for k in arange(size(l)):
            fgeo.write(str(r[k])+' '+str(theta[k])+' '+str(alpha[k])+' '+str(across[k])+' '+str(l[k])+'\n')
        fgeo.close()
    
    return sth, cth, sina, cosa, across, l

def cons(rho, v, u, across, r, sth):
    '''
    computes conserved quantities from primitives
    '''
    m=rho*across # mass per unit length
    s=m*v # momentum per unit length
    e=(u+rho*(v**2/2.- 1./r - 0.5*(omega*r*sth)**2))*across  # total energy (thermal + mechanic) per unit length
    return m, s, e

def fluxes(rho, v, u, across, r, sth):
    '''
    computes the fluxes of conserved quantities, given primitives; 
    radiation diffusion flux is not included, as it is calculated at halfpoints
    '''
    s=rho*v*across # mass flux (identical to momentum per unit length -- can we use it?)
    p=across*(rho*v**2+u/3.) # momentum flux (radiation-dominated case)
    fe=across*v*(4./3.*u+(v**2/2.-1./r-0.5*(omega*r*sth)**2)*rho)
    # flux limiters:
    #    s[0]=0. ; p[0]=u[0]/3.*across[0]; fe[0]=0.
    return s, p, fe

def solver_hlle(f, q, sl, sr):
    '''
    makes a proxy for a half-step flux, HLLE-like
    flux of quantity q, density of quantity q, sound velocity to the left, sound velocity to the right
    '''
    #    sr=1.+vshift[1:] ; sl=-1.+vshift[:-1]
    ds=sr[1:]-sl[:-1]
    wpos=where(ds>0.)
    fhalf=sr[1:]*0.
    if(size(wpos)>0):
        fhalf[wpos] = ((sr[1:]*f[:-1]-sl[:-1]*f[1:]+sl[:-1]*sr[1:]*(q[1:]-q[:-1]))/(sr[1:]-sl[:-1]))[wpos] # classic HLLE
    return fhalf

def sources(rho, v, u, across, r, sth, cth, sina, cosa, dt=0., ltot=0.):
    '''
    computes the RHSs of conservation equations
    no changes in mass
    momentum injection through gravitational and centrifugal forces
    energy losses through the surface
    outputs: dm, ds, de, and separately the amount of energy radiated per unit length per unit time ("flux")
    '''
    sinsum=sina*cth+cosa*sth # cos( pi/2-theta + alpha) = sin(theta-alpha)
    gamedd=eta*ltot/(rho*across/(2.*r*sth)) # so far turned off
    force=(-sinsum/r**2*(1.-gamedd)+omega**2*r*sth*cosa)*rho*across
    if(dt>0.):
        qloss_fac=xirad*8./3./(rho+1.)*r*sth*afac*dt
        qloss=(1.-exp(-qloss_fac))/dt
    else:
        qloss=xirad*8./3.*(u/(rho+1.))*r*sth*afac
    #    qloss*=0.
        
    work=v*force
    return rho*0., force, work-qloss, qloss

def toprim(m, s, e, across, r, sth):
    '''
    convert conserved quantities to primitives
    '''
    rho=m/across
    v=s/m
    v[rho<=rhofloor]=0.
    u=(e-m*(v**2/2.-1./r-0.5*(r*sth*omega)**2))/across
    u[u<=ufloor]=0.
    return rho, v, u

def main_step(m, s, e, l_half, s_half, p_half, fe_half, dm, ds, de, dt, r, sth):
    '''
    main advance in a dt step
    input: three densities, l (midpoints), three fluxes (midpoints), three sources, timestep, r, sin(theta)
    output: three new densities
    includes boundary conditions!
    '''
    nl=size(m)
    m1=zeros(nl) ; s1=zeros(nl); e1=zeros(nl)
    m1[1:-1] = m[1:-1]+ (-(s_half[1:]-s_half[:-1])/(l_half[1:]-l_half[:-1]) + dm[1:-1]) * dt
    s1[1:-1] = s[1:-1]+ (-(p_half[1:]-p_half[:-1])/(l_half[1:]-l_half[:-1]) + ds[1:-1]) * dt 
    e1[1:-1] = e[1:-1]+ (-(fe_half[1:]-fe_half[:-1])/(l_half[1:]-l_half[:-1]) + de[1:-1]) * dt
    # enforcing boundary conditions:
    m1[0] = m[0] + (-(s_half[0]-0.)/(l_half[1]-l_half[0])+dm[0]) * dt # mass flux is zero
    m1[-1] = m[-1] + (-(-mdot-s_half[-1])/(l_half[-1]-l_half[-2])+dm[-1]) * dt  # inflow set to mdot
    edot=-mdot*(vout**2/2.-1./r-0.5*(r*sth*omega)**2)[-1]-4.*vout*pmagout # energy flux from the right boundary
    s1[-1] = -mdot
    e1[0] = e[0]  + (-(fe_half[0]-0.)/(l_half[1]-l_half[0])) * dt #  energy flux is zero
    e1[-1] = e[-1]  +(-(edot-fe_half[-1])/(l_half[-1]-l_half[-2])) * dt  # enegry inlow
    #    s1[0] = s[0] + (-(p_half[0]-pdot)/(l_half[1]-l_half[0])+ds[0]) * dt # zero velocity, finite density (damped)
    s1[0]=0.
    return m1, s1, e1

#########################################################################################
def alltire():
    '''
    the main routine bringing all together.
    '''
    
    sthd=1./sqrt(1.+(dre/re)**2) # initial sin(theta)
    rmax=re*sthd # slightly less then re 
    r=(rmax-rstar)**(arange(nx0)/double(nx0-1))+rstar # very fine radial mesh
    sth, cth, sina, cosa, across, l = geometry(r) # radial-equidistant mesh
    luni=linspace(l.min(), l.max(), nx) # l-equidistant mesh
    luni_half=(luni[1:]+luni[:-1])/2. # half-step l-equidistant mesh
    rfun=interp1d(l,r, kind='linear') # interpolation function mapping l to r
    rnew=rfun(luni) # radial coordinates for the  l-equidistant mesh
    sth, cth, sina, cosa, across, l = geometry(rnew, writeout='geo.dat') # all the geometric quantities for the l-equidistant mesh
    r=rnew # set a grid uniform in l=luni
    r_half=rfun(luni_half) # half-step radial coordinates
    sth_half, cth_half, sina_half, cosa_half, across_half, l_half = geometry(r_half) # mid-step geometry in r
    l_half+=l[1]/2. # mid-step mesh starts halfstep later
    print("halfstep l correction "+str((l_half-luni_half).std())+"\n")
    
    # initial conditions:
    m=zeros(nx) ; s=zeros(nx) ; e=zeros(nx)
    vinit=vout*(r-rstar)/(r+rstar)*sqrt(re/r) # initial velocity
    m=mdot/abs(vinit) # mass distribution
    m0=m 
    s+=vinit*m
    e+=m*(pmagout/(m/across)[-1]*3.+vinit**2/2.-1./r-0.5*(r*sth*omega)**2)
    
    dlmin=(l[1:]-l[:-1]).min()/2.
    dt = dlmin*0.5
    print("dt = "+str(dt))
    #    ti=input("dt")
    
    t=0.; dtout=20. ; tstore=0. ; tmax=1e5 ; nout=0

    ltot=0. # estimated total luminosity
    fflux=open('flux.dat', 'w')
    
    while(t<tmax):
        # first make a preliminary half-step
        mprev=m ; sprev=s ; eprev=e
        rho, v, u = toprim(m, s, e, across, r, sth) # primitive from conserved
        #        v=v/sqrt(1.+v**2) # try relativistic approach?
        #        print(rho)
        #        print(e)
        #        ii=input('m')
        rho_half = (rho[1:]+rho[:-1])/2. ; v_half = (v[1:]+v[:-1])/2.  ; u_half = (u[1:]+u[:-1])/2. 
        dul_half = across_half/(rho_half+1.)*(u[1:]-u[:-1])/(l[1:]-l[:-1]) # radial diffusion
        dul=rho*0.
        dul[1:-1]=(dul_half[1:]+dul_half[:-1])/2. # we need to smooth it for stability
        s, p, fe = fluxes(rho, v, u, across, r, sth)
        fe+=-dul # adding diffusive flux
        wpos=where(rho>rhofloor)
        vr=v*0. ; vl=v*0.
        vr[wpos]=(v+1.)[wpos] ; vl[wpos]=(v-1.)[wpos]
        s_half=solver_hlle(s, m, vl, vr)
        p_half=solver_hlle(p, s, vl, vr)
        fe_half=solver_hlle(fe, e, vl, vr)
        dm, ds, de, flux = sources(rho, v, u, across, r, sth, cth, sina, cosa)
        ltot=simps(flux[1:-1], x=l[1:-1])
        m,s,e=main_step(m,s,e,l_half, s_half,p_half,fe_half, dm, ds, de, dt, r, sth)
        t+=dt
        if(isnan(rho.max()) | (rho.max() > 1e20)):
            print(m)
            print(s)
            return(1)
        if(t>=tstore):
            tstore+=dtout
            print("t = "+str(t*tscale)+"s")
            fflux.write(str(t*tscale)+' '+str(ltot)+'\n')
            fflux.flush()
            #            oneplot(r, rho, name='rhotie{:05d}'.format(nout))
            uplot(r, u, rho, sth, name='utie{:05d}'.format(nout))
            vplot(r, v, sqrt(4./3.*u/rho), name='vtie{:05d}'.format(nout))
            print("mass = "+str(trapz(m[1:-1], x=l[1:-1])))
            print("ltot = "+str(ltot))
            print("energy = "+str(trapz(e[1:-1], x=l[1:-1])))
            print("momentum = "+str(trapz(s[1:-1], x=l[1:-1])))
            # ascii output:
            fname='tireout{:05d}'.format(nout)+'.dat'
            fstream=open(fname, 'w')
            fstream.write('# t = '+str(t)+'\n')
            fstream.write('# format: l -- rho -- v -- u\n')
            for k in arange(nx):
                fstream.write(str(l[k])+' '+str(v[k])+' '+str(u[k])+'\n')
            fstream.close()
            nout+=1
    fflux.close()
# if you want to make a movie of how the velocity changes with time:
# ffmpeg -f image2 -r 35 -pattern_type glob -i 'vtie*0.png' -pix_fmt yuv420p -b 4096k v.mp4
