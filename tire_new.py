from scipy.integrate import *
from scipy.interpolate import *

import numpy.random
# import time
import os
import os.path

import hdfoutput as hdf
from globals import *
if ifplot:
    import plots

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
    fhalf=(f[1:]+f[:-1])/2.
    if(size(wpos)>0):
        fhalf[wpos] = ((sr[1:]*f[:-1]-sl[:-1]*f[1:]+sl[:-1]*sr[1:]*(q[1:]-q[:-1]))/ds)[wpos] # classic HLLE
    return fhalf

def sources(rho, v, u, across, r, sth, cth, sina, cosa, ltot=0.):
    '''
    computes the RHSs of conservation equations
    no changes in mass
    momentum injection through gravitational and centrifugal forces
    energy losses through the surface
    outputs: dm, ds, de, and separately the amount of energy radiated per unit length per unit time ("flux")
    '''
    sinsum=sina*cth+cosa*sth # cos( pi/2-theta + alpha) = sin(theta-alpha)
    tau = rho*across/(4.*pi*r*sth)
    gamedd=eta * ltot / tau 
    force=(-sinsum/r**2*(1.-gamedd)+omega**2*r*sth*cosa)*rho*across
    qloss=u/(tau*xirad+1.)*r*4.*pi*sth*afac
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
    edot=-mdot*(vout**2/2.-1./r-0.5*(r*sth*omega)**2)[-1]+4.*re*dre*afac*vout*pmagout # energy flux from the right boundary
    s1[0]=0.
    s1[-1] = -mdot
    e1[0] = e[0]  + (-(fe_half[0]-0.)/(l_half[1]-l_half[0])) * dt #  energy flux is zero
    e1[-1] = e[-1]  +(-(edot-fe_half[-1])/(l_half[-1]-l_half[-2])) * dt  # enegry inlow
    #    s1[0] = s[0] + (-(p_half[0]-pdot)/(l_half[1]-l_half[0])+ds[0]) * dt # zero velocity, finite density (damped)
    return m1, s1, e1

#########################################################################################
def alltire():
    '''
    the main routine bringing all together.
    '''
    
    sthd=1./sqrt(1.+(dre/re)**2) # initial sin(theta)
    rmax=re*sthd # slightly less then re 
    r=(((rmax-rstar)/rstar)**(arange(nx0)/double(nx0-1))+1.)*rstar # very fine radial mesh
    sth, cth, sina, cosa, across, l = geometry(r) # radial-equidistant mesh
    l += r.min() # we are starting from a finite radius
    luni=exp(linspace(log(l.min()), log(l.max()), nx, endpoint=False)) # log(l)-equidistant mesh
    l -= r.min() ; luni -= r.min()
    luni_half=(luni[1:]+luni[:-1])/2. # half-step l-equidistant mesh
    rfun=interp1d(l,r, kind='linear') # interpolation function mapping l to r
    #  print("r = "+str(r))
    #  print("l = "+str(l))
    #  print("luni = "+str(luni))
    rnew=rfun(luni) # radial coordinates for the  l-equidistant mesh
    sth, cth, sina, cosa, across, l = geometry(rnew, writeout='geo.dat') # all the geometric quantities for the l-equidistant mesh
    r=rnew # set a grid uniform in l=luni
    r_half=rfun(luni_half) # half-step radial coordinates
    sth_half, cth_half, sina_half, cosa_half, across_half, l_half = geometry(r_half) # mid-step geometry in r
    l_half+=l[1]/2. # mid-step mesh starts halfstep later
#    print("halfstep l correction "+str((l_half-luni_half).std())+"\n")
#    print("r mesh: "+str(r))
#    ii=input("r")
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
    
    t=0.;  tstore=0.  ; nout=0

    ltot=0. # estimated total luminosity
    fflux=open('flux.dat', 'w')
    ftot=open('totals.dat', 'w')
    if(ifhdf):
        hname = 'tireout.hdf5'
        hfile = hdf.init(hname, l, r, sth, cth, m1, mdot, eta, afac, re, dre, omega)
    
    while(t<tmax):
        # first make a preliminary half-step
        mprev=m ; sprev=s ; eprev=e
        rho, v, u = toprim(m, s, e, across, r, sth) # primitive from conserved
        #        v=v/sqrt(1.+v**2) # try relativistic approach?
        #        print(rho)
        #        print(e)
        #        ii=input('m')
        rho_half = (rho[1:]+rho[:-1])/2. ; v_half = (v[1:]+v[:-1])/2.  ; u_half = (u[1:]+u[:-1])/2. 
        dul_half = across_half/(rho_half+1.)*(u[1:]-u[:-1])/(l[1:]-l[:-1])/3. # radial diffusion
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
            if ifplot & (nout%plotalias == 0):
                plots.uplot(r, u, rho, sth, v, name='utie{:05d}'.format(nout))
                plots.vplot(r, v, sqrt(4./3.*u/rho), name='vtie{:05d}'.format(nout))
            mtot=trapz(m[1:-1], x=l[1:-1])
            etot=trapz(e[1:-1], x=l[1:-1])
            print("mass = "+str(mtot))
            print("ltot = "+str(ltot))
            print("energy = "+str(etot))
            print("momentum = "+str(trapz(s[1:-1], x=l[1:-1])))
            
            ftot.write(str(t*tscale)+' '+str(mtot)+' '+str(etot)+'\n')
            ftot.flush()
            if(ifhdf):
                hdf.dump(hfile, nout, t, rho, v, u)
            else:
                # ascii output:
                fname='tireout{:05d}'.format(nout)+'.dat'
                fstream=open(fname, 'w')
                fstream.write('# t = '+str(t)+'\n')
                fstream.write('# format: l -- rho -- v -- u\n')
                for k in arange(nx):
                    fstream.write(str(l[k])+' '+str(rho[k])+''+str(v[k])+' '+str(u[k])+'\n')
                fstream.close()
            nout+=1
    fflux.close()
    ftot.close()
    if(ifhdf):
        hdf.close(hfile)
# if you want to make a movie of how the velocity changes with time:
# ffmpeg -f image2 -r 35 -pattern_type glob -i 'vtie*0.png' -pix_fmt yuv420p -b 4096k v.mp4
