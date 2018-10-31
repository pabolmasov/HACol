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

from timer import Timer
timer = Timer(["total", "step", "io"],
              ["main", "flux", "solver", "toprim", "velocity", "sources"])

###########################################################################################
def geometry(r, writeout=None):
    '''
    computes all the geometrical quantities. Sufficient to run once before the start of the simulation.
    Output: sin(theta), cos(theta), sin(alpha), cos(alpha), tangential cross-section area, 
    and l (zero at the surface, growing with radius)
    adding nontrivial writeout key allows to write the geometry to an ascii file 
    '''
    #    theta=arcsin(sqrt(r/r_e))
    #    sth=sin(theta) ; cth=cos(theta)
    sth=sqrt(r/r_e) ; cth=sqrt(1.-r/r_e) # OK
    across=4.*pi*afac*dr_e*r_e*sth**3/sqrt(1.+3.*cth**2) # OK 
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

def solver_hlle(fs, qs, sl, sr):
    '''
    makes a proxy for a half-step flux, HLLE-like
    flux of quantity q, density of quantity q, sound velocity to the left, sound velocity to the right
    '''
    #    sr=1.+vshift[1:] ; sl=-1.+vshift[:-1]
    f1,f2,f3 = fs  ;  q1,q2,q3 = qs
    ds=sr[1:]-sl[:-1]
    wreg=where((sr[1:]>0.)&(sl[:-1]<0.))
    wleft=where(sr[1:]<0.) ; wright=where(sl[:-1]>0.)
    fhalf1=(f1[1:]+f1[:-1])/2.  ;  fhalf2=(f2[1:]+f2[:-1])/2.  ;  fhalf3=(f3[1:]+f3[:-1])/2.
    if(size(wreg)>0):
        fhalf1[wreg] = ((sr[1:]*f1[:-1]-sl[:-1]*f1[1:]+sl[:-1]*sr[1:]*(q1[1:]-q1[:-1]))/ds)[wreg] # classic HLLE
        fhalf2[wreg] = ((sr[1:]*f2[:-1]-sl[:-1]*f2[1:]+sl[:-1]*sr[1:]*(q2[1:]-q2[:-1]))/ds)[wreg] # classic HLLE
        fhalf3[wreg] = ((sr[1:]*f3[:-1]-sl[:-1]*f3[1:]+sl[:-1]*sr[1:]*(q3[1:]-q3[:-1]))/ds)[wreg] # classic HLLE
    if(size(wleft)>0):
        fhalf1[wleft]=(f1[:-1])[wleft]
        fhalf2[wleft]=(f2[:-1])[wleft]
        fhalf3[wleft]=(f3[:-1])[wleft]
    if(size(wright)>0):
        fhalf1[wright]=(f1[1:])[wright]
        fhalf2[wright]=(f2[1:])[wright]
        fhalf3[wright]=(f3[1:])[wright]
    return fhalf1, fhalf2, fhalf3

def sources(rho, v, u, across, r, sth, cth, sina, cosa, ltot=0.):
    '''
    computes the RHSs of conservation equations
    no changes in mass
    momentum injection through gravitational and centrifugal forces
    energy losses through the surface
    outputs: dm, ds, de, and separately the amount of energy radiated per unit length per unit time ("flux")
    '''
    #  sinsum=sina*cth+cosa*sth # cos( pi/2-theta + alpha) = sin(theta-alpha)
    tau = rho*across/(4.*pi*r*sth*afac)
    taufac = 1.-exp(-tau)
    gamefac = taufac/tau
    # transparent regions
    wtrans=where(tau < taumin)
    taufac[wtrans] = (tau + abs(tau))[wtrans]/2.
    gamefac[wtrans] = 1.
    gamedd = eta * ltot * gamefac 
    force = (-(sina*cth+cosa*sth)/r**2*(1.-gamedd)+omega**2*r*sth*cosa)*rho*across*taufac
    qloss = u/(xirad*tau+1.)*4.*pi*r*sth*afac*taufac # optically thick regime
    #    qloss*=0.       
    #    work=v*force
    return rho*0., force, v*force-qloss, qloss

def toprim(m, s, e, across, r, sth):
    '''
    convert conserved quantities to primitives
    '''
    rho=m/across
    v=s*0.
    v[rho>rhofloor]=(s/m)[rho>rhofloor]
    u=(e-m*(v**2/2.-1./r-0.5*(r*sth*omega)**2))/across
    u[u<=ufloor]=0.
    return rho, v, u

def main_step(m, s, e, l_half, s_half, p_half, fe_half, dm, ds, de, dt, r, sth, across):
    '''
    main advance in a dt step
    input: three densities, l (midpoints), three fluxes (midpoints), three sources, timestep, r, sin(theta), cross-section
    output: three new densities
    includes boundary conditions!
    '''
    nl=size(m)
    m1=zeros(nl) ; s1=zeros(nl); e1=zeros(nl)
    m1[1:-1] = m[1:-1]+ (-(s_half[1:]-s_half[:-1])/(l_half[1:]-l_half[:-1]) + dm[1:-1]) * dt
    s1[1:-1] = s[1:-1]+ (-(p_half[1:]-p_half[:-1])/(l_half[1:]-l_half[:-1]) + ds[1:-1]) * dt 
    e1[1:-1] = e[1:-1]+ (-(fe_half[1:]-fe_half[:-1])/(l_half[1:]-l_half[:-1]) + de[1:-1]) * dt
    # enforcing boundary conditions:
    if(m[0]>mfloor):
        mdot0 = mdotsink # sink present when mass density is positive at the inner boundary
        edot0 = -mdotsink*e[0]/m[0] # energy sink 
    else:
        mdot0 = 0.
        edot0 = 0.
    m1[0] = m[0] + (-(s_half[0]-(-mdot0))/(l_half[1]-l_half[0])+dm[0]) * dt # mass flux is zero through the inner boundary
    m1[-1] = m[-1] + (-(-mdot-s_half[-1])/(l_half[-1]-l_half[-2])+dm[-1]) * dt  # inflow set to mdot (global)
    s1[0] = -mdot0
    s1[-1] = -mdot # if I fix s1[-1], this results in v=vout effectively fixed at the boundary
    vout_current = s1[-1]/m1[-1]
    if galyamode:
        e1[0] = across[0] * umag + m1[0] * (-1./r[0]) # *(u+rho*(v**2/2.- 1./r - 0.5*(omega*r*sth)**2))*across
    else:
        if coolNS:
            e1[0] = - (m1/r)[0]
        else:
            e1[0] = e[0] + (-(fe_half[0]-edot0)/(l_half[1]-l_half[0])) * dt #  energy flux is zero
    if ufixed:
        e1[-1] = (m1*(vout_current**2/2.-1./r-0.5*(r*sth*omega)**2)+3.*across*umagout)[-1] # fixing internal energy at the outer rim
    else:
        edot = -mdot*(vout_current**2/2.-1./r-0.5*(r*sth*omega)**2)[-1]+4.*across[-1]*vout_current*umagout # energy flux from the right boundary
        e1[-1] = e[-1]  + (-(edot-fe_half[-1])/(l_half[-1]-l_half[-2])) * dt  # enegry inlow
    #    s1[0] = s[0] + (-(p_half[0]-pdot)/(l_half[1]-l_half[0])+ds[0]) * dt # zero velocity, finite density (damped)
    return m1, s1, e1

#########################################################################################
def alltire():
    '''
    the main routine bringing all together.
    '''
    timer.start("total")
    
    sthd=1./sqrt(1.+(dr_e/r_e)**2) # initial sin(theta)
    rmax=r_e*sthd # slightly less then r_e 
    r=((2.*(rmax-rstar)/rstar)**(arange(nx0)/double(nx0-1))+1.)*(rstar/2.) # very fine radial mesh
    sth, cth, sina, cosa, across, l = geometry(r) # radial-equidistant mesh
    l += r.min() # we are starting from a finite radius
    if(logmesh):
        luni=exp(linspace(log(l.min()), log(l.max()), nx, endpoint=False)) # log(l)-equidistant mesh
    else:
        luni=linspace(l.min(), l.max(), nx, endpoint=False)
    l -= r.min() ; luni -= r.min()
    luni_half=(luni[1:]+luni[:-1])/2. # half-step l-equidistant mesh
    rfun=interp1d(l,r, kind='linear') # interpolation function mapping l to r
    print("r = "+str(r))
    print("l = "+str(l))
    print("luni = "+str(luni))
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
    vinit=vout*(r-rstar)/(r+rstar)*sqrt(r_e/r) # initial velocity
    m=mdot/(abs(vout)+abs(vinit))*(r/r_e)**2 # mass distribution
    m0=m 
    s+=vinit*m
    e+=(vinit**2/2.-1./r-0.5*(r*sth*omega)**2)*m+3.*umagout*across

    # magnetic field energy density:
    umagtar = umag * (1.+3.*cth**2)/4. * (rstar/r)**6
    
    dlmin=(l[1:]-l[:-1]).min()/2.
    dt = dlmin*0.25
    print("dt = "+str(dt))
    #    ti=input("dt")
    
    t=0.;  tstore=0.  ; nout=0

    ltot=0. # estimated total luminosity
    fflux=open('flux.dat', 'w')
    ftot=open('totals.dat', 'w')
    if(ifhdf):
        hname = 'tireout.hdf5'
        hfile = hdf.init(hname, l, r, sth, cth) # , m1, mdot, eta, afac, re, dre, omega)
    
    timer.start("total")
    while(t<tmax):
        # first make a preliminary half-step
        timer.start_comp("toprim")
        mprev=m ; sprev=s ; eprev=e
        rho, v, u = toprim(m, s, e, across, r, sth) # primitive from conserved
        timer.stop_comp("toprim")
        timer.start_comp("flux")
        rho_half = (rho[1:]+rho[:-1])/2. ; v_half = (v[1:]+v[:-1])/2.  ; u_half = (u[1:]+u[:-1])/2. 
        dul_half = across_half/(rho_half+1.)*(u[1:]-u[:-1])/(l[1:]-l[:-1])/3. # radial diffusion
        dul = rho*0. # just setting the size of the array
        dul[1:-1]=(dul_half[1:]+dul_half[:-1])/2. # we need to smooth it for stability
        s, p, fe = fluxes(rho, v, u, across, r, sth)
        fe+=-dul # adding diffusive flux !!!
        timer.stop_comp("flux")
        timer.start_comp("velocity")
        #        wpos=where((rho>rhofloor)&(u>ufloor))
        vr=v+1. ; vl=v-1. ; cs=v*0.+1.
        #        cs[wpos]=sqrt(4./3.*u[wpos]/rho[wpos])
        #        vr[wpos]=(v+cs)[wpos] ; vl[wpos]=(v-cs)[wpos]
        timer.stop_comp("velocity")
        timer.start_comp("solver")
        #        print(sum(rho>rhofloor))
        s_half, p_half, fe_half = solver_hlle([s, p, fe], [m, s, e], vl, vr)
        timer.stop_comp("solver")
        timer.start_comp("sources")
        dm, ds, de, flux = sources(rho, v, u, across, r, sth, cth, sina, cosa,ltot=ltot)
        ltot=simps(flux[1:-1], x=l[1:-1])
        timer.stop_comp("sources")
        timer.start_comp("main")
        m,s,e=main_step(m,s,e,l_half, s_half,p_half,fe_half, dm, ds, de, dt, r, sth, across)
        timer.stop_comp("main")
        timer.lap("step")
        t+=dt
        if(abs(s).max()>1e20):
            print("m is positive in "+str(sum(m>mfloor))+" points")
            print("s is positive in "+str(sum(abs(s)>mfloor))+" points")
            print("p is positive in "+str(sum(abs(p)>mfloor))+" points")
            winf=(abs(s)>1e20)
            print("t = "+str(t))
            print(r[winf]/rstar)
            print(m[winf])
            print(s[winf])
            print(e[winf])
            print(de[winf])
            print(flux[winf])
            if(ifhdf):
                hdf.close(hfile)
            ss=input('s')
        if(isnan(rho.max()) | (rho.max() > 1e20)):
            print(m)
            print(s)
            if(ifhdf):
                hdf.close(hfile)
            return(1)
        if(t>=tstore):
            timer.start("io")
            tstore+=dtout
            print("t = "+str(t*tscale)+"s")
            fflux.write(str(t*tscale)+' '+str(ltot)+'\n')
            fflux.flush()
            #            oneplot(r, rho, name='rhotie{:05d}'.format(nout))
            if ifplot & (nout%plotalias == 0):
                plots.uplot(r, u, rho, sth, v, name='utie{:05d}'.format(nout))
                plots.vplot(r, v, sqrt(4./3.*u/rho), name='vtie{:05d}'.format(nout))
                plots.splot(r, u/rho**(4./3.), name='entropy{:05d}'.format(nout))
            mtot=simps(m[1:-1], x=l[1:-1])
            etot=simps(e[1:-1], x=l[1:-1])
            print("mass = "+str(mtot))
            print("ltot = "+str(ltot))
            print("energy = "+str(etot))
            print("momentum = "+str(trapz(s[1:-1], x=l[1:-1])))
            
            ftot.write(str(t*tscale)+' '+str(mtot)+' '+str(etot)+'\n')
            ftot.flush()
            if(ifhdf):
                hdf.dump(hfile, nout, t, rho, v, u)
            if not(ifhdf) or (nout%ascalias == 0):
                    # ascii output:
                    fname='tireout{:05d}'.format(nout)+'.dat'
                    fstream=open(fname, 'w')
                    fstream.write('# t = '+str(t*tscale)+'s\n')
                    fstream.write('# format: r/rstar -- rho -- v -- u/umag\n')
                    for k in arange(nx):
                        fstream.write(str(r[k]/rstar)+' '+str(rho[k])+' '+str(v[k])+' '+str(u[k]/umagtar[k])+'\n')
                    fstream.close()
                #print simulation run-time statistics
            timer.stop("io")
            timer.stats("step")
            timer.stats("io")
            timer.comp_stats()

            timer.start("step") #refresh lap counter (avoids IO profiling)
            timer.purge_comps()
   
            nout+=1
    fflux.close()
    ftot.close()
    if(ifhdf):
        hdf.close(hfile)
# if you want to make a movie of how the velocity changes with time:
# ffmpeg -f image2 -r 35 -pattern_type glob -i 'vtie*0.png' -pix_fmt yuv420p -b 4096k v.mp4
# alltire()
