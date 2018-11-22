from scipy.integrate import *
from scipy.interpolate import *

import numpy.random
from numpy.random import rand
from numpy import *
# import time
import os
import os.path
import imp
import sys

import hdfoutput as hdf
import bassun as bs

'''
we need the option of using an arbitrary configuration file
'''
if(size(sys.argv)>1):
    print("launched with arguments "+str(', '.join(sys.argv)))
    # new conf file
    conf=sys.argv[1]
else:
    conf = 'globals'
'''
A trick (borrowed from where???) that allows to load an arbitrary configuration file instead of the standard "globals.py"
'''
fp, pathname, description = imp.find_module(conf)
imp.load_module('globals', fp, pathname, description)
fp.close()

from globals import *

if ifplot:
    import plots

from timer import Timer
timer = Timer(["total", "step", "io"],
              ["main", "flux", "solver", "toprim", "velocity", "sources"])

# smooth factor for optical depth
def taufun(tau):
    wtrans = where(tau<taumin)
    wopaq = where(tau>taumax)
    wmed = where((tau>=taumin) & (tau<=taumax))
    tt = tau*0.
    if(size(wtrans)>0):
        tt[wtrans] = (tau[wtrans]+abs(tau[wtrans]))/2.
    if(size(wopaq)>0):
        tt[wopaq] = 1.
    if(size(wmed)>0):
        tt[wmed] = 1. - exp(-tau[wmed])
    return tt

# pressure ratios:
def Fbeta(rho, u):
    '''
    calculated F(beta) = pg/p as a function of rho and u (dimensionless units)
    F(beta) itself is F = beta / (1-beta)**0.25 / (1-beta/2)**0.75
    '''
    beta = rho*0.+1.
    wpos=where(u>ufloor)
    if(size(wpos)>0):
        beta[wpos]=betacoeff * rho[wpos] / u[wpos]**0.75
    return beta 

def betafun_define():
    '''
    defines the function to calculate beta as a function of rho/u**0.75
    '''
    bepsilon = 1e-8 ; nb = 1e4
    b1 = 0. ; b2 = 1.-bepsilon
    b = (b2-b1)*arange(nb+1)/double(nb)+b1
    fb = b / (1.-b)**0.25 / (1.-b/2.)**0.75
    bfun = interp1d(fb, b, kind='linear', bounds_error=False, fill_value=1.)
    return bfun

# define once and globally
betafun = betafun_define() # defines the interpolated function for beta

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
    across=4.*pi*afac*dr_e*r_e*(r/r_e)**3/sqrt(1.+3.*cth**2) # follows from Galja's formula (17)
    alpha=arctan((cth**2-1./3.)/sth/cth) # Galja's formula (3)
    sina=sin(alpha) ; cosa=cos(alpha)
    l=cumtrapz(sqrt(1.+3.*cth**2)/2./cth, x=r, initial=0.) # coordinate along the field line
    delta = r * sth/sqrt(1.+3.*cth**2) * dr_e/r_e
    # transverse thickness of the flow (Galya's formula 17)
    # dl diverges near R=Re, hence the maximal radius should be smaller than Re
    # ascii output:
    if(writeout != None):
        theta=arctan(sth/cth)
        fgeo=open(writeout, 'w')
        fgeo.write('# format: r -- theta -- alpha -- across -- l -- delta \n')
        for k in arange(size(l)):
            fgeo.write(str(r[k])+' '+str(theta[k])+' '+str(alpha[k])+' '+str(across[k])+' '+str(l[k])+' '+str(delta[k])+'\n')
        fgeo.close()
    
    return sth, cth, sina, cosa, across, l, delta

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
    beta = betafun(Fbeta(rho, u))
    press = u/3./(1.-beta/2.)
    p=across*(rho*v**2+press) # momentum flux
    fe=across*v*(u+press+(v**2/2.-1./r-0.5*(omega*r*sth)**2)*rho) # energy flux without diffusion
    # flux limiters:
    #    s[0]=0. ; p[0]=u[0]/3.*across[0]; fe[0]=0.
    return s, p, fe

def solver_hll(fs, qs, sl, sr):
    '''
    makes a proxy for a half-step flux, HLL-like (without right-only and left-only regimes, like in HLLE)
    flux of quantity q, density of quantity q, sound velocity to the left, sound velocity to the right
    '''
    #    sr=1.+vshift[1:] ; sl=-1.+vshift[:-1]
    f1,f2,f3 = fs  ;  q1,q2,q3 = qs
    ds=sr[1:]-sl[:-1]
    fhalf1=(f1[1:]+f1[:-1])/2.  ;  fhalf2=(f2[1:]+f2[:-1])/2.  ;  fhalf3=(f3[1:]+f3[:-1])/2.
    fhalf1 = ((sr[1:]*f1[:-1]-sl[:-1]*f1[1:]+sl[:-1]*sr[1:]*(q1[1:]-q1[:-1]))/ds)
    fhalf2 = ((sr[1:]*f2[:-1]-sl[:-1]*f2[1:]+sl[:-1]*sr[1:]*(q2[1:]-q2[:-1]))/ds)
    fhalf3 = ((sr[1:]*f3[:-1]-sl[:-1]*f3[1:]+sl[:-1]*sr[1:]*(q3[1:]-q3[:-1]))/ds)
    return fhalf1, fhalf2, fhalf3

def solver_hlle(fs, qs, sl, sr):
    '''
    makes a proxy for a half-step flux, HLLE-like
    flux of quantity q, density of quantity q, sound velocity to the left, sound velocity to the right
    '''
    #    sr=1.+vshift[1:] ; sl=-1.+vshift[:-1]
    f1,f2,f3 = fs  ;  q1,q2,q3 = qs
    sl1 = minimum(sl, 0.) ; sr1 = maximum(sr, 0.)
    ds=sr1[1:]-sl1[:-1] # see Einfeldt et al. 1991 eq. 4.4
    #    print(ds)
    #    i = input("ds")
    # wreg = where((sr[1:]>=0.)&(sl[:-1]<=0.)&(ds>0.))
    wreg = where(ds>0.)
    #    wleft=where(sr[1:]<0.) ; wright=where(sl[:-1]>0.)
    fhalf1=(f1[1:]+f1[:-1])/2.  ;  fhalf2=(f2[1:]+f2[:-1])/2.  ;  fhalf3=(f3[1:]+f3[:-1])/2.
    if(size(wreg)>0):
        fhalf1[wreg] = ((sr1[1:]*f1[:-1]-sl1[:-1]*f1[1:]+sl1[:-1]*sr1[1:]*(q1[1:]-q1[:-1]))/ds)[wreg] # classic HLLE
        fhalf2[wreg] = ((sr1[1:]*f2[:-1]-sl1[:-1]*f2[1:]+sl1[:-1]*sr1[1:]*(q2[1:]-q2[:-1]))/ds)[wreg] # classic HLLE
        fhalf3[wreg] = ((sr1[1:]*f3[:-1]-sl1[:-1]*f3[1:]+sl1[:-1]*sr1[1:]*(q3[1:]-q3[:-1]))/ds)[wreg] # classic HLLE
    #    if(size(wleft)>0):
    #        fhalf1[wleft]=(f1[1:])[wleft]
    #        fhalf2[wleft]=(f2[1:])[wleft]
    #        fhalf3[wleft]=(f3[1:])[wleft]
    #    if(size(wright)>0):
    #        fhalf1[wright]=(f1[:-1])[wright]
    #        fhalf2[wright]=(f2[:-1])[wright]
    #        fhalf3[wright]=(f3[:-1])[wright]
    return fhalf1, fhalf2, fhalf3

def solver_godunov(fs):
    '''
    simplified Godunov-type solver 
    '''
    f1,f2,f3 = fs
    fhalf1=(f1[1:]+f1[:-1])/2.  ;  fhalf2=(f2[1:]+f2[:-1])/2.  ;  fhalf3=(f3[1:]+f3[:-1])/2.
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
    taufac = taufun(tau)    # 1.-exp(-tau)
    gamefac = taufac/tau
    gamedd = eta * ltot * gamefac 
    force = (-(sina*cth+cosa*sth)/r**2*(1.-gamedd)+omega**2*r*sth*cosa)*rho*across*taufac
    beta = betafun(Fbeta(rho, u))
    qloss = u * (1.-beta)/(1.-beta/2.)/(xirad*tau+1.)*8.*pi*r*sth*afac*taufac # diffusion approximations; energy lost from 4 sides
    irradheating = heatingeff * gamedd * (sina*cth+cosa*sth)/r**2 # photons absorbed by the matter also heat it
    #    qloss[wtrans] = (u * (1.-beta)/(1.-beta/2.) * taufac *4.*pi*r*sth*afac)[wtrans]
    #    qloss*=0.       
    #    work=v*force
    return rho*0., force, v*force-qloss+irradheating, qloss

def toprim(m, s, e, across, r, sth):
    '''
    convert conserved quantities to primitives
    '''
    rho=m/across
    v=s*0.
    v[rho>rhofloor]=(s/m)[rho>rhofloor]
    u=(e-m*(v**2/2.-1./r-0.5*(r*sth*omega)**2))/across
    u[u<=ufloor]=0.
    beta = betafun(Fbeta(rho, u))
    return rho, v, u, u*(1.-beta)/(1.-beta/2.)

def main_step(m, s, e, l_half, s_half, p_half, fe_half, dm, ds, de, dt, r, sth, across, dlleft, dlright):
    '''
    main advance in a dt step
    input: three densities, l (midpoints), three fluxes (midpoints), three sources, timestep, r, sin(theta), cross-section
    output: three new densities
    includes boundary conditions!
    '''
    #    dlleft = l_half[1]-l_half[0]
    #    dlright = l_half[-1]-l_half[-2]
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
    m1[0] = m[0] + (-(s_half[0]-(-mdot0))/dlleft+dm[0]) * dt # mass flux is zero through the inner boundary
    m1[-1] = m[-1] + (-(-mdot-s_half[-1])/dlright+dm[-1]) * dt  # inflow set to mdot (global)
    # this effectively limits the outer velocity from above
    if(m1[-1]< (mdot / abs(vout))):
        m1[-1] = mdot / abs(vout)
    s1[0] = -mdot0
    s1[-1] = -mdot # if I fix s1[-1], this results in v=vout effectively fixed at the boundary
    vout_current = s1[-1]/m1[-1]
    if galyamode:
        e1[0] = across[0] * umag + m1[0] * (-1./r[0]) # *(u+rho*(v**2/2.- 1./r - 0.5*(omega*r*sth)**2))*across
    else:
        if coolNS:
            e1[0] = - (m1/r)[0]
        else:
            e1[0] = e[0] + (-(fe_half[0]-edot0)/dlleft) * dt #  energy flux is zero
    if ufixed:
        e1[-1] = (m1*(vout**2/2.-1./r-0.5*(r*sth*omega)**2)+3.*across*umagout)[-1] # fixing internal energy at the outer rim (!!!assumed radiation domination) leads to -1 velocity at the outer boundary; can we just use vout here?
    else:
        edot = -mdot*(vout**2/2.-1./r-0.5*(r*sth*omega)**2)[-1]+4.*across[-1]*vout*umagout # energy flux from the right boundary (!!!assumed radiation domination)
        e1[-1] = e[-1] + (-(edot-fe_half[-1])/dlright) * dt  # energy inlow
        #    s1[0] = s[0] + (-(p_half[0]-pdot)/(l_half[1]-l_half[0])+ds[0]) * dt # zero velocity, finite density (damped)
    # what if we get negative mass?
    wneg=where(m1<mfloor)
    m1 = maximum(m1, mfloor)
    if(size(wneg)>0):
        s1[wneg] = (m1 * s/m)[wneg]
        e1[wneg] = (e1 * e/m)[wneg]
    
    return m1, s1, e1

################################################################################
def alltire():
    '''
    the main routine bringing all together.
    '''
    timer.start("total")

    # if the outpur directory does not exist:
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    sthd=1./sqrt(1.+(dr_e/r_e)**2) # initial sin(theta)
    rmax=r_e*sthd # slightly less then r_e 
    r=((2.*(rmax-rstar)/rstar)**(arange(nx0)/double(nx0-1))+1.)*(rstar/2.) # very fine radial mesh
    sth, cth, sina, cosa, across, l, delta = geometry(r) # radial-equidistant mesh
    l += r.min() # we are starting from a finite radius
    if(logmesh):
        luni=exp(linspace(log(l.min()), log(l.max()), nx, endpoint=False)) # log(l)-equidistant mesh
    else:
        luni=linspace(l.min(), l.max(), nx, endpoint=False)
    l -= r.min() ; luni -= r.min()
    luni_half=(luni[1:]+luni[:-1])/2. # half-step l-equidistant mesh
    rfun=interp1d(l,r, kind='linear') # interpolation function mapping l to r
    rnew=rfun(luni) # radial coordinates for the  l-equidistant mesh
    sth, cth, sina, cosa, across, l, delta = geometry(rnew, writeout=outdir+'/geo.dat') # all the geometric quantities for the l-equidistant mesh
    r=rnew # set a grid uniform in l=luni
    r_half=rfun(luni_half) # half-step radial coordinates
    sth_half, cth_half, sina_half, cosa_half, across_half, l_half, delta_half = geometry(r_half) # mid-step geometry in r
    l_half+=l[1]/2. # mid-step mesh starts halfstep later
    dlleft = 2.*(l_half[1]-l_half[0])-(l_half[2]-l_half[1])
    dlright = 2.*(l_half[-1]-l_half[-2])-(l_half[-2]-l_half[3])

    # testing bassun.py
    print("delta = "+str((across/(4.*pi*afac*r*sth))[0]))
    print("delta = "+str((sth*r/sqrt(1.+3.*cth**2))[0] * dr_e/r_e))
    print("delta = "+str(delta[0]))
    BSgamma = (across/delta**2)[0]/mdot*rstar
    BSeta = (8./21./sqrt(2.)*umag)**0.25*sqrt(delta[0])/(rstar)**0.125
    print("BS parameters:")
    print("   gamma = "+str(BSgamma))
    print("   eta = "+str(BSeta))
    x1 = 1. ; x2 = 1000. ; nxx=1000
    xtmp=(x2/x1)**(arange(nxx)/double(nxx))*x1
    plots.someplots(xtmp, [bs.fxis(xtmp, BSgamma, BSeta, 3.)], name='fxis', ytitle=r'$F(x)$')
    xs = bs.xis(BSgamma, BSeta)
    print("   xi_s = "+str(xs))
    input("BS")
    # magnetic field energy density:
    umagtar = umag * (1.+3.*cth**2)/4. * (rstar/r)**6
    # initial conditions:
    m=zeros(nx) ; s=zeros(nx) ; e=zeros(nx)
    vinit=vout*((r-rstar)/(r+rstar))*sqrt(r_e/r) # initial velocity
    m=mdot/fabs(vout*sqrt(r_e/r)) # mass distribution
    m0=m 
    s+=vinit*m
    e+=(vinit**2/2.-1./r-0.5*(r*sth*omega)**2)*m+3.*umagout*across*(r_e/r)**(-10./3.) * (1.+0.01*rand(size(r)))
    
    dlmin=(l_half[1:]-l_half[:-1]).min()
    dt = dlmin*0.5
    print("dt = "+str(dt))
    #    ti=input("dt")
    
    t=0.;  tstore=0.  ; nout=0

    ltot=0. # estimated total luminosity
    fflux=open(outdir+'/'+'flux.dat', 'w')
    ftot=open(outdir+'/'+'totals.dat', 'w')
    if(ifhdf):
        hname = outdir+'/'+'tireout.hdf5'
        hfile = hdf.init(hname, l, r, sth, cth) # , m1, mdot, eta, afac, re, dre, omega)
    
    timer.start("total")
    while(t<tmax):
        # first make a preliminary half-step
        timer.start_comp("toprim")
        mprev=m ; sprev=s ; eprev=e
        rho, v, u, urad = toprim(m, s, e, across, r, sth) # primitive from conserved
        timer.stop_comp("toprim")
        timer.start_comp("flux")
        rho_half = (rho[1:]+rho[:-1])/2. ; v_half = (v[1:]+v[:-1])/2.  ; u_half = (u[1:]+u[:-1])/2. 
        dul_half = across_half/(rho_half+1.)*(urad[1:]-urad[:-1])/(l[1:]-l[:-1])/3. # radial diffusion
        dul = rho*0. # just setting the size of the array
        dul[1:-1]=(dul_half[1:]+dul_half[:-1])/2. # we need to smooth it for stability
        s, p, fe = fluxes(rho, v, u, across, r, sth)
        fe+=-dul # adding diffusive flux 
        timer.stop_comp("flux")
        timer.start_comp("velocity")
        wpos=where((rho>rhofloor)&(u>ufloor))
        vr=v+1. ; vl=v-1.
        #        vr[wpos]=v[wpos]+1. ; vl[wpos]=v[wpos]-1.
        cs = v*0.
        cs[wpos]=sqrt(4.*u[wpos]/(rho[wpos]+u[wpos])) # slightly over-estimating the SOS to get stable signal velocities; exact for radiation-dominated
        vr[wpos]=(v+cs)[wpos] ; vl[wpos]=(v-cs)[wpos]
        timer.stop_comp("velocity")
        timer.start_comp("solver")
        #        print(sum(rho>rhofloor))
        s_half, p_half, fe_half = solver_hlle([s, p, fe], [m, s, e], vl, vr)
        # solver_godunov([s, p, fe]) 
        timer.stop_comp("solver")
        timer.start_comp("sources")
        dm, ds, de, flux = sources(rho, v, u, across, r, sth, cth, sina, cosa,ltot=ltot)
        #        ltot=simps(flux[1:-1], x=l[1:-1])
        #        ltot=trapz(flux[1:-1], x=l[1:-1]) # 
        ltot=simps(flux, x=l) # no difference
        timer.stop_comp("sources")
        timer.start_comp("main")
        m,s,e=main_step(m,s,e,l_half, s_half,p_half,fe_half, dm, ds, de, dt, r, sth, across, dlleft, dlright)
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
            #            oneplot(r, rho, name=outdir+'/rhotie{:05d}'.format(nout))
            if ifplot & (nout%plotalias == 0):
                plots.uplot(r, u, rho, sth, v, name=outdir+'/utie{:05d}'.format(nout))
                plots.vplot(r, v, sqrt(4./3.*u/rho), name=outdir+'/vtie{:05d}'.format(nout))
                plots.someplots(r, [u/rho**(4./3.)], name=outdir+'/entropy{:05d}'.format(nout), ytitle=r'$S$', ylog=True)
                plots.someplots(r, [(u-urad)/(u-urad/2.), 1.-(u-urad)/(u-urad/2.)],
                                name=outdir+'/beta{:05d}'.format(nout), ytitle=r'$\beta$, $1-\beta$', ylog=True)
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
                    fname=outdir+'/tireout{:05d}'.format(nout)+'.dat'
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
# ffmpeg -f image2 -r 15 -pattern_type glob -i 'out/vtie*0.png' -pix_fmt yuv420p -b 4096k v.mp4
# alltire()
