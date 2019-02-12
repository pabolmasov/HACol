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

# loading local modules:
if ifplot:
    import plots
import hdfoutput as hdf
import bassun as bs
import solvers as solv
from sigvel import *
from geometry import *

from timer import Timer
timer = Timer(["total", "step", "io"],
              ["advance"])

#
def regularize(u, rho):
    '''
    if internal energy goes below ufloor, we heat the matter up artificially
    '''
    u1=u-ufloor ; rho1=rho-rhofloor
    return (u1+fabs(u1))/2.+ufloor, (rho1+fabs(rho1))/2.+rhofloor

#
def quasirelfunction(v, v0):
    '''
    this function matches f(v)=v below v0 and approaches 1 at v \to 0 
    '''
    sv0 = sqrt(1.+v0**2) ; sv = sqrt(1.+v**2)
    a = (1.+2.*v0**2)/sv0 ; b= -v0**3/sv0
    return (a*abs(v)+b)/sv*sign(v)

# speed of sound multiplier (see Chandrasekhar 1967 or Johnson 2008):
def Gamma1(gamma, beta):
    g1 = gamma - 1.
    return beta + 9. * g1 * (beta-4./3.)**2/(beta+12.*g1 * (1.-beta))

# smooth factor for optical depth
def taufun(tau):
    '''
    calculates 1-exp(-x) in a reasonably smooth way trying to avoid round-off errors for small and large x
    '''
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

def tratfac(x):
    '''
    the correction factor used when local thermal time scales are small
    '''
    xmin = taumin ; xmax = taumax # limits the same as for optical depth
    tt=x*0.
    w1 = where(x<= xmin) ;  w2 = where(x>= xmax) ; wmed = where((x < xmax) & (x > xmin))
    if(size(w1)>0):
        tt[w1] = 1.
    if(size(w2)>0):
        tt[w2] = 1./x[w2]
    if(size(wmed)>0):
        tt[wmed] = (1.-exp(-x[wmed]))/x[wmed]
    wnan=where(isnan(x))
    if(size(wnan)>0):
        tt[wnan] = 0.
        print("trat = "+str(x.min())+".."+str(x.max()))
        ip = input('trat')
    return tt

# pressure ratios:
def Fbeta(rho, u):
    '''
    calculates a function of 
    beta = pg/p from rho and u (dimensionless units)
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

def cons(rho, v, u, g):
    '''
    computes conserved quantities from primitives
    '''
    m=rho*g.across # mass per unit length
    s=m*v # momentum per unit length
    e=(u+rho*(v**2/2.- 1./g.r - 0.5*(omega*g.r*g.sth)**2))*g.across  # total energy (thermal + mechanic) per unit length
    return m, s, e

def diffuse(rho, urad, dl, across_half):
    '''
    radial energy diffusion;
    calculates energy flux contribution already at the cell boundary
    '''
    #    rho_half = (rho[1:]+rho[:-1])/2. # ; v_half = (v[1:]+v[:-1])/2.  ; u_half = (u[1:]+u[:-1])/2.
    rtau_right = rho[1:] * dl / 2.# optical depths along the field line, to the right of the cell boundaries
    rtau_left = rho[:-1] * dl / 2.# -- " -- to the left -- " --
    dul_half = across_half*((urad * tratfac(rtau_right))[1:] - (urad * tratfac(rtau_right))[:-1])/6. 
    #    ((urad/(rho+1.)/dl)[1:]*(1.-exp(-rtau_right))-(urad/(rho+1.)/dl)[:-1]*(1.-exp(-rtau_left)))/3. # radial diffusion
    # introducing exponential factors helps reduce the numerical noise from rho variations
    return -dul_half 

def fluxes(rho, v, u, g):
    '''
    computes the fluxes of conserved quantities, given primitives; 
    radiation diffusion flux is not included, as it is calculated at halfpoints
    inputs:
    rho -- density, v -- velocity, u -- thermal energy density
    g is geometry (structure)
    '''
    s = rho*v*g.across # mass flux (identical to momentum per unit length -- can we use it?)
    beta = betafun(Fbeta(rho, u))
    press = u/3./(1.-beta/2.)
    p = g.across*(rho*v**2+press) # momentum flux
    fe = g.across*v*(u+press+(v**2/2.-1./g.r-0.5*(omega*g.r*g.sth)**2)*rho) # energy flux without diffusion
    return s, p, fe

def sources(rho, v, u, g, ltot=0., dt=None):
    '''
    computes the RHSs of conservation equations
    no changes in mass
    momentum injection through gravitational and centrifugal forces
    energy losses through the surface
    outputs: dm, ds, de, and separately the amount of energy radiated per unit length per unit time ("flux")
    additional output:  equilibrium energy density
    '''
    #  sinsum=sina*cth+cosa*sth # cos( pi/2-theta + alpha) = sin(theta-alpha)
    tau = rho*g.across/(4.*pi*g.r*g.sth*afac)
    taufac = taufun(tau)    # 1.-exp(-tau)
    gamefac = taufac/tau
    gamedd = eta * ltot * gamefac
    sinsum = (g.sina*g.cth+g.cosa*g.sth)# sin(theta+alpha)
    force = (-sinsum/g.r**2*(1.-gamedd)+omega**2*g.r*g.sth*g.cosa)*rho*g.across #*taufac
    beta = betafun(Fbeta(rho, u))
    urad = u * (1.-beta)/(1.-beta/2.)
    qloss = urad/(xirad*tau+1.)*8.*pi*g.r*g.sth*afac*taufac  # diffusion approximations; energy lost from 4 sides
    irradheating = heatingeff * mdot / g.r * g.sth * sinsum
    # heatingeff * gamedd * (g.sina*g.cth+g.cosa*g.sth)/g.r**2*8.*pi*g.r*g.sth*afac*taufac # photons absorbed by the matter also heat it
    if(dt is not None):            
        trat = qloss * dt / u
        #        qloss *= tratfac(trat)
    else:
        trat = u*0. +1.
    ueq = heatingeff * mdot / g.r**2 * sinsum * urad/(xirad*tau+1.)
    dudt = v*force-qloss+irradheating
    if(trat.max()>0.1):
        # if the thermal time scales are very small, we just set the local thermal energy density to equilibrium (irradiation heating = diffusive cooling)
        #    print("dudt = "+str(dudt))
        dudt = v*force +(irradheating - qloss) * exp(-trat) + (ueq-u) / (ueq+u) * irradheating * (1.-exp(-trat)) 
        #   print("dudt = "+str(dudt))
        #   ii=input("dudt")
        
    return rho*0., force, dudt, qloss, ueq

def toprim(m, s, e, g):
    '''
    convert conserved quantities to primitives
    '''
    rho=m/g.across
    v=s/m
#    wrel = where(fabs(v)>vmax)
#    if(size(wrel)>0):
#        v[wrel] =  quasirelfunction(v[wrel], vmax) # equal to vmax when v[wrel]=vmax, approaching 1 at large values 
    # v[wrel]*sqrt((1.+vmax**2)/(1.+v[wrel]**2))
    # if(m.min()<mfloor):
      #  print("toprim: m.min = "+str(m.min()))
      #  print("... at "+str(g.r[m.argmin()]))
        #        exit(1)
    #    v=s*0.
    #    v[rho>rhofloor]=(s/m)[rho>rhofloor]
    #    v=v/sqrt(1.+v**2)
    u=(e-m*(v**2/2.-1./g.r-0.5*(g.r*g.sth*omega)**2))/g.across
    umin = u.min()
    #    if(rho.min() < rhofloor):
    #        print("rhomin = "+str(rho.min()))
    #        print("mmin = "+str(m.min()))
        # exit(1)
        #    u[u<=ufloor]=0.
    beta = betafun(Fbeta(rho, u))
    press = 3.*(1.-beta/2.) * u
    return rho, v, u, u*(1.-beta)/(1.-beta/2.), beta, press

def derivo(m, s, e, l_half, s_half, p_half, fe_half, dm, ds, de, g, dlleft, dlright, dt, edot = None):
    '''
    main advance in a dt step
    input: three densities, l (midpoints), three fluxes (midpoints), three sources, timestep, r, sin(theta), cross-section
    output: three temporal derivatives later used for the time step
    includes boundary conditions for mass and energy!
    '''
    #    print("main_step: mmin = "+str(m.min()))
    nl=size(m)
    dmt=zeros(nl) ; dst=zeros(nl); det=zeros(nl)
    dmt[1:-1] = -(s_half[1:]-s_half[:-1])/(l_half[1:]-l_half[:-1]) + dm[1:-1]
    dst[1:-1] = -(p_half[1:]-p_half[:-1])/(l_half[1:]-l_half[:-1]) + ds[1:-1]
    det[1:-1] = -(fe_half[1:]-fe_half[:-1])/(l_half[1:]-l_half[:-1]) + de[1:-1]

    #left boundary conditions:
    dmt[0] = -(s_half[0]-(-mdotsink))/dlleft
    #    dst[0] = (-mdotsink-s[0])/dt # ensuring approach to -mdotsink
    dst[0] = 0. # 
    det[0] = -(fe_half[0]-(0.))/dlleft+de[0] # no energy sink anyway
    # right boundary conditions:
    dmt[-1] = -((-mdot)-s_half[-1])/dlright
    #    dst[-1] = (-mdot-s[-1])/dt # ensuring approach to -mdot
    dst[-1] = 0.
    #    edot =  abs(mdot) * 0.5/g.r[-1] + s[-1]/m[-1] * u[-1] # virial equilibrium
    if(edot is None):
        edot = 0.
    det[-1] = -((edot)-s_half[-1])/dlright + de[-1]
    return dmt, dst, det

def RKstep(m, s, e, g, ghalf, dl, dlleft, dlright, dt):
    '''
    calculating elementary increments of conserved quantities
    '''
    rho, v, u, urad, beta, press = toprim(m, s, e, g) # primitive from conserved
    u, rho = regularize(u, rho)
    fm, fs, fe = fluxes(rho, v, u, g)
    g1 = Gamma1(5./3., beta)
    cs=sqrt(g1*press/rho)
    # slightly under-estimating the SOS to get stable signal velocities; exact for u<< rho
    vl, vm, vr = sigvel_isentropic(v, cs, g1)        
    fm_half, fs_half, fe_half =  solv.HLLC([fm, fs, fe], [m, s, e], vl, vr, vm)
    #    dul_half = diffuse(rho, urad, dl, ghalf.across)
    # diffusion term introduces instabilities -- what shall we do?
    #    fe_half += dul_half
    dm, ds, de, flux, ueq = sources(rho, v, u, g,ltot=0., dt=dt)
    
    ltot=simps(flux, x=g.l) # no difference
    dmt, dst, det = derivo(m, s, e, ghalf.l, fm_half, fs_half, fe_half, dm, ds, de, g, dlleft, dlright, dt, edot = fe[-1])
                           #fe_half[-1])
    return dmt, dst, det, ltot, ueq
    
################################################################################
def alltire():
    '''
    the main routine bringing all together.
    '''
    timer.start("total")

    # if the outpur directory does not exist:
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # setting geometry:
    sthd=1./sqrt(1.+(dr_e/r_e)**2) # initial sin(theta)
    rmax=r_e*sthd # slightly less then r_e 
    r=((2.*(rmax-rstar)/rstar)**(arange(nx0)/double(nx0-1))+1.)*(rstar/2.) # very fine radial mesh
    g = geometry_initialize(r, r_e, dr_e, afac=afac) # radial-equidistant mesh
    g.l += r.min() # we are starting from a finite radius
    if(logmesh):
        luni=exp(linspace(log((g.l).min()), log((g.l).max()), nx, endpoint=False)) # log(l)-equidistant mesh
    else:
        luni=linspace((g.l).min(), (g.l).max(), nx, endpoint=False)
    g.l -= r.min() ; luni -= r.min()
    luni_half=(luni[1:]+luni[:-1])/2. # half-step l-equidistant mesh
    rfun=interp1d(g.l,g.r, kind='linear') # interpolation function mapping l to r
    rnew=rfun(luni) # radial coordinates for the  l-equidistant mesh
    g = geometry_initialize(rnew, r_e, dr_e, writeout=outdir+'/geo.dat', afac=afac) # all the geometric quantities for the l-equidistant mesh
    r=rnew # set a grid uniform in l=luni
    r_half=rfun(luni_half) # half-step radial coordinates
    ghalf = geometry_initialize(r_half, r_e, dr_e, afac=afac) # mid-step geometry in r
    ghalf.l += g.l[1]/2. # mid-step mesh starts halfstep later
    #    dlleft = ghalf.l[1]-ghalf.l[0] # 2.*(ghalf.l[1]-ghalf.l[0])-(ghalf.l[2]-ghalf.l[1])
    #    dlright = ghalf.l[-1]-ghalf.l[-2] # 2.*(ghalf.l[-1]-ghalf.l[-2])-(ghalf.l[-2]-ghalf.l[3])
    dl=g.l[1:]-g.l[:-1] # cell sizes
    dlleft = dl[0] ; dlright = dl[-1]
    #
    
    # testing bassun.py
    print("delta = "+str((g.across/(4.*pi*afac*g.r*g.sth))[0]))
    print("delta = "+str((g.sth*g.r/sqrt(1.+3.*g.cth**2))[0] * dr_e/r_e))
    print("delta = "+str(g.delta[0]))
    BSgamma = (g.across/g.delta**2)[0]/mdot*rstar
    BSeta = (8./21./sqrt(2.)*umag)**0.25*sqrt(g.delta[0])/(rstar)**0.125
    print("BS parameters:")
    print("   gamma = "+str(BSgamma))
    print("   eta = "+str(BSeta))
    x1 = 1. ; x2 = 1000. ; nxx=1000
    xtmp=(x2/x1)**(arange(nxx)/double(nxx))*x1
    if(ifplot):
        plots.someplots(xtmp, [bs.fxis(xtmp, BSgamma, BSeta, 3.)], name='fxis', ytitle=r'$F(x)$')
    xs = bs.xis(BSgamma, BSeta)
    print("   xi_s = "+str(xs))
    # input("BS")
    # magnetic field energy density:
    umagtar = umag * (1.+3.*g.cth**2)/4. * (rstar/g.r)**6
    # initial conditions:
    m=zeros(nx) ; s=zeros(nx) ; e=zeros(nx)
    vinit=vout *sqrt(r_e/g.r) # initial velocity

    # setting the initial distributions of the primitive variables:
    rho = abs(mdot) / (abs(vout)+abs(vinit)) / g.across
    vinit *= ((g.r-rstar)/(g.r+rstar))
    u =  3.*umagtar[-1] * (r_e/g.r) * (rho/rho[-1])
    # 3.*umagout+(rho/rho[-1])*0.01/g.r
    print("U = "+str(u.min())+" to "+str(u.max()))
    m, s, e = cons(rho, vinit, u, g)
    ulast = u[-1] # 
    #    print(u/rho)
    #    ii=input('m')
    
    rho1, v1, u1, urad, beta, press = toprim(m, s, e, g) # primitive from conserved
    print(str((rho-rho1).std())) 
    print(str((vinit-v1).std()))
    print(str((u-u1).std())) # accuracy 1e-14
    print("primitive-conserved")
    print("rhomin = "+str(rho.min())+" = "+str(rho1.min()))
    print("umin = "+str(u.min())+" = "+str(u1.min()))
    #    input("P")
    m0=m
    
    t=0.;  tstore=0.  ; nout=0

    # if we want to restart from a stored configuration
    # works so far correctly ONLY if the mesh is identical!
    if(ifrestart):
        if(ifhdf):
            # restarting from a HDF5 file
            entryname, t, l, r, sth, rho, u, v = hdf.read(restartfile, restartn)
            tstore = t
            print("restarted from file "+restartfile+", entry "+entryname)
        else:
            # restarting from an ascii output
            ascrestartname = restartprefix + hdf.entryname(restartn, ndig=5) + ".dat"
            lines = loadtxt(ascrestartname, comments="#")
            rho = lines[:,1] ; v = lines[:,2] ; u = lines[:,3] * umagtar
            # what about t??
            print("restarted from ascii output "+ascrestartname)
        r *= rstar
        m, s, e = cons(rho, v, u, g)
        nout = restartn
            
    dlmin=dl.min()
    dt = dlmin*0.25
    print("dt = "+str(dt))
    #    ti=input("dt")
    
    ltot=0. # estimated total luminosity
    if(ifrestart):
        fflux=open(outdir+'/'+'flux.dat', 'a')
        ftot=open(outdir+'/'+'totals.dat', 'a')
    else:
        fflux=open(outdir+'/'+'flux.dat', 'w')
        ftot=open(outdir+'/'+'totals.dat', 'w')

    if(ifhdf):
        hname = outdir+'/'+'tireout.hdf5'
        hfile = hdf.init(hname, g) # , m1, mdot, eta, afac, re, dre, omega)
    
    timer.start("total")
    while(t<tmax):
        timer.start_comp("advance")
        # Runge-Kutta, fourth order, one step:
        k1m, k1s, k1e, ltot1, ueq1 = RKstep(m, s, e, g, ghalf, dl, dlleft, dlright, dt)
        k2m, k2s, k2e, ltot2, ueq2 = RKstep(m+k1m*dt/2., s+k1s*dt/2., e+k1e*dt/2., g, ghalf, dl, dlleft, dlright, dt)
        k3m, k3s, k3e, ltot3, ueq3 = RKstep(m+k2m*dt/2., s+k2s*dt/2., e+k2e*dt/2., g, ghalf, dl, dlleft, dlright, dt)
        k4m, k4s, k4e, ltot4, ueq4 = RKstep(m+k3m*dt, s+k3s*dt, e+k3e*dt, g, ghalf, dl, dlleft, dlright, dt)
        m += (k1m+2.*k2m+2.*k3m+k4m) * dt/6.
        s += (k1s+2.*k2s+2.*k3s+k4s) * dt/6.
        e += (k1e+2.*k2e+2.*k3e+k4e) * dt/6.
        s[0] = -mdotsink ; s[-1] = -mdot
        if(ufixed):
            # imposes a constant-thermal-energy outer BC
            # sort of redundant because it converts the whole variable set instead of the single last point; need to optimize it!
            rhotmp, vtmp, utmp, uradtmp, betatmp, presstmp = toprim(m, s, e, g) 
            mtmp, stmp, etmp = cons(rhotmp, vtmp, ulast, g)
            e[-1] = etmp[-1]
        ltot = (ltot1 + 2.*ltot2 + 2.*ltot3 + ltot4) / 6.
        ueq = (ueq1 + 2.*ueq2 + 2.*ueq3 + ueq4) / 6.
        t += dt
        csqest = 4./3.*u/rho
        rho, v, u, urad, beta, press = toprim(m, s, e, g) # primitive from conserved         
        dt = 0.5 * dlmin / sqrt(1.+2.*csqest.max()+2.*(v**2).max())
        timer.stop_comp("advance")
        timer.lap("step")
        if(t>=tstore):
            tstore += dtout
            timer.start("io")
            #            rho, v, u, urad, beta, press = toprim(m, s, e, g) # primitive from conserved            tstore+=dtout
            print("t = "+str(t*tscale)+"s")
            print("dt = "+str(dt*tscale)+"s")
            #            print("tratmax = "+str(tratmax))
            fflux.write(str(t*tscale)+' '+str(ltot)+'\n')
            fflux.flush()
            if ifplot & (nout%plotalias == 0):
                print("plotting")
                plots.uplot(g.r, u, rho, g.sth, v, name=outdir+'/utie{:05d}'.format(nout), umagtar = umagtar, ueq=ueq)
                plots.vplot(g.r, v, sqrt(4./3.*u/rho), name=outdir+'/vtie{:05d}'.format(nout))
                plots.someplots(g.r, [u/rho**(4./3.)], name=outdir+'/entropy{:05d}'.format(nout), ytitle=r'$S$', ylog=True)
                plots.someplots(g.r, [(u-urad)/(u-urad/2.), 1.-(u-urad)/(u-urad/2.)],
                                name=outdir+'/beta{:05d}'.format(nout), ytitle=r'$\beta$, $1-\beta$', ylog=True)
            mtot=simps(m[1:-1], x=g.l[1:-1])
            etot=simps(e[1:-1], x=g.l[1:-1])
            print("mass = "+str(mtot))
            print("ltot = "+str(ltot))
            #            print("heat = "+str(heat))
            print("energy = "+str(etot))
            print("momentum = "+str(trapz(s[1:-1], x=g.l[1:-1])))
            
            ftot.write(str(t*tscale)+' '+str(mtot)+' '+str(etot)+'\n')
            ftot.flush()
            if(ifhdf):
                hdf.dump(hfile, nout, t, rho, v, u)
            if not(ifhdf) or (nout%ascalias == 0):
                # ascii output:
                print(nout)
                fname=outdir+'/tireout{:05d}'.format(nout)+'.dat'
                fstream=open(fname, 'w')
                fstream.write('# t = '+str(t*tscale)+'s\n')
                fstream.write('# format: r/rstar -- rho -- v -- u/umag\n')
                for k in arange(nx):
                    fstream.write(str(g.r[k]/rstar)+' '+str(rho[k])+' '+str(v[k])+' '+str(u[k]/umagtar[k])+'\n')
                fstream.close()
            timer.stop("io")
            if(nout%ascalias == 0):
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
# if you want the main procedure to start running immediately after the compilation, uncomment the following:
# alltire()

