from scipy.integrate import *
from scipy.interpolate import *

import numpy.random
from numpy.random import rand
from numpy import *
# import time
import os
import re
import linecache
import os.path
import imp
import sys

import multiprocessing
from multiprocessing import Pool, Pipe, Process

if(size(sys.argv)>1):
    print("launched with arguments "+str(', '.join(sys.argv)))
    # new conf file
    conf=sys.argv[1]
    print(conf+" configuration set by the arguments")
else:
    conf='DEFAULT'

# configuration file:
import configparser as cp
conffile = 'globals.conf'
config = cp.ConfigParser(inline_comment_prefixes="#")
config.read(conffile) 
ifplot = config[conf].getboolean('ifplot')
ifhdf = config[conf].getboolean('ifhdf')
verbose = config[conf].getboolean('verbose')
autostart = config[conf].getboolean('autostart')
#

# loading local modules:
if ifplot:
    import plots
if ifhdf:
    import hdfoutput as hdf
import bassun as bs
import solvers as solv
from sigvel import *
from geometry import *

from timer import Timer
timer = Timer(["total", "step", "io"],
              ["BC", "dt", "RKstep", "updateCon"])

def time_step(prim, g, dl):
    # time step adjustment:
    csqest = 4./3.*prim['press']/prim['rho']
    dt_CFL = CFL * (dl / (sqrt(csqest)+abs(prim['v']))[1:-1]).min()
    qloss = 2.*prim['urad']/prim['rho'] / xirad * (g.across/g.delta + 2.*g.delta)**2/g.across
    # approximate qloss

    wpos = ((qloss) > 0.) & (prim['rho'] > 0.)
    wpos = wpos[1:-1]
    dt_thermal = Cth * abs((prim['u']*g.across)/qloss).min()
    
    if(raddiff):
        ctmp = dl**2 * 3.*prim['rho'][1:-1]
        dt_diff = Cdiff * quantile(ctmp[ctmp>0.], 0.01) # (dx^2/D)
    else:
        dt_diff = dt_CFL * 50. # effectively infinity ;)
    
    return minimum(dt_CFL, minimum(dt_diff, dt_thermal))# 1./(1./dt_CFL + 1./dt_thermal + 1./dt_diff)

def timestepdetails(g, rho, press, u, v, urad):
    dl = g.l[1:]-g.l[:-1]
    rho_half = (rho[1:]+rho[:-1])/2. ; press_half = (press[1:]+press[:-1])/2. ; u_half = (u[1:]+u[:-1])/2. ; v_half = (v[1:]+v[:-1])/2. ; urad_half = (urad[1:]+urad[:-1])/2.
    csqest = 4./3.*press_half/rho_half
    dt_CFL = CFL * (dl / (sqrt(csqest)+abs(v_half))).min()
    qloss = 2.*urad/rho / xirad * (g.across/g.delta + 2.*g.delta)**2/g.across
    wpos = ((qloss) > 0.) & (rho > 0.)
    dt_thermal = Cth * abs((u*g.across)[wpos]/qloss[wpos]).min()
    if(raddiff):
        ctmp = dl**2 * 3.*rho_half
        dt_diff = Cdiff * quantile(ctmp[ctmp>0.], 0.01) # (dx^2/D)
    else:
        dt_diff = dt_CFL * 5. # effectively infinity ;)
    return minimum(dt_CFL, minimum(dt_diff, dt_thermal)), dt_CFL, dt_thermal, dt_diff
    
def timestepmin(prim, g, dl, dtpipe):
    nd = prim['N']
    if nd > 0:
        dtlocal = time_step(prim, g, dl)
        dtpipe.send(dtlocal)
    else:
        #        dtlocal = time_step(prim, g, dl)
        dt = time_step(prim, g, dl)
        for k in range(parallelfactor -1):
            dtk = dtpipe[k].recv()
            dt = minimum(dt, dtk)
    if nd == 0:
        for k in range(parallelfactor -1):
            dtpipe[k].send(dt)
    else:
        dt = dtpipe.recv()             
    return dt

#
def regularize(u, rho, press):
    '''
    if internal energy goes below ufloor, we heat the matter up artificially
    '''
    #    u1=u-ufloor ; rho1=rho-rhofloor ; press1 = press-ufloor
    return (u+ufloor+fabs(u-ufloor))/2., (rho+rhofloor+fabs(rho-rhofloor))/2., (press+ufloor +fabs(press-ufloor))/2.

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
    tt = copy(tau)
    if(size(wtrans)>0):
        tt[wtrans] = (tau[wtrans]+abs(tau[wtrans]))/2.
    if(size(wopaq)>0):
        tt[wopaq] = 1.
    if(size(wmed)>0):
        tt[wmed] = 1. - exp(-tau[wmed])
    return tt

def tratfac(x):
    '''
    an accurate smooth version of (1-e^{-x})/x
    '''
    xmin = taumin ; xmax = taumax # limits the same as for optical depth
    nx = size(x)
    tt = copy(x)
    if nx>1:
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
    else:
        if x <= xmin:
            return 1.
        else:
            if x>=xmax:
                return 1./x
            else:
                return (1.-exp(x))/x
            

# define once and globally
from beta import *
betafun = betafun_define() # defines the interpolated function for beta (\rho, U)
betafun_p = betafun_press_define() # defines the interpolated function for beta (\rho, P)

##############################################################################

# conversion between conserved and primitive variables for separate arrays and for a single domain

def toprim_separate(m, s, e, g):
    rho=m/g.across
    v=s/m
    u=(e-m*(v**2/2.-1./g.r-0.5*(g.r*g.sth*omega)**2))/g.across
    #    umin = u.min()
    beta = betafun(Fbeta(rho, u, betacoeff))
    press = u/3./(1.-beta/2.)
    u, rho, press = regularize(u, rho, press)
    return rho, v, u, u*(1.-beta)/(1.-beta/2.), beta, press

def tocon_separate(rho, v, u, g):
    
    m=rho*g.across # mass per unit length
    s=m*v # momentum per unit length
    e=(u+rho*(v**2/2.- 1./g.r - 0.5*(omega*g.r*g.sth)**2))*g.across  # total energy (thermal + mechanic) per unit length
    return m, s, e

# conversion between conserved and primitive variables using dictionaries and multiple domains

def tocon(prim, gnd = None):
    '''
    computes conserved quantities from primitives
    '''
    #    m = con['m'] ; s = con['s'] ; e = con['e'] ; nd = con['N']
    #    rho = prim['rho'] ; v = prim['v'] ; u = prim['u'] ; nd = prim['N']
    if gnd is None:
        gnd = l_g[prim['N']]
    
    m=prim['rho']*gnd.across # mass per unit length
    s=m*prim['v'] # momentum per unit length
    e=(prim['u']+prim['rho']*(prim['v']**2/2.- 1./gnd.r - 0.5*(omega*gnd.r*gnd.sth)**2))*gnd.across  # total energy (thermal + mechanic) per unit length
    return {'m': m, 's': s, 'e': e, 'N': prim['N']}

def toprim(con, gnd = None):
    '''
    convert conserved quantities to primitives
    '''
    #  m = con['m'] ; s = con['s'] ; e = con['e'] ; nd = con['N']
    if gnd is None:
        gnd = l_g[con['N']]
    rho = con['m']/gnd.across
    v = con['s']/con['m']
    u = (con['e']-con['m']*(v**2/2.-1./gnd.r-0.5*(gnd.r*gnd.sth*omega)**2))/gnd.across
    #    umin = u.min()
    beta = betafun(Fbeta(rho, u, betacoeff))
    press = u/3./(1.-beta/2.)
    u, rho, press = regularize(u, rho, press)
    urad = u*(1.-beta)/(1.-beta/2.)
    prim = {'rho': rho, 'v': v, 'u': u, 'beta': beta, 'urad': urad, 'press': press, 'N': con['N']}
    return prim

def diffuse(rho, urad, v, dl, across):
    '''
    radial energy diffusion;
    calculates energy flux contribution already at the cell boundary
    across should be set at half-steps
    '''
    #    rho_half = (rho[1:]+rho[:-1])/2. # ; v_half = (v[1:]+v[:-1])/2.  ; u_half = (u[1:]+u[:-1])/2.
    # rtau_right = rho[1:] * dl / 2.# optical depths along the field line, to the right of the cell boundaries
    # rtau_left = rho[:-1] * dl / 2. # -- " -- to the left -- " --
    # rtau = dl * (rho[1:]+rho[:-1])/2.
    # rtau_left + rtau_right
    rtau_exp = tratfac(dl * (rho[1:]+rho[:-1])/2.)
    # across_half = (across[1:]+across[:-1])/2.
    
    duls_half =  nubulk  * (( urad * v)[1:] - ( urad * v)[:-1])\
                 *(across[1:]+across[:-1]) / 6. * rtau_exp #  / (rtau_left + rtau_right)
    # -- photon bulk viscosity
    dule_half = ((urad)[1:] - (urad)[:-1])\
                *(across[1:]+across[:-1]) / 6. * rtau_exp # / (rtau_left + rtau_right)
    dule_half +=  duls_half * (v[1:]+v[:-1])/2. # adding the viscous energy flux
    # -- radial diffusion

    return -duls_half, -dule_half 
            
def fluxes(g, rho, v, u, press):
    '''
    computes the fluxes of conserved quantities, given primitives; 
    radiation diffusion flux is not included, as it is calculated at halfpoints
    inputs:
    rho -- density, v -- velocity, u -- thermal energy density
    g is geometry (structure)
    Note: fluxes do not include diffusion (added separately)
    '''
    #    nd = prim['N']  # ; rho = prim['rho'] ; v = prim['v'] ; u = prim['u'] ; press = prim['press'] ; beta = prim['beta']
    #    gnd = l_g[nd]
    # across = g.across ; r = g.r  ; sth = g.sth 
    s = rho * v * g.across # mass flux (identical to momentum per unit length -- can we use it?)
    p = g.across * (rho*v**2 + press) # momentum flux
    fe = g.across * v * (u + press + (v**2/2.-1./g.r-0.5*(omega*g.r*g.sth)**2)*rho) # energy flux without diffusion
    return s, p, fe

def sources(g, rho, v, u, urad, ltot = 0., forcecheck = False, dmsqueeze = 0., desqueeze = 0.):
        # prim, ltot=0., dmsqueeze = 0., desqueeze = 0., forcecheck = False):
    '''
    computes the RHSs of conservation equations
    no changes in mass
    momentum injection through gravitational and centrifugal forces
    energy losses through the surface
    outputs: dm, ds, de, and separately the amount of energy radiated per unit length per unit time ("flux")
    additional output:  equilibrium energy density
    if the "forcecheck" flag is on, outputs the grav.potential difference between the outer and inner boundaries and compares to the work of the force along the field line
    '''
    #    nd = prim['N'] ; rho = prim['rho'] ; v = prim['v'] ; u = prim['u']
    # gnd = l_g[nd]
    #  sinsum=sina*cth+cosa*sth # cos( pi/2-theta + alpha) = sin(theta-alpha)
    #     tau = rho*g.across/(4.*pi*g.r*g.sth*afac)
    delta = g.delta[1:-1] ;  across = g.across[1:-1] ; r = g.r[1:-1] ; cth = g.cth[1:-1] ; sth = g.sth[1:-1] ; cosa = g.cosa[1:-1] ; sina = g.sina[1:-1]
    #    tau = rho * delta # tau in transverse direction
    #    tauphi = rho * across / delta / 2. # optical depth in azimuthal direction
    taueff = rho / (1./delta + 2. * delta /  across)
    # copy(1./(1./tau + 1./tauphi))
    #    taufac = taufun(taueff)    # 1.-exp(-tau)
    #    taufac = 1. 
    #    gamefac = tratfac(tau)
    #    gamedd = eta * ltot * gamefac
    sinsum = (sina*cth+cosa*sth) # sin(theta+alpha)
    force = ((-sinsum/r**2*(1.-eta * ltot * tratfac(rho*delta))
              +omega**2*r*sth*cosa)*rho*across) # *taufac
    if(forcecheck):
        network = simps(force/(rho*across), x=g.l)
        return network, (1./r[0]-1./r[-1])
    #    beta = prim['beta'] # betafun(Fbeta(rho, u, betacoeff))
    #    urad = prim['urad']
    #     urad = copy(u * (1.-beta)/(1.-beta/2.))
    #    urad = (urad+abs(urad))/2.
    qloss = 2.*urad/(xirad*taueff+1.)*(across/delta+2.*delta)* taufun(taueff)  # diffusion approximation; energy lost from 4 sides
    # irradheating = heatingeff * eta * mdot *afac / r * sth * sinsum * taufun(taueff) !!! need to include irradheating later!
    #    ueq = heatingeff * mdot / g.r**2 * sinsum * urad/(xirad*tau+1.)
    dm = rho*0.-dmsqueeze # copy(rho*0.-dmsqueeze)
    dudt = v*force-qloss # +irradheating # copy
    ds = force - dmsqueeze * v # lost mass carries away momentum
    de = dudt - desqueeze # lost matter carries away energy (or enthalpy??)
    #    return dm, force, dudt, qloss, ueq
    return dm, ds, de, qloss

def qloss_separate(rho, v, u, g):
    '''
    standalone estimate for flux distribution
    '''
    tau = rho * g.delta
    tauphi = rho * g.across / g.delta / 2. # optical depth in azimuthal direction
    taueff = copy(1./(1./tau + 1./tauphi))
    taufac = taufun(taueff)    # 1.-exp(-tau)
    beta = betafun(Fbeta(rho, u, betacoeff))
    urad = copy(u * (1.-beta)/(1.-beta/2.))
    urad = (urad+abs(urad))/2.    
    qloss = copy(2.*urad/(xirad*taueff+1.)*(g.across/g.delta+2.*g.delta)*taufac)  # diffusion approximation; energy lost from 4 sides
    return qloss

def derivo(l_half, m, s, e, s_half, p_half, fe_half, dm, ds, de):
           #, dlleft, dlright,
           #sleft, sright, pleft, pright, feleft, feright):
    '''
    main advance step
    input: three densities, l (midpoints), three fluxes (midpoints), three sources, timestep, r, sin(theta), cross-section
    output: three temporal derivatives later used for the time step
    includes boundary conditions for mass and energy!
    '''
    nl=size(m)
    dmt=zeros(nl) ; dst=zeros(nl); det=zeros(nl)
    dmt = -(s_half[1:]-s_half[:-1])/(l_half[1:]-l_half[:-1]) + dm
    dst = -(p_half[1:]-p_half[:-1])/(l_half[1:]-l_half[:-1]) + ds
    det = -(fe_half[1:]-fe_half[:-1])/(l_half[1:]-l_half[:-1]) + de

    return dmt, dst, det

def RKstep(gnd, lhalf, prim, leftpack, rightpack, umagtar = None, ltot = 0.):
    # BCleft, BCright, 
    # m, s, e, g, ghalf, dl, dlleft, dlright, ltot=0., umagtar = None, momentum_inflow = None, energy_inflow =  None):
    '''
    calculating elementary increments of conserved quantities
    '''
    #    prim = toprim(con) # primitive from conserved
    rho = prim['rho'] ;  press = prim['press'] ;  v = prim['v'] ; urad = prim['urad'] ; u = prim['u']
    beta = prim['beta']
    #    beta = betafun(Fbeta(rho, u, betacoeff))
    nd = prim['N']
    #
    # adding ghost zones:
    if leftpack is not None:
        rholeft, vleft, uleft = leftpack
        betaleft = betafun(Fbeta(rholeft, uleft, betacoeff))
        uradleft = uleft * (1.-betaleft)/(1.-betaleft/2.)
        pressleft = uleft / (1.-betaleft/2.)/3.
        # gnd = geometry_add(gleft, gnd)
        rho = concatenate([[rholeft], rho])
        v = concatenate([[vleft], v])
        u = concatenate([[uleft], u])
        urad = concatenate([[uradleft], urad])
        press = concatenate([[pressleft], press])
        beta = concatenate([[betaleft], beta])
    else:
        if nd>0:
            print("topology issue, nd = "+str(nd))
        rho = concatenate([[rho[0]], rho])
        v = concatenate([[v[0]], v])
        u = concatenate([[u[0]], u])
        urad = concatenate([[urad[0]], urad])
        press = concatenate([[press[0]], press])
        beta = concatenate([[beta[0]], beta])        
    if rightpack is not None:
        rhoright, vright, uright = rightpack
        betaright = betafun(Fbeta(rhoright, uright, betacoeff))
        pressright = uright / (1.-betaright/2.)/3.
        uradright = uright * (1.-betaright)/(1.-betaright/2.)
        # gnd = geometry_add(gnd, gright)
        rho = concatenate([rho, [rhoright]])
        v = concatenate([v, [vright]])
        u = concatenate([u, [uright]])
        urad = concatenate([urad, [uradright]])
        press = concatenate([press, [pressright]])
        beta = concatenate([beta, [betaright]])
    else:
        if nd < (parallelfactor-1):
            print("topology issue, nd = "+str(nd))
        rho = concatenate([rho, [rho[-1]]])
        v = concatenate([v, [v[-1]]])
        u = concatenate([u, [u[-1]]])
        urad = concatenate([urad, [urad[-1]]])
        press = concatenate([press, [press[-1]]])
        beta = concatenate([beta, [beta[-1]]])        
    fm, fs, fe = fluxes(gnd, rho, v, u, press)
    if leftpack is None:
        fm[0] = 0.
        #        fs[0] = fs[1]
        fe[0] = 0.
        # gnd = geometry_add(geometry_local(gnd,0), gnd)       
    if rightpack is None:
        fm[-1] = -mdot
        #        fs[-1] = fs[-2]
        #        fe[-1] = fe[-2]
        # gnd = geometry_add(gnd, geometry_local(gnd,0))
    g1 = Gamma1(5./3., beta)
    '''
    csq=g1*press/rho
    if(csq.min()<csqmin):
        wneg = (csq<=csqmin)
        csq[wneg] = csqmin
    #    cs = sqrt(csq)
    '''
    vl, vm, vr =sigvel_mean(v, sqrt(g1*press/rho))
    # sigvel_linearized(v, cs, g1, rho, press)
    # sigvel_isentropic(v, cs, g1, csqmin=csqmin)
    if any(vl>=vm) or any(vm>=vr):
        wwrong = (vl >=vm) | (vm<=vr)
        print("rho = "+str((rho[1:])[wwrong]))
        print("press = "+str((press[1:])[wwrong]))
        print(vl[wwrong])
        print(vm[wwrong])
        print(vr[wwrong])
        print(vm[wwrong])
        print("R = "+str((gnd.r[1:])[wwrong]))
        print("signal velocities crashed")
        ii=input("cs")
    m, s, e = tocon_separate(rho, v, u, gnd) # extended conserved quantities
    #    print("size(m) = "+str(size(m)))
    # print("size(s) = "+str(size(s)))
    #    print("size(vl) = "+str(size(vl)))
    # print("size(e) = "+str(size(e)))
    #  print("size(fe) = "+str(size(fe))+" ( nd = "+str(nd)+" leftpack = "+str(leftpack)+") ")
    fm_half, fs_half, fe_half =  solv.HLLC([fm, fs, fe], [m, s, e], vl, vr, vm)
    if(raddiff):
        #        dl = gnd.l[1:]-gnd.l[:-1]
        #  across = gnd.across
        '''
        print("flux size = "+str(size(fe_half)))
        print("rho size = "+str(size(rho)))
        print("v size = "+str(size(v)))
        print("urad size = "+str(size(urad)))
        print("dl size = "+str(size(dl)))
        print("across size = "+str(size(across)))
        '''
        duls_half, dule_half = diffuse(rho, urad, v, gnd.l[1:]-gnd.l[:-1], gnd.across)
        # radial diffusion suppressed, if transverse optical depth is small:
        delta = (gnd.delta[1:]+gnd.delta[:-1])/2.
        duls_half *= 1.-exp(-delta * (rho[1:]+rho[:-1])/2.)
        dule_half *= 1.-exp(-delta * (rho[1:]+rho[:-1])/2.)
        fs_half += duls_half ; fe_half += dule_half         
    #  sinks and sources:
    if(squeezemode):
        if umagtar is None:
            umagtar = umag * ((1.+3.*gnd.cth**2)/4. * (rstar/gnd.r)**6)[1:-1]
        dmsqueeze = 2. * m[1:-1] * sqrt(g1[1:-1]*maximum((press[1:-1]-umagtar)/rho[1:-1], 0.))/gnd.delta[1:-1]
        if squeezeothersides:
            dmsqueeze += 4. * m[1:-1] * sqrt(g1[1:-1]*maximum((press[1:-1]-umagtar)/rho[1:-1], 0.))/ (gnd.across[1:-1] / gnd.delta[1:-1])
        desqueeze = dmsqueeze * (e[1:-1]+press[1:-1]* gnd.across[1:-1]) / m[1:-1] # (e-u*g.across)/m
    else:
        dmsqueeze = 0.
        desqueeze = 0.
    dm, ds, de, flux = sources(gnd, rho[1:-1], v[1:-1], u[1:-1], urad[1:-1], ltot=ltot, dmsqueeze = dmsqueeze, desqueeze = desqueeze)
    #    ltot=trapz(flux, x=gnd.l[1:-1]) 

    dmt, dst, det = derivo(lhalf, m, s, e, fm_half, fs_half, fe_half, dm, ds, de)

    # con1 = {'N': nd, 'm': dmt, 's': dst, 'e': det} # , 'ltot': ltot}

    return {'N': nd, 'm': dmt, 's': dst, 'e': det}

def updateCon(l, dl, dt):
    '''
    updates the conserved variables vector l1, adding dl*dt to l
    '''
    ndl = size(dl)
    l1 = l.copy()
    if ndl <= 1:
        l1['m'] = l['m']+dl['m']*dt
        l1['s'] = l['s']+dl['s']*dt
        l1['e'] = l['e']+dl['e']*dt
    else:
        if size(dt) < ndl:
            dt = [dt] * ndl
        for k in range(ndl):
            l1['m'] += dl[k]['m']*dt[k]
            l1['s'] += dl[k]['s']*dt[k]
            l1['e'] += dl[k]['e']*dt[k]            
    return l1

################################################################################

def BCsend(leftpipe, rightpipe, leftpack_send, rightpack_send):
    if leftpipe is not None:
        leftpipe.send(leftpack_send)
    if rightpipe is not None:
        rightpipe.send(rightpack_send)
        
    if leftpipe is not None:
        leftpack = leftpipe.recv()
    else:
        leftpack = None
    if rightpipe is not None:
        rightpack = rightpipe.recv()
    else:
        rightpack = None
    return leftpack, rightpack

def onedomain(g, lcon, ghostleft, ghostright, dtpipe, outpipe, hfile,
              t = 0., nout = 0):
    '''
    single domain, calculated by a single core
    '''
    con = lcon.copy()
    con1 = lcon.copy()
    nd = con['N']
    
    prim = toprim(con, gnd = g) # primitive from conserved

    #    t = 0.
    print("nd = "+str(nd))
    print("tmax = "+str(tmax))

    gleftbound = geometry_local(g, 0)
    grightbound = geometry_local(g, -1)
    # topology test:
    if ghostleft is not None:
        ghostleft.send([nd, nd-1])
    if ghostright is not None:
        ghostright.send([nd, nd+1])
    if ghostleft is not None:
        ndleft, ndcheckleft = ghostleft.recv()
        print("received from "+str(ndleft)+" (left), I should be "+str(ndcheckleft)+", and I am "+str(nd))
    if ghostright is not None:
        ndright, ndcheckright = ghostright.recv()
        print("received from "+str(ndright)+" (right), I should be "+str(ndcheckright)+", and I am "+str(nd))
    print("this was topology test\n")
    # exchange geometry!
    if ghostleft is not None:
        ghostleft.send(gleftbound)
    if ghostright is not None:
        ghostright.send(grightbound)
    if ghostleft is not None:
        gleftbound = ghostleft.recv()
    if ghostright is not None:
        grightbound = ghostright.recv()
    # if there is no exchange, the leftmost geometry just reproduces the leftmost point of the actual mesh
    print("nd = "+str(nd)+": gright = "+str(grightbound.l))
    print("nd = "+str(nd)+": gleft = "+str(gleftbound.l))
    #    print("g size = "+str(shape(g.l)))
    gext = geometry_add(g, grightbound)
    gext = geometry_add(gleftbound, gext)
    dlleft_nd = dlleft[nd] ; dlright_nd = dlright[nd]
    lhalf = (gext.l[1:]+gext.l[:-1])/2.
    dl = (l_ghalf[nd].l[1:]-l_ghalf[nd].l[:-1])

    #    umagtar = umag * ((1.+3.*gext.cth**2)/4. * (rstar/gext.r)**6)[1:-1]
    
    #    print("nd = "+str(nd)+": "+str(lhalf))
    #    ii = input('lhfl')
    tstore = t # ; nout = 0
    timectr = 0
    if nd == 0:
        timer.start("total")
        print("dtout = "+str(dtout))

    while(t<tmax):
        timer.start_comp("BC")
        leftpack_send = [prim['rho'][0], prim['v'][0], prim['u'][0]] # , prim['beta'][0]]
        rightpack_send = [prim['rho'][-1], prim['v'][-1], prim['u'][-1]] #, prim['beta'][-1]]
        leftpack, rightpack = BCsend(ghostleft, ghostright, leftpack_send, rightpack_send)
        timer.stop_comp("BC")

        # time step: all the domains send dt to first, and then the first sends the minimum value back
        timer.start_comp("dt")
        if timectr == 0:
            dt = timestepmin(prim, g, dl, dtpipe)
        #  dt = time_step(prim, g, dl)
        #        print("timectr = "+str(timectr))
        # print("dt = "+str(dt))
        timectr += 1
        if timectr >= timeskip:
            timectr = 0
        timer.stop_comp("dt")
        timer.start_comp("RKstep")
        dcon1 = RKstep(gext, lhalf, prim, leftpack, rightpack, umagtar = con['umagtar'])
        con1 = updateCon(con, dcon1, dt/2.)
     
        if ghostright is None:
            con1['s'][-1] = -mdot
            con1['e'][-1] = 0.
        if ghostleft is None:
            con1['s'][0] = 0.
     
        timer.stop_comp("RKstep")
        timer.start_comp("BC")
        prim = toprim(con1, gnd = g)
        leftpack_send = [prim['rho'][0], prim['v'][0], prim['u'][0]] # , prim['beta'][0]]
        rightpack_send = [prim['rho'][-1], prim['v'][-1], prim['u'][-1]] # , prim['beta'][-1]]
        leftpack, rightpack = BCsend(ghostleft, ghostright, leftpack_send, rightpack_send)
        timer.stop_comp("BC")
        timer.start_comp("RKstep")
        dcon2 = RKstep(gext, lhalf, prim, leftpack, rightpack, umagtar = con['umagtar']) # , BCfluxleft, BCfluxright)
        con2 = updateCon(con, dcon2, dt/2.)
     
        if ghostright is None:
            con2['s'][-1] = -mdot
            con2['e'][-1] = 0.
        if ghostleft is None:
            con2['s'][0] = 0.
     
        timer.stop_comp("RKstep")
        timer.start_comp("BC")
        prim = toprim(con2, gnd = g)
        leftpack_send = [prim['rho'][0], prim['v'][0], prim['u'][0]] # , prim['beta'][0]]
        rightpack_send = [prim['rho'][-1], prim['v'][-1], prim['u'][-1]] # , prim['beta'][-1]]
        leftpack, rightpack = BCsend(ghostleft, ghostright, leftpack_send, rightpack_send)
        timer.stop_comp("BC")
        timer.start_comp("RKstep")
        dcon3 = RKstep(gext, lhalf, prim, leftpack, rightpack, umagtar = con['umagtar']) #, BCfluxleft, BCfluxright)

        con3 = updateCon(con, dcon3, dt)
        
        if ghostright is None:
            con3['s'][-1] = -mdot
            con3['e'][-1] = 0.
        if ghostleft is None:
            con3['s'][0] = 0.
     
        timer.stop_comp("RKstep")
        timer.start_comp("BC")
        prim = toprim(con3, gnd = g)
        leftpack_send = [prim['rho'][0], prim['v'][0], prim['u'][0]] # , prim['beta'][0]]
        rightpack_send = [prim['rho'][-1], prim['v'][-1], prim['u'][-1]] # , prim['beta'][-1]]
        leftpack, rightpack = BCsend(ghostleft, ghostright, leftpack_send, rightpack_send)
        timer.stop_comp("BC")
        timer.start_comp("RKstep")
        dcon4 = RKstep(gext, lhalf, prim, leftpack, rightpack, umagtar = con['umagtar']) # , BCfluxleft, BCfluxright)
        timer.stop_comp("RKstep")
        timer.start_comp("updateCon")

        con = updateCon(con, [dcon1, dcon2, dcon3, dcon4], [dt/6., dt/3., dt/3., dt/6.])
        
        if ghostright is None:
            con['s'][-1] = -mdot
            con['e'][-1] = 0.
        if ghostleft is None:
            con['s'][0] = 0.
            #        prim = toprim(con, gnd = g)
        timer.stop_comp("updateCon")
 
        t += dt
        #        print("nd = "+str(nd)+"; t = "+str(t)+"; dt = "+str(dt))
        prim = toprim(con, gnd = g) # primitive from conserved
        if nd == 0:
            timer.lap("step")

        if (t>=tstore):            
            #            ltot = (dcon1['ltot'] + 2.*dcon2['ltot'] + 2.*dcon3['ltot'] + dcon4['ltot'])/6.
            tstore += dtout
            # sending data:
            timer.stop("step")
            timer.start("io")
            outpipe.send([t, g, con, prim])
            timer.stop("io")
            if (nd == 0) & (nout%ascalias == 0):
                timer.stats("step")
                timer.stats("io")
                timer.comp_stats()
                timer.start("step") #refresh lap counter (avoids IO profiling)
                timer.purge_comps()
            nout += 1
    #    dtpipe.close()
    
##########################################################
def tireouts(outpipes, hfile, fflux, ftot, nout = 0, t=0.):

    while t<tmax:
        print("tireouts: t = "+str(t))
        for k in range(size(outpipes)):
            t, g, con, prim = outpipes[k].recv()
            if k == 0:
                m = con['m'] ; e = con['e'] ; umagtar = con['umagtar']
                rho = prim['rho'] ; v = prim['v'] ; u = prim['u'] ; urad = prim['urad'] ; beta = prim['beta'] ; press = prim['press']
                r = g.r
            else:
                r = concatenate([r, g.r])
                m = concatenate([m, con['m']])
                e = concatenate([e, con['e']])
                rho = concatenate([rho, prim['rho']])
                press = concatenate([press, prim['press']])
                v = concatenate([v, prim['v']])
                u = concatenate([u, prim['u']])
                urad = concatenate([urad, prim['urad']])
                beta = concatenate([beta, prim['beta']])
                umagtar = concatenate([umagtar, con['umagtar']])
        wsort = argsort(r) # restoring the proper order of inputs
        r = r[wsort] ;  m = m[wsort] ;  e = e[wsort]
        rho = rho[wsort] ; v = v[wsort] ; u = u[wsort] ; urad = urad[wsort] ; beta = beta[wsort] ; umagtar = umagtar[wsort]
        qloss = qloss_separate(rho, v, u, gglobal)
        ltot = trapz(qloss, x = gglobal.l)
        mtot = trapz(m, x = gglobal.l)
        etot = trapz(e, x = gglobal.l)
        fflux.write(str(t*tscale)+' '+str(ltot)+'\n')
        ftot.write(str(t*tscale)+' '+str(mtot)+' '+str(etot)+'\n')
        print(str(t/tmax)+" calculated\n")
        print(str(t*tscale)+' '+str(ltot)+'\n')
        print(str(t*tscale)+' '+str(mtot)+'\n')
        # print("dt = "+str(dt)+'\n')
        dt, dt_CFL, dt_thermal, dt_diff = timestepdetails(gglobal, rho, press, u, v, urad)
        print("dt = "+str(dt)+" = "+str(dt_CFL)+"; "+str(dt_thermal)+"; "+str(dt_diff)+"\n")
        fflux.flush() ; ftot.flush()
        if hfile is not None:
            hdf.dump(hfile, nout, t, rho, v, u, qloss)
        if not(ifhdf) or (nout%ascalias == 0):
            if ifplot:
                plots.vplot(gglobal.r, v, sqrt(4./3.*u/rho), name=outdir+'/vtie{:05d}'.format(nout))
                plots.uplot(gglobal.r, u, rho, gglobal.sth, v, name=outdir+'/utie{:05d}'.format(nout), umagtar = umagtar)
                plots.someplots(gglobal.r, [beta, 1.-beta], formatsequence=['r-', 'b-'],
                                name=outdir+'/beta{:05d}'.format(nout), ytitle=r'$\beta$, $1-\beta$', ylog=True)
                plots.someplots(gglobal.r, [qloss*gglobal.r],
                                name=outdir+'/qloss{:05d}'.format(nout),
                                ytitle=r'$\frac{d^2 E}{d\ln l dt}$', ylog=False,
                                formatsequence = ['k-', 'r-'])
            # ascii output:
            # print(nout)
            fname=outdir+'/tireout{:05d}'.format(nout)+'.dat'
            if verbose:
                print(" ASCII output to "+fname)
            fstream=open(fname, 'w')
            fstream.write('# t = '+str(t*tscale)+'s\n')
            fstream.write('# format: r/rstar -- rho -- v -- u/umag\n')
            nx = size(gglobal.r)
            for k in arange(nx):
                fstream.write(str(gglobal.r[k]/rstar)+' '+str(rho[k])+' '+str(v[k])+' '+str(u[k]/umagtar[k])+'\n')
            fstream.close()
        nout += 1

def alltire():
    '''
    the main routine starting all the processes 
    if autostart is set True in globals, it is started automatically
    '''
    global configactual # the config set of the current simulation 
    global ifrestart
    global ufloor, rhofloor, betacoeff, csqmin # floors for primitive variables
    global raddiff, squeezemode, squeezeothersides, ufixed # if we take into account radiation diffusion along the line, matter loss from two or four sides, and also the properties of the BC for energy at the upper boundary
    global taumin, taumax # minimal/maximal optical depth (if tau< taumin, we consider the flow optically thick; if tau>taumax, we use diffusion approximation)
    global m1, mdot, mdotsink, afac, r_e, dr_e, omega, rstar, umag, xirad # mass, accretion rate, mass loss through the NS surface (ignored so far), azimuthal size of the flow, magnetosphere size, penetration depth, rotation frequency, NS radius, magnetic energy density on the NS surface, radiation diffusion parameter \xi_r
    global eta, heatingeff, nubulk, weinberg # irratiation parameter eta, heating efficiency \eta_h, bulk viscosity parameter, weinberg regime (boolean) for bulk viscosity
    global CFL, Cth, Cdiff, timeskip # Fourant-Friedrichs-Levy, thermal, and diffusion dime step multipliers; tie alias for dt adjustment
    global tscale # time scale     
    global tmax, parallelfactor, dtout, outdir, ascalias, ifhdf # maximal time, number of parallel cores used for calculations (+ one core for output!), dt for outputs, output directory, alias for graphic outputs, (boolean) if we are using HDF5 for outputs
    global gglobal, l_g, l_ghalf, dlleft, dlright  # geometry and boundary positions
    
    # initializing variables:
    if conf is None:
        configactual = config['DEFAULT']
    else:
        configactual = config[conf]
    # geometry:
    nx = configactual.getint('nx')
    nx0 = configactual.getint('nx0factor') * nx
    parallelfactor = configactual.getint('parallelfactor')
    logmesh = configactual.getboolean('logmesh')
    rbasefactor = configactual.getfloat('rbasefactor')

    # numerical parameters:
    CFL = configactual.getfloat('CFL')
    Cth = configactual.getfloat('Cth')
    Cdiff = configactual.getfloat('Cdiff')
    timeskip = configactual.getint('timeskip')
    ufloor = configactual.getfloat('ufloor')
    rhofloor = configactual.getfloat('rhofloor')
    csqmin = configactual.getfloat('csqmin')
    
    # physics:
    mu30 = configactual.getfloat('mu30')
    m1 = configactual.getfloat('m1')
    mdot = configactual.getfloat('mdot') * 4. *pi # internal units
    mdotsink = configactual.getfloat('mdotsink') * 4. *pi # internal units
    rstar = configactual.getfloat('rstar')
    b12 = 2.*mu30*(rstar*m1/6.8)**(-3) # dipolar magnetic field on the pole, 1e12Gs units
    mow = configactual.getfloat('mow')
    betacoeff = configactual.getfloat('betacoeff') * (m1)**(-0.25)/mow

    # BC modes:
    BSmode = configactual.getboolean('BSmode')
    coolNS = configactual.getboolean('coolNS')
    ufixed = configactual.getboolean('ufixed')
    squeezemode = configactual.getboolean('squeezemode')
    squeezeothersides = configactual.getboolean('squeezeothersides')

    # radiation transfer:
    raddiff = configactual.getboolean('raddiff')
    xirad = configactual.getfloat('xirad')
    taumin = configactual.getfloat('taumin')
    taumax = configactual.getfloat('taumax')

    # additional parameters:
    xifac = configactual.getfloat('xifac')
    afac = configactual.getfloat('afac')
    nubulk = configactual.getfloat('nubulk')
    weinberg = configactual.getboolean('weinberg')
    eta = configactual.getfloat('eta')
    heatingeff = configactual.getfloat('heatingeff')
    
    # derived quantities:
    r_e = configactual.getfloat('r_e_coeff') * (mu30**2/mdot)**(2./7.)*m1**(-10./7.) * xifac # magnetosphere radius
    dr_e = configactual.getfloat('drrat') * r_e
    omega = configactual.getfloat('omegafactor')*r_e**(-1.5)
    if verbose:
        print(conf+": "+str(omega))

    vout = configactual.getfloat('voutfactor')  /sqrt(r_e) # velocity at the outer boundary
    minitfactor = configactual.getfloat('minitfactor') # initial total mass in the units of equilibrium mass
    umag = b12**2*2.29e6*m1
    umagout = 0.5**2*umag*(rstar/r_e)**6
    csqout = vout**2
    if verbose:
        print(conf+": "+str(csqout))
    config.set(conf,'r_e', str(r_e))
    config.set(conf,'dr_e', str(dr_e))
    config.set(conf,'umag', str(umag))
    config.set(conf,'omega', str(omega))
    config.set(conf,'vout', str(vout))
    
    # physical scales:
    tscale = configactual.getfloat('tscale') * m1
    rscale = configactual.getfloat('rscale') * m1
    rhoscale = configactual.getfloat('rhoscale') / m1
    
    if verbose & (omega>0.):
        print(conf+": spin period "+str(2.*pi/omega*tscale)+"s")
    # output options
    tr = afac * dr_e/r_e /xifac / rstar * r_e**2.5 # replenishment time scale of the column
    if verbose:
        print(conf+": replenishment time "+str(tr))
    tmax = tr * configactual.getfloat('tmax')
    dtout = tr * configactual.getfloat('dtout')                   # tr * configactual.getfloat('dtout')
    ifplot = configactual.getboolean('ifplot')
    ifhdf = configactual.getboolean('ifhdf')
    plotalias = configactual.getint('plotalias')
    ascalias = configactual.getint('ascalias')
    outdir = configactual.get('outdir')
    ifrestart = configactual.getboolean('ifrestart')
    
    if verbose:
        print(conf+": Alfven = "+str(r_e/xifac / rstar)+"stellar radii")
        print(conf+": magnetospheric radius r_e = "+str(r_e)+" = "+str(r_e/rstar)+"stellar radii")
    
        # estimating optimal N for a linear grid
        print(conf+": nopt(lin) = "+str(r_e/dr_e * (r_e/rstar)**2/5))
        print(conf+": nopt(log) = "+str(rstar/dr_e * (r_e/rstar)**2/5))

    #    timer.start("total")

    # if the outpur directory does not exist:
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # setting geometry:
    sthd=1./sqrt(1.+(dr_e/r_e)**2) # initial sin(theta)
    rmax=r_e*sthd # slightly less then r_e
    r=((2.*(rmax-rstar)/rstar)**(arange(nx0)/double(nx0-1))+1.)*(rstar/2.) # very fine radial mesh
    g = geometry_initialize(r, r_e, dr_e, afac=afac) # radial-equidistant mesh
    #     print(str(r.min()) + " = " + str(rstar)+"?")
    if (rbasefactor is None):
        rbase = r.min()
    else:
        rbase = r.min()*rbasefactor
    g.l += rbase # we are starting from a finite radius
    if(logmesh):
        luni=exp(linspace(log((g.l).min()), log((g.l).max()), nx, endpoint=False)) # log(l)-equidistant mesh
    else:
        luni=linspace((g.l).min(), (g.l).max(), nx, endpoint=False)
    g.l -= rbase ; luni -= rbase
    rfun=interp1d(g.l, g.r, kind='linear', bounds_error = False, fill_value=(g.r[0], g.r[-1])) # interpolation function mapping l to r
    rnew=rfun(luni) # radial coordinates for the  l-equidistant mesh
    iftail = configactual.getboolean('iftail')
    # if we want to make the radial mesh even more non-linear:
    if (iftail):
        print(luni)
        rtail = configactual.getfloat('rtailfactor') * rmax
        lend = luni.max()
        luni *= sqrt((1.+exp((rnew/rtail)**2))/2.)
        luni *= lend / luni.max()    
        print(luni)
        #       ii = input('luni')
        # rfun=interp1d(g.l, g.r, kind='linear', bounds_error = False, fill_value=(g.r[0], g.r[-1])) # interpolation function mapping l to r
        rnew=rfun(luni)

    luni_half=(luni[1:]+luni[:-1])/2. # half-step l-equidistant mesh
    g = geometry_initialize(rnew, r_e, dr_e, writeout=outdir+'/geo.dat', afac=afac) # all the geometric quantities for the l-equidistant mesh
    if verbose:
        print(conf+": Across(0) = "+str(g.across[0]))
        # basic estimate for the replenishment time scale:
        print(conf+": t_r = A_\perp u_mag / g / dot{M} = "+str(g.across[0] * umag * rstar**2 / mdot*tscale)+"s")
    r=rnew # set a grid uniform in l=luni
    r_half=rfun(luni_half) # half-step radial coordinates
    ghalf = geometry_initialize(r_half, r_e, dr_e, afac=afac) # mid-step geometry in r
    ghalf.l += g.l[1]/2. # mid-step mesh starts halfstep later
    dl=g.l[1:]-g.l[:-1] # cell sizes
    dlhalf=ghalf.l[1:]-ghalf.l[:-1] # cell sizes

    BSgamma = (g.across/g.delta**2)[0]/mdot*rstar
    BSeta = (8./21./sqrt(2.)*umag*3.)**0.25*sqrt(g.delta[0])/(rstar)**0.125
    if verbose:
        print(conf+" BS parameters:")
        print(conf+"   gamma = "+str(BSgamma))
        print(conf+"   eta = "+str(BSeta))
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
    vinit=vout *sqrt(rmax/rnew) # initial velocity

    # Initial Conditions:
    # setting the initial distributions of the primitive variables:
    rho = copy(abs(mdot) / (abs(vout)+abs(vinit)) / g.across*0.5)
    # total mass
    mass = trapz(rho*g.across, x=g.l)
    meq = (g.across*umag*rstar**2)[0]
    print('meq = '+str(meq)+"\n")
    # ii = input('M')
    rho *= meq/mass * minitfactor # normalizing to the initial mass
    vinit = vout * sqrt(rmax/rnew * (rnew-rstar)/(rmax-rstar)) # to fit the v=0 condition at the surface of the star
    v = copy(vinit)
    press = umagout * (g.r/r_e) * (rho/rho[-1]+1.)/2.
    rhonoise = 1.e-3 * random.random_sample(nx) # noise (entropic)
    rho *= (rhonoise+1.)
    beta = betafun_p(Fbeta_press(rho, press, betacoeff))
    u = press * 3. * (1.-beta/2.)
    u, rho, press = regularize(u, rho, press)
    if verbose:
        print(conf+": U = "+str((u/umagtar).min())+" to "+str((u/umagtar).max()))
    m, s, e = tocon_separate(rho, vinit, u, g)
    ulast = u[-1] #
    rholast = rho[-1]
    if verbose:
        print(conf+": U/Umag(Rout) = "+str((u/umagtar)[-1]))
        print(conf+": V(Rout) = "+str(v[-1]))
    #    ii=input('m')
    
    rho1, v1, u1, urad, beta, press = toprim_separate(m, s, e, g) # primitive from conserved
    #    workout, dphi = sources(rho1, v1, u1, g, forcecheck = True) # checking whether the force corresponds to the potential
    if verbose:
        #    print(conf+": potential at the surface = "+str(-workout)+" = "+str(dphi))
        print(str((rho-rho1).std())) 
        print(str((vinit-v1).std()))
        print(str((u-u1).std())) # accuracy 1e-14
        print("primitive-conserved")
        print(conf+": rhomin = "+str(rho.min())+" = "+str(rho1.min())+" (primitive-conserved and back)")
        print(conf+": umin = "+str(u.min())+" = "+str(u1.min()))
        # ii = input('prim')
    m0=m

    t = 0. ; nout = 0
    
    # if we want to restart from a stored configuration
    # works so far correctly ONLY if the mesh is identical!
    if(ifrestart):
        ifhdf_restart = configactual.getboolean('ifhdf_restart')
        restartn = configactual.getint('restartn')
        nout = restartn
        if(ifhdf_restart):
            # restarting from a HDF5 file
            restartfile = configactual.get('restartfile')
            entryname, t, l1, r1, sth1, rho1, u1, v1, qloss1, glosave = hdf.read(restartfile, restartn)
            print("restarted from file "+restartfile+", entry "+entryname)
        else:
            # restarting from an ascii output
            restartprefix = configactual.get('restartprefix')
            restartdir = os.path.dirname(restartprefix)
            ascrestartname = restartprefix + hdf.entryname(restartn, ndig=5) + ".dat"
            lines = loadtxt(ascrestartname, comments="#")
            r1 = lines[:,0]
            r1, theta1, alpha1, across1, l1, delta1 = gread(restartdir+"/geo.dat")
            r1 /= rstar
            sth1 = sin(theta1) ; cth1 = cos(theta1)
            umagtar1 = umag * (1.+3.*(1.-sth1**2))/4. * (1./r1)**6
            rho1 = lines[:,1] ; v1 = lines[:,2] ; u1 = lines[:,3] * umagtar1
            # what about t??
            tfile = open(ascrestartname, "r") # linecache.getline(restartfile, 1)
            tline = tfile.readline()
            tfile.close()
            t=double(re.search(r'\d+.\d+', tline).group()) / tscale
            print("restarted from ascii output "+ascrestartname)
            print("t = "+str(t))
        if verbose:
            print(conf+": r from "+str(r.min()/rstar)+" to "+str(r.max()/rstar))
            print(conf+": r1 from "+str(r1.min())+" to "+str(r1.max()))
        if(r.max()>(1.01*r1.max()*rstar)):
            print("restarting: size does not match!")
            return(1)
        if ((size(r1) != nx) | (r.max() < (0.99 * r1.max()))):
            # minimal heat and minimal mass
            #
            #            rhorestartfloor = 1e-5 * mdot / r**1.5 ; urestartfloor = 1e-5 * rhorestartfloor / r
            rho1 = maximum(rho1, rhofloor) ; u1 = maximum(u1, ufloor)
            print("interpolating from "+str(size(r1))+" to "+str(nx))
            print("rho1 from "+str(rho1.min())+" to "+str(rho1.max()))
            rhofun = interp1d(log(r1), log(rho1), kind='linear', bounds_error=False, fill_value = (log(rho1[0]), log(rho1[-1])))
            vfun = interp1d(log(r1), v1, kind='linear', bounds_error=False, fill_value = (v1[0], v1[-1]))
            ufun = interp1d(log(r1), log(u1), kind='linear', bounds_error=False, fill_value = (log(u1[0]), log(u1[-1])))
            rho = exp(rhofun(log(r/rstar))) ; v = vfun(log(r/rstar)) ; u = exp(ufun(log(r/rstar)))
            print("interpolated values: rho = "+str(rho.min())+" to "+str(rho.max()))
            print("interpolated values: v = "+str(v.min())+" to "+str(v.max()))
            print("interpolated values: u = "+str(u.min())+" to "+str(u.max()))
            print("v[-1] = "+str(v[-1]))
            ulast = u[-1]
        else:
            print("restarting with the same resolution")
            rho = rho1 ; v = v1 ; u = u1
            # r *= rstar
            #        print(r)
            #        print(r1)
            #        ii = input('r')
        beta = betafun(Fbeta(rho, u, betacoeff))
        press = u / (3.*(1.-beta/2.))
        if ifplot:
            plots.uplot(g.r, u, rho, g.sth, v, name=outdir+'/utie_restart', umagtar = umagtar)
            plots.vplot(g.r, v, sqrt(4./3.*u/rho), name=outdir+'/vtie_restart')
            plots.someplots(g.r, [u/rho**(4./3.)], name=outdir+'/entropy_restart', ytitle=r'$S$', ylog=True)
            plots.someplots(g.r, [beta, 1.-beta], formatsequence=['r-', 'b-'],
                            name=outdir+'/beta_restart', ytitle=r'$\beta$, $1-\beta$', ylog=True)

        m, s, e = tocon_separate(rho, v, u, g)
        nout = restartn
        #    plots.uplot(g.r, u, rho, g.sth, v, name=outdir+'/utie_restart', umagtar = umagtar)
    fname=outdir+'/tireout_start.dat'
    if verbose:
        print("ASCII initial state")
        fstream=open(fname, 'w')
        #        fstream.write('# t = '+str(t*tscale)+'s\n')
        fstream.write('# format: r/rstar -- rho -- v -- u/umag\n')
        for k in arange(nx):
            fstream.write(str(g.r[k]/rstar)+' '+str(rho[k])+' '+str(v[k])+' '+str(u[k]/umagtar[k])+'\n')
        fstream.close()
    #################
    ulast = u[-1]
    if ulast < 0.:
        print("negative internal energy in the IC\n")
        print(ulast)
        ii = input("C")        
    if(ifhdf):
        hname = outdir+'/'+'tireout.hdf5'
        hfile = hdf.init(hname, g, configactual) # , m1, mdot, eta, afac, re, dre, omega)
        print("output to "+hname)
    else:
        hfile = None
        
    crash = False # we have not crashed (yet)

    # if we want the simulation domains to be unequal in size, scaling as a square of their number:
    ifquadratic = configactual.getboolean('ifquadratic')
    if ifquadratic:
        aind = ceil(nx/parallelfactor**2).astype(int) # ceil(6.*nx/parallelfactor/(parallelfactor+1.)/(2.*parallelfactor+1.)).astype(int)
        print("first partition is "+str())
        inds = aind * (arange(parallelfactor-1, dtype = int)+1)**2
        print(inds)
    else:
        inds = parallelfactor
    
    # if we are doing parallel calculation:
    #    if parallelfactor > 1:
    gglobal = g
    l_g = geometry_split(g, inds)
    l_ghalf = geometry_split(ghalf, inds, half = True)

    '''
    print(size(l_g[0].r))
    print(size(l_g[1].r))
    print(size(l_g[2].r))
    print(shape(array_split(m, inds)[0]))
    print(shape(array_split(m, inds)[1]))
    print(shape(array_split(m, inds)[2]))
    print(size(l_g[2].r))
    print(size(l_ghalf[2].r))
    ii = input("inds")
    '''

    l_m = array_split(m, inds) ; l_e = array_split(e, inds) ; l_s = array_split(s, inds)
    l_con = [{'N': i, 'm': l_m[i], 's': l_s[i], 'e': l_e[i]} for i in range(parallelfactor)] # list of conserved quantities, each item organized as a dictionary
    l_u = array_split(u, inds) ; l_rho = array_split(rho, inds) ; l_v = array_split(v, inds)
    l_umagtar = array_split(umagtar, inds)
    l_press = array_split(press, inds) ;    l_urad = array_split(urad, inds)
    beta = betafun(Fbeta(rho, u, betacoeff))
    l_beta = array_split(beta, inds) 

    l_prim = [{'N': i, 'rho': l_rho[i], 'v': l_v[i], 'u': l_u[i], 'press': l_press[i], 'urad': l_urad[i], 'beta': l_beta[i]} for i in range(parallelfactor)]
    
    print(shape(l_prim))
    print(shape(l_g))
    dlleft_tmp, dlright_tmp = dlbounds_define(l_g)
    dlleft = copy(dlleft_tmp) ; dlright = copy(dlright_tmp)
    print(dlright)
    [ l_con[i].update([ ('umagtar', l_umagtar[i]),
                        ('dlleft', dlleft[i]), ('dlright', dlright[i])]) for i in range(parallelfactor) ]
    #        ii = input('dll')

    ghostpipe1 = []
    ghostpipe2 = []
    dtpipes1 = []
    opipes1 = []
    dtpipes2 = []
    opipes2 = []

    #    dtpipe1, dtpipe2 = Pipe(duplex = True)
    #    outpipe1, outpipe2 = Pipe(duplex = True)
    
    for k in range(parallelfactor-1):
        # connects kth with (k+1)th
        gh1, gh2 = Pipe(duplex = True) # pipes between the domains
        ghostpipe1.append(gh1) ;  ghostpipe2.append(gh2)
        dt1, dt2 = Pipe(duplex = True) # pipes exchanging the time step
        dtpipes1.append(dt1) ;  dtpipes2.append(dt2)
    for k in range(parallelfactor):
        o1, o2 = Pipe(duplex = True) # pipes for output
        opipes1.append(o1) ; opipes2.append(o2)
    plist = []
        
    #    if ifrestart:
    #        fflux=open(outdir+'/'+'flux.dat', 'a')
    #        ftot=open(outdir+'/'+'totals.dat', 'a')
    #    else:
    fflux=open(outdir+'/'+'flux.dat', 'w')
    ftot=open(outdir+'/'+'totals.dat', 'w')

    op = Process(target = tireouts, args = (opipes1, hfile, fflux, ftot), kwargs = {'t': t, 'nout': nout}) # output process
    for k in range(parallelfactor):
        # starting in reverse order
        if k>0:
            ghostleft = ghostpipe2[k-1]
            dtpipe = dtpipes2[k-1]
        else:
            ghostleft = None
            dtpipe = dtpipes1
        if k<(parallelfactor-1):
            ghostright = ghostpipe1[k]
        else:
            ghostright = None
        outpipe = opipes2[k]
        p = Process(target = onedomain, args = (l_g[k], l_con[k], ghostleft, ghostright, dtpipe, outpipe, hfile), kwargs = {'t': t, 'nout': nout}) # main calculations 
        plist.append(p)

    for k in range(parallelfactor):
        plist[k].start()
    op.start()
    #    hfile.close()
    '''
    for k in range(parallelfactor):
        plist[k].join()
    op.join()
    ''' 
    fflux.close() ; ftot.close()
    if(ifhdf):
        hdf.close(hfile)
    print("zehu\n")
    for k in range(parallelfactor):
        plist[k].join()
    op.join()
# if we start the simulation automatically:
if autostart:
    alltire()
