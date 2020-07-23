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
from multiprocessing import Pool

# configuration file:
import configparser as cp
conffile = 'globals.conf'
config = cp.ConfigParser(inline_comment_prefixes="#")
config.read(conffile) 
ifplot = config['DEFAULT'].getboolean('ifplot')
ifhdf = config['DEFAULT'].getboolean('ifhdf')
verbose = config['DEFAULT'].getboolean('verbose')
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
              ["advance"])

def time_step(prim):
    # time step adjustment:
    nd = prim['N'] # number of the domain
    rho = prim['rho'] ; v = prim['v'] ; u = prim['u']
    dlmin = (l_g[nd].l[1:]-l_g[nd].l[:-1]).min()
    dlhalf = (l_ghalf[nd].l[1:]-l_ghalf[nd].l[:-1])
    csqest = 4./3.*u/rho
    dt_CFL = CFL * dlmin / sqrt(csqest.max()+(v**2).max())
    qloss = qloss_separate(rho, v, u, l_g[nd])
    dt_thermal = Cth * abs(u*l_g[nd].across/qloss)[where((u*qloss)>0.)].min()
    
    if(raddiff):
        ctmp = dlhalf**2 * 3.*rho[1:-1]
        dt_diff = Cdiff * quantile(ctmp[where(ctmp>0.)], 0.1) # (dx^2/D)
    else:
        dt_diff = dt_CFL * 5. # effectively infinity ;)
    dt = 1./(1./dt_CFL + 1./dt_thermal + 1./dt_diff)
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
    tt = copy(tau)*0.
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
    tt=copy(x)*0.
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
        #        ip = input('trat')
    return tt

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

def tocon(prim):
    '''
    computes conserved quantities from primitives
    '''
    m = con['m'] ; s = con['s'] ; e = con['e'] ; nd = con['N']
    gnd = l_g[nd]
    
    m=rho*gnd.across # mass per unit length
    s=m*v # momentum per unit length
    e=(u+rho*(v**2/2.- 1./gnd.r - 0.5*(omega*gnd.r*gnd.sth)**2))*gnd.across  # total energy (thermal + mechanic) per unit length
    return {'m': m, 's': s, 'e': e, 'N': nd}

def toprim(con):
    '''
    convert conserved quantities to primitives
    '''
    m = con['m'] ; s = con['s'] ; e = con['e'] ; nd = con['N']
    gnd = l_g[nd]
    rho = m/gnd.across
    v = s/m
    u = (e-m*(v**2/2.-1./gnd.r-0.5*(gnd.r*gnd.sth*omega)**2))/gnd.across
    #    umin = u.min()
    beta = betafun(Fbeta(rho, u, betacoeff))
    press = u/3./(1.-beta/2.)
    u, rho, press = regularize(u, rho, press)
    urad = u*(1.-beta)/(1.-beta/2.)
    prim = {'rho': rho, 'v': v, 'u': u, 'beta': beta, 'urad': urad, 'press': press, 'N': nd}
    return prim

def diffuse(rho, urad, v, dl, across, gamma):
    '''
    radial energy diffusion;
    calculates energy flux contribution already at the cell boundary
    across should be set at half-steps
    '''
    #    rho_half = (rho[1:]+rho[:-1])/2. # ; v_half = (v[1:]+v[:-1])/2.  ; u_half = (u[1:]+u[:-1])/2.
    rtau_right = rho[1:] * dl / 2.# optical depths along the field line, to the right of the cell boundaries
    rtau_left = rho[:-1] * dl / 2. # -- " -- to the left -- " --
    rtau = rtau_left + rtau_right
    rtau_exp = tratfac(copy(rtau))
    
    duls_half =  nubulk  * (( urad * v)[1:] - ( urad * v)[:-1])\
                 *across / 3. * rtau_exp #  / (rtau_left + rtau_right)
    if (weinberg):
        gamma_half = (gamma[1:]+gamma[:-1])/2.
        duls_half *= (gamma_half-4./3.)**2 # Weinberg 1972: 2.11.28
        # bulk viscosity in monatomic gas + radiation disappears is gamma=4/3
        # note that bulk viscosity is mediated, in this approximation, by photons only
    # -- photon bulk viscosity
    dule_half = ((urad)[1:] - (urad)[:-1])\
                *across / 3. * rtau_exp # / (rtau_left + rtau_right)
    dule_half +=  duls_half * (v[1:]+v[:-1])/2. # adding the viscous energy flux

    # -- radial diffusion
    # introducing exponential factors helps reduce the numerical noise from rho variations

    # no diffusion for the last cell:
    #    dule_half[-1] = 0. ; duls_half[-1] = 0.
    return -duls_half, -dule_half 

def bound_fluxes(l_prim, l_con, edot = 0.):
    '''
    computes the fluxes at the domain boundaries
    '''
    dno = size(l_prim)
    #    sleft = zeros(dno) ; pleft = zeros(dno) ; feleft = zeros(dno)
    #    sright = zeros(dno) ; pright = zeros(dno) ; feright = zeros(dno)
    BClist = []
    
    #    sleft[0] = -mdotsink ; sright[-1] = -mdot
    #    feleft[0] = 0. ; feright[-1] = -edot
    
    for nd in range(dno):
        if nd == 0:
            sleft = 0. ; pleft = 0. ; feleft = 0.
        if nd == (dno-1):
            sright = -mdot ; pright = 0.; feright = -edot
        if nd > 0:
            across_left = l_g[nd-1].across[-1] ;  r_left = l_g[nd-1].r[-1] ;  sth_left = l_g[nd-1].sth[-1]
            rho_left = l_prim[nd-1]['rho'][-1] ; v_left = l_prim[nd-1]['v'][-1] ; u_left = l_prim[nd-1]['u'][-1]
            sleft = rho_left * v_left * across_left
            beta_left = betafun(Fbeta(rho_left, u_left, betacoeff))
            press_left = u_left/3./(1.-beta_left/2.)
            pleft = across_left * (sleft * v_left + press_left)
            feleft = across_left*v_left*(u_left+press_left+(v_left**2/2.-1./r_left-0.5*(omega*r_left*sth_left)**2)*rho_left)
            # need to include energy diffusion!
        if nd<(dno-1):
            across_right = l_g[nd+1].across[0] ;  r_right = l_g[nd+1].r[0] ;  sth_right = l_g[nd+1].sth[0]
            rho_right = l_prim[nd+1]['rho'][0] ; v_right = l_prim[nd+1]['v'][0] ; u_right = l_prim[nd+1]['u'][0]
            sright = rho_right * v_right * across_right
            beta_right = betafun(Fbeta(rho_right, u_right, betacoeff))
            press_right = u_right/3./(1.-beta_right/2.)
            pright = across_right * (sright * v_right + press_right)
            feright = across_right*v_right*(u_right+press_right+(v_right**2/2.-1./r_right-0.5*(omega*r_right*sth_right)**2)*rho_right)
        BClist.append({'sleft': sleft, 'sright': sright, 'pleft': pleft, 'pright': pright, 'feleft': feleft, 'feright': feright})
    BClist1 = copy(BClist)
    for nd in range(dno-1):
        fm = asarray([BClist[nd+1]['sleft'], BClist[nd]['sright']])
        fs = asarray([BClist[nd+1]['pleft'], BClist[nd]['pright']])
        fe = asarray([BClist[nd+1]['feleft'], BClist[nd]['feright']])
        m = asarray([l_con[nd]['m'][-1], l_con[nd+1]['m'][0]])
        s = asarray([l_con[nd]['s'][-1], l_con[nd+1]['s'][0]])
        e = asarray([l_con[nd]['e'][-1], l_con[nd+1]['e'][0]])
        v = asarray([l_prim[nd]['v'][-1], l_prim[nd+1]['v'][0]])
        # ADD g1!!!
        csq = asarray([l_prim[nd]['press'][-1]/l_prim[nd]['rho'][-1], l_prim[nd+1]['press'][0]/l_prim[nd+1]['rho'][0]])
        print("press = "+str(l_prim[nd]['press'][-1]))
        print("v = "+str(v))
        print("csq = "+str(csq))
        vl, vm, vr = sigvel_mean(v, sqrt(csq))
        fm_half, fs_half, fe_half = solv.HLLC([fm, fs, fe], [m, s, e], vl, vr, vm)
        BClist1[nd+1].update([('sleft', fm_half), ('pleft', fs_half),('feleft', fe_half)])
        BClist1[nd].update([('sright', fm_half), ('pright', fs_half),('feright', fe_half)])
    print("before:\n")
    print(BClist)
    print("after: \n")
    print(BClist1)
    ii = input('BC')
    return BClist1 
            
def fluxes(prim):
    '''
    computes the fluxes of conserved quantities, given primitives; 
    radiation diffusion flux is not included, as it is calculated at halfpoints
    inputs:
    rho -- density, v -- velocity, u -- thermal energy density
    g is geometry (structure)
    Note: fluxes do not include diffusion (added separately)
    '''
    nd = prim['N'] ; rho = prim['rho'] ; v = prim['v'] ; u = prim['u']
    gnd = l_g[nd]
    across = gnd.across
    s = rho*v*across # mass flux (identical to momentum per unit length -- can we use it?)
    beta = betafun(Fbeta(rho, u, betacoeff))
    press = u/3./(1.-beta/2.)
    p = across*(rho*v**2+press) # momentum flux
    fe = across*v*(u+press+(v**2/2.-1./gnd.r-0.5*(omega*gnd.r*gnd.sth)**2)*rho) # energy flux without diffusion
    return s, p, fe

def sources(prim, ltot=0., dmsqueeze = 0., desqueeze = 0., forcecheck = False):
    '''
    computes the RHSs of conservation equations
    no changes in mass
    momentum injection through gravitational and centrifugal forces
    energy losses through the surface
    outputs: dm, ds, de, and separately the amount of energy radiated per unit length per unit time ("flux")
    additional output:  equilibrium energy density
    if the "forcecheck" flag is on, outputs the grav.potential difference between the outer and inner boundaries and compares to the work of the force along the field line
    '''
    nd = prim['N'] ; rho = prim['rho'] ; v = prim['v'] ; u = prim['u']
    gnd = l_g[nd]
    #  sinsum=sina*cth+cosa*sth # cos( pi/2-theta + alpha) = sin(theta-alpha)
    #     tau = rho*g.across/(4.*pi*g.r*g.sth*afac)
    tau = rho * gnd.delta # tau in transverse direction
    tauphi = rho * gnd.across / gnd.delta / 2. # optical depth in azimuthal direction
    taueff = copy(1./(1./tau + 1./tauphi))
    taufac = taufun(taueff)    # 1.-exp(-tau)
    #    taufac = 1. # !!! temporary
    gamefac = tratfac(tau)
    gamedd = eta * ltot * gamefac
    sinsum = copy(gnd.sina*gnd.cth+gnd.cosa*gnd.sth) # sin(theta+alpha)
    force = copy((-sinsum/gnd.r**2*(1.-gamedd)+omega**2*gnd.r*gnd.sth*gnd.cosa)*rho*gnd.across) # *taufac
    if(forcecheck):
        network = simps(force/(rho*gnd.across), x=gnd.l)
        return network, (1./gnd.r[0]-1./gnd.r[-1])
    beta = prim['beta'] # betafun(Fbeta(rho, u, betacoeff))
    urad = prim['urad']
    #     urad = copy(u * (1.-beta)/(1.-beta/2.))
    urad = (urad+abs(urad))/2.
    qloss = copy(2.*urad/(xirad*taueff+1.)*(gnd.across/gnd.delta+2.*gnd.delta)*taufac)  # diffusion approximation; energy lost from 4 sides
    irradheating = heatingeff * eta * mdot *afac / gnd.r * gnd.sth * sinsum * taufun(tau)
    #    ueq = heatingeff * mdot / g.r**2 * sinsum * urad/(xirad*tau+1.)
    dm = copy(rho*0.-dmsqueeze)
    dudt = copy(v*force-qloss+irradheating)
    ds = copy(force - dmsqueeze * v) # lost mass carries away momentum
    de = copy(dudt - desqueeze) # lost matter carries away energy (or enthalpy??)
    #    return dm, force, dudt, qloss, ueq
    return {'m': dm, 's': ds, 'e': de, 'flux': qloss} #, ueq

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

def derivo(con, s_half, p_half, fe_half, src):
    '''
    main advance step
    input: three densities, l (midpoints), three fluxes (midpoints), three sources, timestep, r, sin(theta), cross-section
    output: three temporal derivatives later used for the time step
    includes boundary conditions for mass and energy!
    '''

    m = con['m'] ; s = con['s'] ; e = con['e'] ; nd = con['N']
    #    print("r = "+str(dlright))
    l_half = l_ghalf[nd].l
    dlleft_nd = con['dlleft'] ; dlright_nd = con['dlright']
    # why is dlleft not resolved as a global?
    #    print("main_step: mmin = "+str(m.min()))
    dm = src['m'] ; ds = src['s'] ; de = src['e'] 
    nl=size(m)
    dmt=zeros(nl) ; dst=zeros(nl); det=zeros(nl)
    dmt[1:-1] = -(s_half[1:]-s_half[:-1])/(l_half[1:]-l_half[:-1]) + dm[1:-1]
    dst[1:-1] = -(p_half[1:]-p_half[:-1])/(l_half[1:]-l_half[:-1]) + ds[1:-1]
    det[1:-1] = -(fe_half[1:]-fe_half[:-1])/(l_half[1:]-l_half[:-1]) + de[1:-1]

    print("conpleft = "+str(con['pleft']))
    #left boundary conditions:
    dmt[0] = -(s_half[0]-con['sleft'])/dlleft_nd +dm[0]
    #    dst[0] = (-mdotsink-s[0])/dt # ensuring approach to -mdotsink
    dst[0] = -(p_half[0]-con['pleft'])/dlleft_nd +ds[0] # mdotsink_eff does not enter here, as matter should escape sideways, but through the bottom
    #     edotsink_eff = mdotsink * (e[0]/m[0])
    det[0] = -(fe_half[0]-con['feleft'])/dlleft_nd + de[0] # no energy sink anyway
    # right boundary conditions:
    dmt[-1] = -(con['sright']-s_half[-1])/dlright_nd +dm[-1]
    #    dst[-1] = (-mdot-s[-1])/dt # ensuring approach to -mdot
    
    dst[-1] = -(con['pright'] - p_half[-1])/dlright_nd + ds[-1] # momentum flow through the outer boundary (~= pressure in the disc)
    #    edot =  abs(mdot) * 0.5/g.r[-1] + s[-1]/m[-1] * u[-1] # virial equilibrium
    det[-1] = -(con['feright'] - fe_half[-1])/dlright_nd + de[-1]

    return {'m': dmt, 's': dst, 'e': det}

def RKstep(con):
    # m, s, e, g, ghalf, dl, dlleft, dlright, ltot=0., umagtar = None, momentum_inflow = None, energy_inflow =  None):
    '''
    calculating elementary increments of conserved quantities
    '''
    dt = con['dt']
    prim = toprim(con) # primitive from conserved
    fm, fs, fe = fluxes(prim)
    g1 = Gamma1(5./3., prim['beta'])
    rho = prim['rho'] ;  press = prim['press'] ;  v = prim['v'] ; urad = prim['urad']
    nd = prim['N']
    csq=g1*press/rho
    if(csq.min()<csqmin):
        wneg = (csq<=csqmin)
        csq[wneg] = csqmin
    cs = sqrt(csq)
    vl, vm, vr =sigvel_mean(v, cs)
    # sigvel_linearized(v, cs, g1, rho, press)
    # sigvel_isentropic(v, cs, g1, csqmin=csqmin)
    if any(vl>=vm) or any(vm>=vr):
        wwrong = (vl >=vm) | (vm<=vr)
        print("rho = "+str((rho[1:])[wwrong]))
        print("u = "+str((u[1:])[wwrong]))
        print(vl[wwrong])
        print(vm[wwrong])
        print(vr[wwrong])
        print(vm[wwrong])
        print("R = "+str(ghalf.r[wwrong]))
        print("signal velocities crashed")
        #        ii=input("cs")
    m = con['m'] ; s = con['s'] ; e = con['e']
    fm_half, fs_half, fe_half =  solv.HLLC([fm, fs, fe], [m, s, e], vl, vr, vm)
    if(raddiff):
        duls_half, dule_half = diffuse(rho, urad, v, dl, ghalf.across, g1)
        # radial diffusion suppressed, if transverse optical depth is small:
        duls_half *= 1.-exp(-ghalf.delta * (rho[1:]+rho[:-1])/2.)
        dule_half *= 1.-exp(-ghalf.delta * (rho[1:]+rho[:-1])/2.)
        fs_half += duls_half ; fe_half += dule_half
    if(squeezemode):
        umagtar = con['umagtar']
        #        if umagtar is None:
        #            umagtar = umag * (1.+3.*g[nd].cth**2)/4. * (rstar/g[nd].r)**6
        dmsqueeze = 2. * m * sqrt(g1*maximum((press-umagtar)/rho, 0.))/l_g[nd].delta
        if squeezeothersides:
            dmsqueeze += 4. * m * sqrt(g1*maximum((press-umagtar)/rho, 0.))/ (l_g[nd].across / l_g[nd].delta)
        desqueeze = dmsqueeze * (e+press* l_g[nd].across) / m # (e-u*g.across)/m
    else:
        dmsqueeze = 0.
        desqueeze = 0.
        
    # dm, ds, de, flux
    src = sources(prim, ltot=0., dmsqueeze = dmsqueeze, desqueeze = desqueeze)
    
    ltot=trapz(src['flux'], x=l_g[nd].l) 
    # dmt, dst, det
    print(con['dlright'])
    ds = derivo(con, fm_half, fs_half, fe_half, src)
    con1 = {'N': nd, 'm': m+ds['m']*dt, 's': s+ds['s']*dt, 'e': e+ds['e']*dt, 'ltot': ltot}
    return con1

################################################################################
print("if you want to start the simulation, now type `alltire(`conf')` \n")

def alltire(conf):
    '''
    the main routine bringing all together
    '''
    global configactual
    global ufloor, rhofloor, betacoeff, csqmin
    global raddiff, squeezemode, ufixed, squeezeothersides
    global taumin, taumax
    global m1, mdot, mdotsink, afac, r_e, dr_e, omega, rstar, umag, xirad
    global eta, heatingeff, nubulk, weinberg
    global CFL, Cth, Cdiff

    global l_g, l_ghalf, dlleft, dlright  # geometry and boundary positions
    
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
    dtout = 0.0001/tscale                    # tr * configactual.getfloat('dtout')
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

    timer.start("total")

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
    luni_half=(luni[1:]+luni[:-1])/2. # half-step l-equidistant mesh
    rfun=interp1d(g.l, g.r, kind='linear', bounds_error = False, fill_value=(g.r[0], g.r[-1])) # interpolation function mapping l to r
    rnew=rfun(luni) # radial coordinates for the  l-equidistant mesh
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

    # if we are doing parallel calculation:
    if parallelfactor > 1:
        l_g = geometry_split(g, parallelfactor)
        l_ghalf = geometry_split(ghalf, parallelfactor, half = True)
        # now, g is a list of geometries
    else:
        print("to perform parallel computation, you need parallelfactor > 1!")
        exit(1)
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
    vinit=vout *sqrt(rmax/g.r) # initial velocity

    # Initial Conditions:
    # setting the initial distributions of the primitive variables:
    rho = abs(mdot) / (abs(vout)+abs(vinit)) / g.across*0.5
    # total mass
    mass = trapz(rho*g.across, x=g.l)
    meq = (g.across*umag*rstar**2)[0]
    print('meq = '+str(meq)+"\n")
    #    ii = input('M')
    rho *= meq/mass * minitfactor # normalizing to the initial mass
    vinit *= ((g.r-rstar)/(rmax-rstar))**0.5 # to fit the v=0 condition at the surface of the star
    v = copy(vinit)
    press = umagtar[-1] * (g.r/r_e) * (rho/rho[-1]+1.)/2.
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
        # print(str((rho-rho1).std())) 
        # print(str((vinit-v1).std()))
        # print(str((u-u1).std())) # accuracy 1e-14
        # print("primitive-conserved")
        print(conf+": rhomin = "+str(rho.min())+" = "+str(rho1.min())+" (primitive-conserved and back)")
        print(conf+": umin = "+str(u.min())+" = "+str(u1.min()))
    m0=m
    
    t=0.;  tstore=0.  ; nout=0

    # if we want to restart from a stored configuration
    # works so far correctly ONLY if the mesh is identical!
    if(ifrestart):
        ifhdf_restart = configactual.getboolean('ifhdf_restart')
        restartn = configactual.getint('restartn')
        
        if(ifhdf_restart):
            # restarting from a HDF5 file
            restartfile = configactual.get('restartfile')
            entryname, t, l1, r1, sth1, rho1, u1, v1, qloss1, glosave = hdf.read(restartfile, restartn)
            tstore = t
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
            tstore = t
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

        m, s, e = cons(rho, v, u, g)
        nout = restartn

    ulast = u[-1]
    if ulast < 0.:
        print("negative internal energy in the IC\n")
        print(ulast)
        ii = input("C")
        
    ltot=0. # estimated total luminosity
    if(ifrestart):
        fflux=open(outdir+'/'+'flux.dat', 'a')
        ftot=open(outdir+'/'+'totals.dat', 'a')
    else:
        fflux=open(outdir+'/'+'flux.dat', 'w')
        ftot=open(outdir+'/'+'totals.dat', 'w')

    if(ifhdf):
        hname = outdir+'/'+'tireout.hdf5'
        hfile = hdf.init(hname, g, configactual) # , m1, mdot, eta, afac, re, dre, omega)
        print("output to "+hname)
        
    crash = False # we have not crashed (yet)

    l_m = array_split(m, parallelfactor) ; l_e = array_split(e, parallelfactor) ; l_s = array_split(s, parallelfactor)
    l_con = [{'N': i, 'm': l_m[i], 's': l_s[i], 'e': l_e[i]} for i in range(parallelfactor)] # list of conserved quantities, each item organized as a dictionary
    l_u = array_split(u, parallelfactor) ; l_rho = array_split(rho, parallelfactor) ; l_v = array_split(v, parallelfactor)
    l_umagtar = array_split(umagtar, parallelfactor)
    l_press = array_split(press, parallelfactor) ;    l_urad = array_split(urad, parallelfactor)
    l_prim = [{'N': i, 'rho': l_rho[i], 'v': l_v[i], 'u': l_u[i], 'press': l_press[i], 'urad': l_urad[i]} for i in range(parallelfactor)]
    pool = multiprocessing.Pool(parallelfactor)
    print(shape(l_prim))
    print(shape(l_g))
    dlleft_tmp, dlright_tmp = dlbounds_define(l_g)
    dlleft = copy(dlleft_tmp) ; dlright = copy(dlright_tmp)
    print(dlright)
    #        ii = input('dll')
        
    timer.start("total")
    while(t<tmax):
        timer.start_comp("advance")

        dtlist = pool.map(time_step, l_prim) 
        dt = asarray(dtlist).min() # later, we can try different time steps at different distances

        # BC
        BCList = bound_fluxes(l_prim, l_con) # calculates fluxes for BC as a list of dictionaries. Must include Riemann solver!
        # each dic is {'sleft': , 'sright': , 'pleft': , 'pright': , 'feleft': , 'feright':}
        # Adding BC to the con dict:
        [ l_con[i].update([('sleft', BCList[i]['sleft']), ('sright',BCList[i]['sright']),
                           ('pleft', BCList[i]['pleft']), ('pright',BCList[i]['pright']),
                           ('feleft', BCList[i]['feleft']), ('feright', BCList[i]['feright']),
                           ('dt', dt), ('umagtar', l_umagtar[i]),
                           ('dlleft', dlleft[i]), ('dlright', dlright[i])]) for i in range(parallelfactor) ]
        #
        l_con = pool.map(RKstep, l_con)
        t += dt
        l_prim = pool.map(toprim, l_con) # primitive from conserved

        print("ltot = "+str([l_con[i]['ltot'] for i in range(parallelfactor)]))
        ii = input('dT')
        
        
        timer.stop_comp("advance")
        timer.lap("step")
        if (t>=tstore) | crash:
            # need to stitch the parts together:
            rho = (asarray( [ l_prim[i]['rho'] for i in range(parallelfactor) ] )).flatten()
            u = (asarray( [ l_prim[i]['u'] for i in range(parallelfactor) ] )).flatten()
            v = (asarray( [ l_prim[i]['v'] for i in range(parallelfactor) ] )).flatten()
            urad =  (asarray( [ l_prim[i]['urad'] for i in range(parallelfactor) ] )).flatten()
            qloss = qloss_separate(rho, v, u, g)
            ltot = trapz(qloss, x=g.l)
            m = (asarray( [ l_con[i]['m'] for i in range(parallelfactor) ] )).flatten()
            s = (asarray( [ l_con[i]['s'] for i in range(parallelfactor) ] )).flatten()
            e = (asarray( [ l_con[i]['e'] for i in range(parallelfactor) ] )).flatten()
            tstore += dtout
            timer.start("io")
            #            rho, v, u, urad, beta, press = toprim(m, s, e, g) # primitive from conserved            tstore+=dtout
            if verbose:
                print(conf+": t = "+str(t*tscale)+"s")
                # print("dt = "+str(dt*tscale)+"s")
                print(conf+": ltot = "+str(ltot))
                print(conf+": V (out) = "+str(v[-1]))
                print(conf+": U/Umag (out) = "+str(u[-1]/umagtar[-1]))
            fflux.write(str(t*tscale)+' '+str(ltot)+'\n')
            fflux.flush()
            if ifplot & (nout%plotalias == 0):
                print("plotting")
                plots.uplot(g.r, u, rho, g.sth, v, name=outdir+'/utie{:05d}'.format(nout), umagtar = umagtar)
                plots.vplot(g.r, v, sqrt(4./3.*u/rho), name=outdir+'/vtie{:05d}'.format(nout))
                plots.someplots(g.r, [u/rho**(4./3.)], name=outdir+'/entropy{:05d}'.format(nout), ytitle=r'$S$', ylog=True)
                plots.someplots(g.r, [(u-urad)/(u-urad/2.), 1.-(u-urad)/(u-urad/2.)],
                                name=outdir+'/beta{:05d}'.format(nout), ytitle=r'$\beta$, $1-\beta$',
                                ylog=True, formatsequence=['r-', 'b-'])
                plots.someplots(g.r, [qloss*g.l, -cumtrapz(qloss[::-1], x=g.l[::-1], initial = 0.)[::-1]],
                                name=outdir+'/qloss{:05d}'.format(nout),
                                ytitle=r'$\frac{d^2 E}{d\ln l dt}$', ylog=False,
                                formatsequence = ['k-', 'r-'])
            mtot=trapz(m, x=g.l)
            etot=trapz(e, x=g.l)
            if verbose:
                print(conf+": mass = "+str(mtot))
                print(conf+": ltot = "+str(ltot))
                print(conf+": energy = "+str(etot))
                print(conf+": momentum = "+str(trapz(s, x=g.l)))
            
            ftot.write(str(t*tscale)+' '+str(mtot)+' '+str(etot)+'\n')
            ftot.flush()
            if(ifhdf):
                hdf.dump(hfile, nout, t, rho, v, u, qloss)
            if not(ifhdf) or (nout%ascalias == 0):
                # ascii output:
                # print(nout)
                fname=outdir+'/tireout{:05d}'.format(nout)+'.dat'
                if verbose:
                    print(conf+" ASCII output to "+fname)
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
            if(crash):
                break
    fflux.close()
    ftot.close()
    if(ifhdf):
        hdf.close(hfile)
# if you want to make a movie of how the velocity changes with time:
# ffmpeg -f image2 -r 15 -pattern_type glob -i 'out/vtie*0.png' -pix_fmt yuv420p -b 4096k v.mp4
# if you want the main procedure to start running immediately after the compilation, uncomment the following:
# alltire('globals')

