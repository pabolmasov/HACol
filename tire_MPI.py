from scipy.integrate import *
from scipy.interpolate import *

import numpy.random
from numpy.random import rand
from numpy import *
# import time
import os
# import re
import linecache
import os.path
# import imp
import sys
import configparser as cp
import gc

if(size(sys.argv)>1):
    print("launched with arguments "+str(', '.join(sys.argv)))
    # new conf file
    conf=sys.argv[1]
    print(conf+" configuration set by the arguments")
else:
    conf='DEFAULT'

from mpi4py import MPI
# MPI parameters:
comm = MPI.COMM_WORLD
crank = comm.Get_rank()
csize = comm.Get_size()
    
# configuration file (read by each thread):
conffile = 'globals.conf'
config = cp.ConfigParser(inline_comment_prefixes="#")
config.read(conffile) 
ifplot = config[conf].getboolean('ifplot')
ifhdf = config[conf].getboolean('ifhdf')
verbose = config[conf].getboolean('verbose')
if crank != 0:
    verbose = False
autostart = config[conf].getboolean('autostart')
# initializing variables:
if conf is None:
    configactual = config['DEFAULT']
else:
    configactual = config[conf]
# geometry:
nx = configactual.getint('nx')
nx0 = configactual.getint('nx0factor') * nx
parallelfactor = configactual.getint('parallelfactor')
last = parallelfactor-1 ; first = 0

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
cslimit = configactual.getboolean('cslimit')

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
cooltwosides = configactual.getboolean('cooltwosides')

# radiation transfer:
ifthin = configactual.getboolean('ifthin')
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
ifturnoff = configactual.getboolean('ifturnoff')
if ifturnoff:
    turnofffactor = configactual.getfloat('turnofffactor')
    print("TURNOFF: mass accretion rate decreased by "+str(turnofffactor))
else:
    turnofffactor =  1. # no mdot reduction

# derived quantities:
r_e = configactual.getfloat('r_e_coeff') * (mu30**2/mdot)**(2./7.)*m1**(-10./7.) * xifac # magnetosphere radius
dr_e = configactual.getfloat('drrat') * r_e
omega = configactual.getfloat('omegafactor')*r_e**(-1.5)
if verbose:
    print("r_e = "+str(r_e/rstar))
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
tr = afac * dr_e/r_e / xifac * r_e**2.5 / rstar # replenishment time scale of the column
if verbose:
    print("r_e = "+str(r_e))
    print(conf+": replenishment time "+str(tr*tscale))
    #   ii =input("R")
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

# eto vs0 priskazka
    
ttest = False # set True to output topology test (which domains are connected to which)

# loading local modules:
if ifplot:
    import plots
if ifhdf:
    import hdfoutput as hdf
import bassun as bs # Basko-Sunyaev solution 
import solvers as solv # Riemann solvers
from sigvel import * # signal velocities
from geometry import * #
from tauexp import *

from timer import Timer

# beta = Pgas / Ptot: define once and globally
from beta import *
betafun = betafun_define() # defines the interpolated function for beta (\rho, U)
betafun_p = betafun_press_define() # defines the interpolated function for beta (\rho, P)

from timestep import * 

def regularize(u, rho, press):
    '''
    if internal energy goes below ufloor, we heat the matter up artificially
    '''
    #    u1=u-ufloor ; rho1=rho-rhofloor ; press1 = press-ufloor
    return (u+ufloor+fabs(u-ufloor))/2., (rho+rhofloor+fabs(rho-rhofloor))/2., (press+ufloor +fabs(press-ufloor))/2.    

##############################################################################

# conversion between conserved and primitive variables for separate arrays and for a single domain

def toprim_separate(m, s, e, g):
    '''
    conversion to primitives, given mass (m), momentum (s), and energy (e) densities as arrays; g is geometry structure
    outputs: density, velocity, internal energy density, urad (radiation internal energy density), beta (=pgas/p), pressure
    '''
    rho=m/g.across
    v=s/m
    u=(e-m*v**2/2.)/g.across
    #    umin = u.min()
    beta = betafun(Fbeta(rho, u, betacoeff))
    press = u/3./(1.-beta/2.)
    u, rho, press = regularize(u, rho, press)
    # after regularization, we need to update beta
    beta = betafun(Fbeta(rho, u, betacoeff))
    return rho, v, u, u*(1.-beta)/(1.-beta/2.), beta, press

def tocon_separate(rho, v, u, g):
    '''
    conversion from primitivies (density rho, velcity v, internal energy u) to conserved quantities m, s, e; g is geometry (structure)
    '''
    m=rho*g.across # mass per unit length
    s=m*v          # momentum per unit length
    e=(u+rho*v**2/2.)*g.across  # total energy (thermal + mechanic) per unit length
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
    e=(prim['u']+prim['rho']*prim['v']**2/2.)*gnd.across  # total energy (thermal + mechanic) per unit length
    return {'m': m, 's': s, 'e': e}

def toprim(con, gnd = None):
    '''
    convert conserved quantities to primitives for one domain
    '''
    #  m = con['m'] ; s = con['s'] ; e = con['e'] ; nd = con['N']
    if gnd is None:
        gnd = g
    rho = con['m']/gnd.across
    v = con['s']/con['m']
    u = (con['e']-con['m']*v**2/2.)/gnd.across
    #    umin = u.min()
    beta = betafun(Fbeta(rho, u, betacoeff))
    press = u/3./(1.-beta/2.)
    u, rho, press = regularize(u, rho, press)
    beta = betafun(Fbeta(rho, u, betacoeff)) # not the most efficient
    urad = u*(1.-beta)/(1.-beta/2.)
    prim = {'rho': rho, 'v': v, 'u': u, 'beta': beta, 'urad': urad, 'press': press}
    return prim

def diffuse(rho, urad, v, dl, across):
    '''
    radial energy diffusion;
    calculates energy flux contribution already at the cell boundary
    across should be set at half-steps
    '''
    rtau_exp = tratfac(dl * (rho[1:]+rho[:-1])/2., taumin, taumax)
    
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
    fe = g.across * v * (u + press + v**2/2.*rho) # energy flux without diffusion    
    return s, p, fe

def qloss_separate(rho, v, u, g, gin = False):
    '''
    standalone estimate for flux distribution
    '''
    #    tau = rho * g.delta
    #    tauphi = rho * g.across / g.delta / 2. # optical depth in azimuthal direction
    taueff = copy(rho)*0.
    #   print("size rho = "+str(size(rho)))
    #   print("size g = "+str(size(g.delta)))
    if gin:
        delta = g.delta[1:-1]
        across = g.across[1:-1]
        r = g.r[1:-1]
    else:
        delta = g.delta
        across = g.across
        r = g.r
    if cooltwosides:
        taueff = rho * delta 
    else:
        taueff = rho / (1. / delta + 2. * delta /  across) 
    # taufac = taufun(taueff, taumin, taumax)    # 1.-exp(-tau)
    beta = betafun(Fbeta(rho, u, betacoeff))
    urad = copy(u * (1.-beta)/(1.-beta/2.))
    urad = (urad+fabs(urad))/2.    
    if ifthin:
        taufactor = tratfac(taueff, taumin, taumax) / xirad
    else:
        taufactor = taufun(taueff, taumin, taumax) / (xirad*taueff+1.)
    if cooltwosides:
        qloss = copy(2.*urad*(across/delta) * taufactor)  # diffusion approximation; energy lost from 4 sides
    else:
        qloss = copy(2.*urad*(across/delta+2.*delta) * taufactor)  # diffusion approximation; energy lost from 4 sides

    if cslimit:
        # if u/rho \sim cs^2 << 1/r, 1-exp(...) decreases, and cooling stops
        qloss *= taufun((u/rho)/(csqmin/r), taumin, taumax) 
        #        (1.-exp(-(u+ufloor)/(rho+rhofloor))/(csqmin/r))) 
        
    return qloss

def sources(g, rho, v, u, urad, ltot = 0., forcecheck = False, dmsqueeze = 0., desqueeze = 0.):
        # prim, ltot=0., dmsqueeze = 0., desqueeze = 0., forcecheck = False):
    '''
    computes the RHSs of conservation equations
    mass loss (and associated energy loss) is calculated separately (dmsqueeze)
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
    taueff = copy(rho)*0.
    if cooltwosides:
        taueff[:] = rho *delta 
    else:
        taueff[:] = rho / (1./delta + 2. * delta /  across) 
    # copy(1./(1./tau + 1./tauphi))
    #    taufac = taufun(taueff, taumin, taumax)    # 1.-exp(-tau)
    #    taufac = 1. 
    #    gamefac = tratfac(tau, taumin, taumax)
    #    gamedd = eta * ltot * gamefac
    sinsum = 2.*cth / sqrt(3.*cth**2+1.)  # = sina*cth+cosa*sth = sin(theta+alpha)
    force = copy((-sinsum/r**2*(1.-eta * ltot * tratfac(rho*delta, taumin, taumax))
                  +omega**2*r*sth*cosa)*rho*across) # *taufac
    # gammaforce = sinsum/r**2 * eta * ltot * tratfac(rho*delta, taumin, taumax)
    if(forcecheck):
        network = simps(force/(rho*across), x=g.l)
        return network, (1./r[0]-1./r[-1])
    qloss = qloss_separate(rho, v, u, g, gin = True)    
    # irradheating = heatingeff * eta * mdot *afac / r * sth * sinsum * taufun(taueff, taumin, taumax) !!! need to include irradheating later!
    #    ueq = heatingeff * mdot / g.r**2 * sinsum * urad/(xirad*tau+1.)
    dm = copy(rho*0.-dmsqueeze) # copy(rho*0.-dmsqueeze)
    #  dudt = copy(v*force-qloss) # +irradheating # copy
    ds = copy(force - dmsqueeze * v) # lost mass carries away momentum
    de = copy(force * v - qloss - desqueeze) # lost matter carries away energy (or enthalpy??)
    #    return dm, force, dudt, qloss, ueq
    return dm, ds, de

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
    #
    # adding ghost zones:
    if leftpack is not None:
        #    rholeft, vleft, uleft = leftpack
        rholeft = leftpack['rho'] ; vleft = leftpack['v'] ; uleft = leftpack['u']
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
        rho = concatenate([[rho[0]], rho])
        v = concatenate([[v[0]], v])
        u = concatenate([[u[0]], u])
        urad = concatenate([[urad[0]], urad])
        press = concatenate([[press[0]], press])
        beta = concatenate([[beta[0]], beta])        
    if rightpack is not None:
        rhoright = rightpack['rho'] ; vright = rightpack['v'] ; uright = rightpack['u']
        # rhoright, vright, uright = rightpack
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
        rho = concatenate([rho, [rho[-1]]])
        v = concatenate([v, [v[-1]]])
        u = concatenate([u, [u[-1]]])
        urad = concatenate([urad, [urad[-1]]])
        press = concatenate([press, [press[-1]]])
        beta = concatenate([beta, [beta[-1]]])        
    fm, fs, fe = fluxes(gnd, rho, v, u, press)
    if rightpack is None:
        fm[-1] = -mdot * turnofffactor
    g1 = Gamma1(5./3., beta)
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
    if crank == first:
        fm[0] = 0.
        #        fs[0] = fs[1]
        fe[0] = 0.
    m, s, e = tocon_separate(rho, v, u, gnd) # extended conserved quantities
    fm_half, fs_half, fe_half =  solv.HLLC([fm, fs, fe], [m, s, e], vl, vr, vm)
    if(raddiff):
        #        dl = gnd.l[1:]-gnd.l[:-1]
        #  across = gnd.across
        duls_half, dule_half = diffuse(rho, urad, v, gnd.l[1:]-gnd.l[:-1], gnd.across)
        # radial diffusion suppressed, if transverse optical depth is small:
        delta = (gnd.delta[1:]+gnd.delta[:-1])/2.
        across = (gnd.across[1:]+gnd.across[:-1])/2.
        if cooltwosides:
            taueff = delta  * (rho[1:]+rho[:-1])/2.
        else:
            taueff = (rho[1:]+rho[:-1])/2. / (1./delta + 2. * delta /  across) 
        duls_half *= taufun(taueff, taumin, taumax) 
        dule_half *= taufun(taueff, taumin, taumax) 
        # duls_half *= 1.-exp(-delta * (rho[1:]+rho[:-1])/2.)
        #  dule_half *= 1.-exp(-delta * (rho[1:]+rho[:-1])/2.)
        fs_half += duls_half ; fe_half += dule_half         
    #  sinks and sources:
    if(squeezemode):
        if umagtar is None:
            umagtar = umag * ((1.+3.*gnd.cth**2)/4. * (rstar/gnd.r)**6)[1:-1]
        dmsqueeze = 2. * m[1:-1] * sqrt(g1[1:-1]*maximum((press[1:-1]-umagtar)/rho[1:-1], 0.))/gnd.delta[1:-1]
        if squeezeothersides:
            dmsqueeze += 4. * m[1:-1] * sqrt(g1[1:-1]*maximum((press[1:-1]-umagtar)/rho[1:-1], 0.))/ (gnd.across[1:-1] / gnd.delta[1:-1])
        desqueeze = dmsqueeze * (e[1:-1] + press[1:-1] * gnd.across[1:-1]) / m[1:-1] # (e-u*g.across)/m
    else:
        dmsqueeze = 0.
        desqueeze = 0.
    dm, ds, de = sources(gnd, rho[1:-1], v[1:-1], u[1:-1], urad[1:-1], ltot=ltot, dmsqueeze = dmsqueeze, desqueeze = desqueeze)
    #    ltot=trapz(flux, x=gnd.l[1:-1]) 

    dmt, dst, det = derivo(lhalf, m, s, e, fm_half, fs_half, fe_half, dm, ds, de)

    # con1 = {'N': nd, 'm': dmt, 's': dst, 'e': det} # , 'ltot': ltot}

    return {'m': dmt, 's': dst, 'e': det}

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

def BCsend(leftpack_send, rightpack_send, comm):
    leftpack = None ; rightpack = None
    left = crank-1 ; right = crank+1
    if crank > first:
        comm.send(leftpack_send, dest = left, tag = crank)
    if crank < last:
        comm.send(rightpack_send, dest = right, tag = crank)
    if crank > first:
        leftpack = comm.recv(source = left, tag = left)
    if crank < last:
        rightpack = comm.recv(source = right, tag = right)
    return leftpack, rightpack

def onedomain(g, ghalf, icon, comm, hfile = None, fflux = None, ftot = None, t=0., nout = 0, thetimer = None):
#(g, lcon, ghostleft, ghostright, dtpipe, outpipe, hfile, t = 0., nout = 0):
    '''
    single domain, calculated by a single core
    arguments: geometry, geometry+halfstep, conserved quantities, MPI communicator
    '''
    con = icon.copy()
    con1 = icon.copy()
    
    prim = toprim(con, gnd = g) # primitive from conserved

    ltot = 0. # total luminosity of the flow (required for IRR)
    timectr = 0
    
    # basic topology:
    left = crank - 1 ; right = crank + 1
    
    #    t = 0.
    #    print("rank = "+str(crank))
    #    print("tmax = "+str(tmax))

    gleftbound = geometry_local(g, 0)
    grightbound = geometry_local(g, -1)
    # topology test: tag traces the origin domain
    if ttest and (t<dtout):
        if crank > first:
            comm.send({'data': 'from '+str(crank)+' to '+str(left)}, dest = left, tag = crank)
        if crank < last:
            comm.send({'data': 'from '+str(crank)+' to '+str(right)}, dest = right, tag = crank)
        if crank > first:
            leftdata = comm.recv(source = left, tag = left)
            print("I, "+str(crank)+", received from "+str(left)+": "+leftdata['data'])
        if rang < last:
            rightdata = comm.recv(source = right, tag = right)
            print("I, "+str(crank)+", received from "+str(right)+": "+rightdata['data'])
        print("this was topology test\n")
        # tt = input("t")

    # exchange geometry:
    if crank > first:
        comm.send({'g':gleftbound}, dest = left, tag = crank)
    if crank < last:
        comm.send({'g':grightbound}, dest = right, tag = crank)
    if crank > first:
        gdata = comm.recv(source = left, tag = left)
        gleftbound = gdata['g']
    if crank < last:
        gdata = comm.recv(source = right, tag = right)
        grightbound = gdata['g']
        
    # if there is no exchange, the leftmost geometry just reproduces the leftmost point of the actual mesh
    #    print("g size = "+str(shape(g.l)))
    # extended geometry, with left and right boundaries included
    gext = geometry_add(g, grightbound)
    gext = geometry_add(gleftbound, gext)
    #    dlleft_nd = dlleft[nd] ; dlright_nd = dlright[nd]
    lhalf = (gext.l[1:]+gext.l[:-1])/2.
    dl = (ghalf.l[1:]-ghalf.l[:-1])

    #    umagtar = umag * ((1.+3.*gext.cth**2)/4. * (rstar/gext.r)**6)[1:-1]
    
    #    print("nd = "+str(nd)+": "+str(lhalf))
    #    ii = input('lhfl')
    tstore = t # ; nout = 0
    timectr = 0
    # initial conditions 
    if thetimer is not None:
        thetimer.start("total")
        thetimer.start("io")
    outblock = {'nout': nout, 't': t, 'g': g, 'con': con, 'prim': prim}
    if (crank != first):                
        comm.send(outblock, dest = first, tag = crank)
    else:
        tireouts(hfile, comm, outblock, fflux, ftot, nout = nout)
    nout += 1
    if thetimer is not None:
        thetimer.stop("io")
        
    while(t<(tstore+dtout)):
        if thetimer is not None:
            thetimer.start_comp("BC")
        leftpack_send = {'rho': prim['rho'][0], 'v': prim['v'][0], 'u': prim['u'][0]} # , prim['beta'][0]]
        rightpack_send = {'rho': prim['rho'][-1], 'v': prim['v'][-1], 'u': prim['u'][-1]} #, prim['beta'][-1]]
        leftpack, rightpack = BCsend(leftpack_send, rightpack_send, comm)
        if thetimer is not None:
            thetimer.stop_comp("BC")        
            thetimer.start_comp("dt")
        # time step: all the domains send dt to first, and then the first sends the minimum value back
        if timectr == 0:
            dt = time_step(prim, g, dl, xirad = xirad, raddiff = raddiff, eta = eta, CFL = CFL, Cdiff = Cdiff, Cth = Cth, taumin = taumin, taumax = taumax) # this is local dt
            dt = comm.allreduce(dt, op=MPI.MIN) # calculates one minimal dt
        timectr += 1
        if timectr >= timeskip:
            timectr = 0
        if thetimer is not None:
            thetimer.stop_comp("dt")        
            thetimer.start_comp("RKstep")
        dcon1 = RKstep(gext, lhalf, prim, leftpack, rightpack, umagtar = con['umagtar'], ltot = ltot)
        if thetimer is not None:
            thetimer.stop_comp("RKstep")
            thetimer.start_comp("updateCon")
        con1 = updateCon(con, dcon1, dt/2.)
        # ultimate BC:
        if crank == last:
            con1['s'][-1] = -mdot * turnofffactor
            con1['e'][-1] = umagout*g.across[-1]-vout*mdot/2. # (con1['m'] / 2. /g.r)[-1]
        if crank == first:
            con1['s'][0] = 0.     
        if thetimer is not None:
            thetimer.stop_comp("updateCon")
            thetimer.start_comp("BC")
        prim = toprim(con1, gnd = g)
        leftpack_send = {'rho': prim['rho'][0], 'v': prim['v'][0], 'u': prim['u'][0]} # , prim['beta'][0]]
        rightpack_send = {'rho': prim['rho'][-1], 'v': prim['v'][-1], 'u': prim['u'][-1]} #, prim['beta'][-1]]
        leftpack, rightpack = BCsend(leftpack_send, rightpack_send, comm)
        if thetimer is not None:
            thetimer.stop_comp("BC")
            thetimer.start_comp("RKstep")
        dcon2 = RKstep(gext, lhalf, prim, leftpack, rightpack, umagtar = con['umagtar'], ltot = ltot) # , BCfluxleft, BCfluxright)
        if thetimer is not None:
            thetimer.stop_comp("RKstep")
            thetimer.start_comp("updateCon")
        con2 = updateCon(con, dcon2, dt/2.)     
        if crank == last:
            con2['s'][-1] = -mdot * turnofffactor
            con2['e'][-1] =  umagout*g.across[-1]-vout*mdot/2. # (con2['m'] / 2. /g.r)[-1]
        if crank == first:
            con2['s'][0] = 0.
        if thetimer is not None:
            thetimer.stop_comp("updateCon")
            thetimer.start_comp("BC")
        prim = toprim(con2, gnd = g)
        leftpack_send = {'rho': prim['rho'][0], 'v': prim['v'][0], 'u': prim['u'][0]} # , prim['beta'][0]]
        rightpack_send = {'rho': prim['rho'][-1], 'v': prim['v'][-1], 'u': prim['u'][-1]} #, prim['beta'][-1]]
        leftpack, rightpack = BCsend(leftpack_send, rightpack_send, comm)
        if thetimer is not None:
            thetimer.stop_comp("BC")
            thetimer.start_comp("RKstep")
        dcon3 = RKstep(gext, lhalf, prim, leftpack, rightpack, umagtar = con['umagtar'], ltot = ltot) #, BCfluxleft, BCfluxright)
        if thetimer is not None:
            thetimer.stop_comp("RKstep")
            thetimer.start_comp("updateCon")
        con3 = updateCon(con, dcon3, dt)
        if crank == last:
            con3['s'][-1] = -mdot * turnofffactor 
            con3['e'][-1] =  umagout*g.across[-1]-vout*mdot/2. #(con3['m'] / 2. /g.r)[-1]
        if crank == first:
            con3['s'][0] = 0.
        if thetimer is not None:
            thetimer.stop_comp("updateCon")
            thetimer.start_comp("BC")
        prim = toprim(con3, gnd = g)
        leftpack_send = {'rho': prim['rho'][0], 'v': prim['v'][0], 'u': prim['u'][0]} # , prim['beta'][0]]
        rightpack_send = {'rho': prim['rho'][-1], 'v': prim['v'][-1], 'u': prim['u'][-1]} #, prim['beta'][-1]]
        leftpack, rightpack = BCsend(leftpack_send, rightpack_send, comm)
        if thetimer is not None:
            thetimer.stop_comp("BC")
            thetimer.start_comp("RKstep")
        dcon4 = RKstep(gext, lhalf, prim, leftpack, rightpack, umagtar = con['umagtar'], ltot = ltot) # , BCfluxleft, BCfluxright)
        if thetimer is not None:
            thetimer.stop_comp("RKstep")
            thetimer.start_comp("updateCon")

        con = updateCon(con, [dcon1, dcon2, dcon3, dcon4], [dt/6., dt/3., dt/3., dt/6.])
        
        if crank == last:
            con['s'][-1] = -mdot * turnofffactor 
            con['e'][-1] =  umagout*g.across[-1]-vout*mdot/2. # (con['m'] / 2. /g.r)[-1]
        if crank == first:
            con['s'][0] = 0.
            #        prim = toprim(con, gnd = g)
        if thetimer is not None:
            thetimer.stop_comp("updateCon")
 
        t += dt
        #        print("nd = "+str(nd)+"; t = "+str(t)+"; dt = "+str(dt))
        prim = toprim(con, gnd = g) # primitive from conserved
        if thetimer is not None:
            thetimer.lap("step")
    # sending data:
    if thetimer is not None:
        thetimer.stop("step")
        thetimer.start("io")
    outblock = {'nout': nout, 't': t, 'g': g, 'con': con, 'prim': prim}
    if (crank != first):                
        comm.send(outblock, dest = first, tag = crank)
    else:
        tireouts(hfile, comm, outblock, fflux, ftot, nout = nout)
    if thetimer is not None:
        thetimer.stop("io")
    if (thetimer is not None) & (nout%ascalias == 1):
        thetimer.stats("step")
        thetimer.stats("io")
        thetimer.comp_stats()
        thetimer.start("step") #refresh lap counter (avoids IO profiling)
        thetimer.purge_comps()
    nout += 1

    return nout, t, con
    
##########################################################
def tireouts(hfile, comm, outblock, fflux, ftot, nout = 0):
    '''
    single-core output 
    '''        
    t = outblock['t'] ; g = outblock['g'] ; con = outblock['con'] ; prim = outblock['prim']
    m = con['m'] ; e = con['e'] ; umagtar = con['umagtar']
    rho = prim['rho'] ; v = prim['v'] ; u = prim['u'] ; urad = prim['urad'] ; beta = prim['beta'] ; press = prim['press']
    r = g.r

    if csize > 1:
        for k in arange(csize-1)+1:
            outblock = comm.recv(source = k, tag = k)
            t = outblock['t'] ; g = outblock['g'] ; con = outblock['con'] ; prim = outblock['prim']
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
    # wsort = argsort(r) # restoring the proper order of inputs
    #    r = r[wsort] ;  m = m[wsort] ;  e = e[wsort]
    # rho = rho[wsort] ; v = v[wsort] ; u = u[wsort] ; urad = urad[wsort] ; beta = beta[wsort] ; umagtar = umagtar[wsort]
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
    dt, dt_CFL, dt_thermal, dt_diff = timestepdetails(gglobal, rho, press, u, v, urad,  xirad = xirad, raddiff = raddiff, CFL = CFL, Cdiff = Cdiff, Cth = Cth, taumin = taumin, taumax = taumax)
    print("dt = "+str(dt)+" = "+str(dt_CFL)+"; "+str(dt_thermal)+"; "+str(dt_diff)+"\n")
    fflux.flush() ; ftot.flush()
    if hfile is not None:
        hdf.dump(hfile, nout, t, rho, v, u, qloss)
        hfile.flush()
    if not(ifhdf) or (nout%ascalias == 0):
        if ifplot:
            plots.vplot(gglobal.r, v, sqrt(4./3.*u/rho), name=outdir+'/vtie{:05d}'.format(nout))
            plots.uplot(gglobal.r, u, rho, gglobal.sth, v, name=outdir+'/utie{:05d}'.format(nout), umagtar = umagtar)
            plots.someplots(gglobal.r, [beta, 1.-beta], formatsequence=['r-', 'b-'],
                            name=outdir+'/beta{:05d}'.format(nout), ytitle=r'$\beta$, $1-\beta$', ylog=True)
            plots.someplots(gglobal.r, [qloss*gglobal.r],
                            name=outdir+'/qloss{:05d}'.format(nout),
                            ytitle=r'$\frac{r{\rm d} E}{{\rm d}l {\rm d} t}$', ylog=False,
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
            fstream.write(str(gglobal.r[k]/rstar)+' '+str(rho[k])+' '+str(v[k])+' '+str(u[k]/umagtar[k])+' '+str(qloss[k])+'\n')
        fstream.flush()
        fstream.close()


# eto byli neobxodimye dla skazki volkl0rnye elementy
# a teper pojd0t skazka...

def alltire():
    global gglobal
    global mdot
    global tmax
    ######################### main thread:  #############################
    if crank == 0: 
        # if the output directory does not exist:
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        ################## setting geometry: ########################
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

        iftail = configactual.getboolean('iftail')
        # if we want to make the radial mesh even more non-linear:
        if (iftail):
            luni_store = copy(luni)
            rtail = configactual.getfloat('rtailfactor') * rmax
            lend = luni.max()
            rnew=rfun(luni) # radial coordinates for the  l-equidistant mesh
            luni *= sqrt((1.+exp((rnew/rtail)**2))/2.)
            luni *= lend / luni.max()    
            print(luni-luni_store)
            #     ii = input('r')

        rnew=rfun(luni) # radial coordinates for the  l-equidistant mesh
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

        # Basko-Sunyaev 76 parameters 
        BSgamma = (g.across/g.delta**2)[0]/mdot*rstar
        BSeta = (8./21./sqrt(2.)*umag*3.)**0.25*sqrt(g.delta[0])/(rstar)**0.125
        if verbose:
            print(conf+" BS parameters:")
            print(conf+"   gamma = "+str(BSgamma))
            print(conf+"   eta = "+str(BSeta))
        x1 = 1. ; x2 = 1000. ; nxx=1000
        xtmp=(x2/x1)**(arange(nxx)/double(nxx))*x1
        xs = bs.xis(BSgamma, BSeta)
        print("   xi_s = "+str(xs))
        # input("BS")
        # magnetic field energy density:
        umagtar = umag * (1.+3.*g.cth**2)/4. * (rstar/g.r)**6
        #
    
        ##### initial conditions: ####
        m=zeros(nx) ; s=zeros(nx) ; e=zeros(nx)
        vinit=vout *sqrt(rmax/g.r) # initial velocity
        
        # setting the initial distributions of the primitive variables:
        rho = copy(abs(mdot) / (abs(vout)+abs(vinit)) / g.across*0.5)
        #   rho *= 1. + (g.r/rstar)**2
        # total mass
        mass = trapz(rho*g.across, x=g.l)
        meq = (g.across*umag*rstar**2)[0]
        print('meq = '+str(meq)+"\n")
        # ii = input('M')
        rho *= meq/mass * minitfactor # normalizing to the initial mass
        vinit = vout * sqrt(rmax/g.r) * (g.r-rstar)/(rmax-rstar) # to fit the v=0 condition at the surface of the star
        v = copy(vinit)
        #        print("umagout = "+str(umagout))
        #        ii = input("vout * mdot = "+str(vout*mdot/g.across[-1]))
        press =  (umagout-vout*mdot/2./g.across[-1]) * (g.r[-1]/g.r)**2 * (rho/rho[-1]+1.)/2. * 0.5
        rhonoise = 1.e-3 * random.random_sample(nx) # noise (entropic)
        rho *= (rhonoise+1.)
        beta = betafun_p(Fbeta_press(rho, press, betacoeff))
        u = press * 3. * (1.-beta/2.)
        u, rho, press = regularize(u, rho, press)

        # restart block:
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
        
        ########### end restart block
        if verbose:
            print(conf+": U = "+str((u/umagtar).min())+" to "+str((u/umagtar).max()))
        m, s, e = tocon_separate(rho, v, u, g)
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

        if not(ifrestart):
            t = 0. ; nout = 0 # time = 0 except when we restart
        ## TODO: restart to be added!
    
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
        fflux=open(outdir+'/'+'flux.dat', 'w')
        ftot=open(outdir+'/'+'totals.dat', 'w')

        ifquadratic = configactual.getboolean('ifquadratic')
        if ifquadratic:
            aind = ceil(nx/parallelfactor**2).astype(int) # ceil(6.*nx/parallelfactor/(parallelfactor+1.)/(2.*parallelfactor+1.)).astype(int)
            print("first partition is "+str())
            inds = aind * (arange(parallelfactor-1, dtype = int)+1)**2
            print(inds)
        else:
            inds = parallelfactor
    
        gglobal = g
        l_g = geometry_split(g, inds)
        l_ghalf = geometry_split(ghalf, inds, half = True)

        # data splitting:
        l_m = array_split(m, inds) ; l_e = array_split(e, inds) ; l_s = array_split(s, inds)
        l_con = [{'m': l_m[i], 's': l_s[i], 'e': l_e[i]} for i in range(parallelfactor)] # list of conserved quantities, each item organized as a dictionary
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
        for kpar in arange(parallelfactor-1)+1:
            comm.send(l_g[kpar], dest = kpar, tag = kpar)
            comm.send(l_ghalf[kpar], dest = kpar, tag = kpar+parallelfactor)
            # sending pieces of the global geometry
            #        comm.send(l_prim[kpar], dest = kpar+1, tag =  kpar+parallelfactor*2)
            comm.send(l_con[kpar], dest = kpar, tag =  kpar+parallelfactor*3)
            print("initialization: sent data to core "+str(kpar))
            # sending primitives and conserveds
        #        tireouts(hfile, comm) # collecting output
        g = l_g[0]
        ghalf = l_ghalf[0]
        con = l_con[0]

        timer = Timer(["total", "step", "io"],
                      ["BC", "dt", "RKstep", "updateCon"])
    else:
        g = comm.recv(source = 0, tag = crank)
        ghalf = comm.recv(source = 0, tag = crank+parallelfactor)
        #    l_prim = comm.recv(source = 0, tag = crank-1+parallelfactor*2)
        con = comm.recv(source = 0, tag = crank+parallelfactor*3)
        print("initialization: recieved data by core "+str(crank))
    if not(ifrestart):
        t=0.  ; nout = 0
    else:
        # exchange nout and t
        if crank ==0:
            tpack = {"t": t, "nout": nout}
        else:
            tpack = None
        tpack = comm.bcast(tpack, root = 0)
        t = tpack["t"] ; nout = tpack["nout"]
        tmax += t
    while (t<tmax):
        if crank ==0:
            nout, t, con = onedomain(g, ghalf, con, comm, hfile = hfile, fflux = fflux, ftot = ftot,
                                     t=t, nout = nout, thetimer = timer)
        else:
            nout, t, con = onedomain(g, ghalf, con, comm, t=t, nout = nout)

if (parallelfactor != csize):
    print("wrong number of processes, "+str(parallelfactor)+" != "+str(csize))
    exit(1)

alltire()
