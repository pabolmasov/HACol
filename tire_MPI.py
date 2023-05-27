# loading libraries: numpy
import numpy.random
from numpy.random import rand
from numpy import *

# from numba import jit

# libraries:scipy
from scipy.integrate import *
from scipy.interpolate import *

# libraries needed for interaction with system 
import os
import linecache
import os.path
import sys
import configparser as cp
import gc

import re

# parallel support
from mpi4py import MPI
# MPI parameters:
comm = MPI.COMM_WORLD
crank = comm.Get_rank()
csize = comm.Get_size()

################ reading arguments ###########################
if(size(sys.argv)>1):
    print("launched with arguments "+str(', '.join(sys.argv)))
    # new conf file
    conf=sys.argv[1]
    print(conf+" configuration set by the arguments")
else:
    conf='DEFAULT'
    
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
nx0 = configactual.getint('nx0factor') * nx # refinement used to make an accurate geometry structure; used only once
parallelfactor = configactual.getint('parallelfactor') # number of cores/processes
last = parallelfactor-1 ; first = 0

logmesh = configactual.getboolean('logmesh') # logarithmic mesh in l
rbasefactor = configactual.getfloat('rbasefactor') # offset of the mesh (making it even more nonlinear than log)

# numerical parameters:
rsolver = configactual.get('rsolver')
fsplitter = configactual.getboolean('fsplitter')
CFL = configactual.getfloat('CFL') # Courant-Friedrichs-Levy coeff. 
Cth = configactual.getfloat('Cth') # numerical coeff. for radiation losses
Cdiff = configactual.getfloat('Cdiff') # numerical coeff. for diffusion 
CMloss = configactual.getfloat('CMloss') # numerical coeff. for mass-loss scaling 
timeskip = configactual.getint('timeskip') # >1 if we want to update the time step every "timeskip" steps; not recommended
ufloor = configactual.getfloat('ufloor') # minimal possible energy density
rhofloor = configactual.getfloat('rhofloor') # minimal possible mass density
cslimit = configactual.getboolean('cslimit') # if we are going to set a lower limit for temperature (thermal bath)
csqmin = configactual.getfloat('csqmin') # minimal possible speed-of-sound-squared (only if cslimit is on)
potfrac = configactual.getfloat('potfrac') # how do we include potential energy: 0 if all the work is treated as an energy source; 1 if the potential is included in the expression for conserved enegry
szero = configactual.getboolean('szero') # if we set velocity to zero in the 0th cell (not recommended, as )
ttest = configactual.getboolean('ttest') # topology test output

# physics:
mu30 = configactual.getfloat('mu30') # magnetic moment, 10^{30} Gs cm^3 units 
m1 = configactual.getfloat('m1') # NS mass (solar units)
mdot = configactual.getfloat('mdot') * 4. * pi # internal units, GM/varkappa c
mdotsink = configactual.getfloat('mdotsink') * 4. *pi # internal units
rstar = configactual.getfloat('rstar') # GM/c^2 units
b12 = 2.*mu30*(rstar*m1/6.8)**(-3) # dipolar magnetic field on the pole, 1e12Gs units
mow = configactual.getfloat('mow') # mean molecular weight
betacoeff = configactual.getfloat('betacoeff') * (m1)**(-0.25)/mow 

# BC modes:
BSmode = configactual.getboolean('BSmode')
coolNS = configactual.getboolean('coolNS')
ufixed = configactual.getboolean('ufixed')
squeezemode = configactual.getboolean('squeezemode')
venttest = configactual.getboolean('venttest') # turns off mass loss above the surface (only SECOND cell is allowed)
zeroeloss = configactual.getboolean('zeroeloss') # mass is lost without thermal energy (kinetic is lost)

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
ifnuloss = configactual.getboolean('ifnuloss')

if ifnuloss:
    import neu

ifturnoff = configactual.getboolean('ifturnoff') # if mdot is artificially reduced (by a factor turnofffactor)
if ifturnoff:
    turnofffactor = configactual.getfloat('turnofffactor')
    print("TURNOFF: mass accretion rate decreased by "+str(turnofffactor))
else:
    turnofffactor =  1. # no mdot reduction

nocool = configactual.getboolean('nocool') # turning off radiation losses

# derived quantities:
r_e = configactual.getfloat('r_e_coeff') * (mu30**2/mdot)**(2./7.)*m1**(-10./7.) * xifac # magnetosphere radius
dr_e = configactual.getfloat('drrat') * r_e
omega = configactual.getfloat('omegafactor')*r_e**(-1.5)
if verbose:
    print("r_e = "+str(r_e/rstar))
    print(conf+": "+str(omega))

vout = configactual.getfloat('voutfactor')  /sqrt(r_e) # velocity at the outer boundary
minitfactor = configactual.getfloat('minitfactor') # initial total mass in the units of equilibrium mass
umag = b12**2*2.29e6*m1 # on the pole!
umagout = 0.5**2*umag*(rstar/r_e)**6 # magnetic energy density at R_e (edge of the magnetosphere)
csqout = vout**2
if verbose:
    print(conf+": "+str(csqout))
# additional global parameters set here:
config.set(conf,'r_e', str(r_e))
config.set(conf,'dr_e', str(dr_e))
config.set(conf,'umag', str(umag))
config.set(conf,'omega', str(omega))
config.set(conf,'vout', str(vout))

# physical scales (adding mass dependence):
tscale = configactual.getfloat('tscale') * m1
rscale = configactual.getfloat('rscale') * m1
rhoscale = configactual.getfloat('rhoscale') / m1
massscale = configactual.getfloat('massscale') * m1**2
energyscale = configactual.getfloat('energyscale') * m1**2

# sinusoidal variations of \dot{M} with time:
ifmdotvar = configactual.getboolean('ifmdotvar') # sinusoidal variations of mdot
if ifmdotvar:
    mdotvar_amp = configactual.getfloat('mdotvar_amp') # amplitude of these variations
    mdotvar_period = configactual.getfloat('mdotvar_period') #
    ovar = 2.*pi * tscale / mdotvar_period # corresponding frequency
else:
    mdotvar_amp = 0.
    ovar = 0.

if verbose & (omega>0.):
    print(conf+": spin period "+str(2.*pi/omega*tscale)+"s")
# replenishment time:
tr = sqrt(2.)/pi / xifac**3.5 * r_e**1.5 * (dr_e/rstar)
# 4. * afac / sqrt(2.) / xifac**3.5 * r_e**1.5 * (dr_e/rstar)
# 2.**1.5/pi*afac * dr_e/r_e / xifac * (r_e/xifac/rstar)**2.5 / rstar # replenishment time scale of the column
if verbose:
    print("r_e = "+str(r_e))
    print(conf+": replenishment time "+str(tr*tscale)) #*2.*pi*rstar**1.5))
    ii =input("R")
# scaling maximal time with tr:
tmax = tr * configactual.getfloat('tmax')
# scaling output frequency with tr:
dtout = tr * configactual.getfloat('dtout')    # tr * configactual.getfloat('dtout')
# if we are planning to do png outputs during the calculation:
ifplot = configactual.getboolean('ifplot')
# if data are written in HDF5 format
ifhdf = configactual.getboolean('ifhdf')
# every n-th snapshot will be plotted, here is the 'n':
plotalias = configactual.getint('plotalias')
# every n-th snapshot will be outputted to ASCII, here is the 'n':
ascalias = configactual.getint('ascalias')
# output directory
outdir = configactual.get('outdir')
# if we are restarting (then, a lot of other keywords should be set):
ifrestart = configactual.getboolean('ifrestart')
    
if verbose:
    print(conf+": Alfven = "+str(r_e/xifac / rstar)+"stellar radii")
    print(conf+": magnetospheric radius r_e = "+str(r_e)+" = "+str(r_e/rstar)+"stellar radii")
    # estimating optimal N for a linear grid
    print(conf+": nopt(lin) = "+str(r_e/dr_e * (r_e/rstar)**2/5))
    print(conf+": nopt(log) = "+str(rstar/dr_e * (r_e/rstar)**2/5))

# eto vs0 priskazka

# loading local modules:
if ifplot:
    import plots
if ifhdf:
    import hdfoutput as hdf
import bassun as bs # Basko-Sunyaev solution 
import solvers as solv # Riemann solvers
from sigvel import * # signal velocities
from geometry import * # geometry
from tauexp import * # optical depth treatment

from timer import Timer # Joonas's timer

# beta = Pgas / Ptot: define the EOS once and globally
from beta import *
betafun = betafun_define() # defines the interpolated function for beta (\rho, U)
betafun_p = betafun_press_define() # defines the interpolated function for beta (\rho, P)

from timestep import * # controlling the time step (CFL, diffusive, thermal)

def gphi(g, dr = 0.):
    # gravitational potential
    # dr0 = (g.r[1]-g.r[0])/2. * 0.
    if crank == first:
        r0 = rstar - (g.r[1]-g.r[0])/2.
        # dr0 = (g.r[1]-g.r[0])
        r = fabs(g.r+dr-r0)+r0  # ((g.r-dr-r0)**4+dr0**4)**0.25+r0
    else:
        r = g.r+dr
    # r0 + abs(g.r+dr-r0)  # mirroring the potential at half the first cell (between the first cell and the ghost)
    #    if crank == first:
    phi = -1./r - 0.5*(r*g.sth*omega)**2 
    return phi

def gforce(sinsum, g, dr):
    # gravitational force calculated self-consistently using gphi on cell boundaries
    #    if crank == first:
    phi = gphi(g, dr/2.)  # 
    phi1 = gphi(g, -dr/2.)  #
    return sinsum * (( phi1 - phi ) / dr)[1:-1] # subefficient!
# ( phi[1:-1] - phi[2:] ) / dr[1:-1] /2.
    
def regularize(u, rho, press):
    '''
    if internal energy goes below ufloor, we heat the matter up artificially
    '''    
    if (u.min() < ufloor):
        u1 = (u+ufloor+fabs(u-ufloor))/2.
        press1 = (press+ufloor +fabs(press-ufloor))/2.
    else:
        u1 = u
        press1 = press
    if (rho.min() < rhofloor):
        rho1 = (rho+rhofloor+fabs(rho-rhofloor))/2.
    else:
        rho1 = rho
            
    return u1, rho1, press1

##############################################################################

# conversion between conserved and primitive variables for separate arrays and for a single domain

def toprim_separate(m, s, e, g):
    '''
    conversion to primitives, given mass (m), momentum (s), and energy (e) densities as arrays; g is geometry structure
    outputs: density, velocity, internal energy density, urad (radiation internal energy density), beta (=pgas/p), pressure
    '''
    rho=m/g.across
    v=s/m
    phi = gphi(g) # copy(-1./g.r-0.5*(g.r*g.sth*omega)**2)
    u=(e-m*(v**2/2.+phi*potfrac))/g.across
    #    umin = u.min()
    beta = betafun(Fbeta(rho, u, betacoeff))
    press = u/3./(1.-beta/2.)
    u, rho, press = regularize(u, rho, press)
    # after regularization, we need to update beta
    beta = betafun(Fbeta(rho, u, betacoeff))
    return rho, v, u, u*(1.-beta)/(1.-beta/2.), beta, press

def tocon_separate(rho, v, u, g, gin = False):
    '''
    conversion from primitivies (density rho, velcity v, internal energy u) to conserved quantities m, s, e; g is geometry (structure)
    '''
    if gin:
        phi = gphi(g)[1:-1] # copy(-1./g.r-0.5*(g.r*g.sth*omega)**2)
        across = g.across[1:-1]
    else:
        phi = gphi(g)
        across = g.across
    m=rho*across # mass per unit length
    s=m*v          # momentum per unit length
    e=(u+rho*(v**2/2.+phi*potfrac))*across  # total energy (thermal + mechanic) per unit length
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
    phi = gphi(g)  # copy(-1./gnd.r-0.5*(gnd.r*gnd.sth*omega)**2)
    m=prim['rho']*gnd.across # mass per unit length
    s=m*prim['v'] # momentum per unit length
    e=(prim['u']+prim['rho']*(prim['v']**2/2.+phi*potfrac))*gnd.across  # total energy (thermal + mechanic) per unit length
    return {'m': m, 's': s, 'e': e}

def toprim(con, gnd = None):
    '''
    convert conserved quantities to primitives for one domain
    '''
    #  m = con['m'] ; s = con['s'] ; e = con['e'] ; nd = con['N']
    if gnd is None:
        gnd = g
    phi = gphi(gnd) # copy(-1./gnd.r-0.5*(gnd.r*gnd.sth*omega)**2)
    rho = con['m']/gnd.across
    v = con['s']/con['m']
    u = (con['e']-con['m']*(v**2/2.+phi*potfrac))/gnd.across
    #    umin = u.min()
    beta = betafun(Fbeta(rho, u, betacoeff))
    press = u/3./(1.-beta/2.)
    u, rho, press = regularize(u, rho, press)
    beta = betafun(Fbeta(rho, u, betacoeff)) # not the most efficient
    urad = u*(1.-beta)/(1.-beta/2.)
    prim = {'rho': rho, 'v': v, 'u': u, 'beta': beta, 'urad': urad, 'press': press}
    return prim

def diffuse(rho, urad, v, dl, across, taueff):
    '''
    radial energy diffusion;
    calculates energy flux contribution already at the cell boundary
    across should be set at half-steps
    '''
    # rtau_exp =  tratfac(3. * (rho[1:]+rho[:-1])/2. * dl, taumin, taumax)
    rtau_exp = 1./(3.*dl* (rho[1:]+rho[:-1])/2.)
    # rtau_exp1 =  tratfac(3. * (rho)[1:] * dl/2., taumin, taumax)
    # rtau_exp2 =  tratfac(3. * (rho)[:-1] * dl/2., taumin, taumax)
    # rtau_exp1 = 1./(3. * (rho)[1:] * dl/2.)
    # rtau_exp2 = 1./(3. * (rho)[:-1] * dl/2.)

    duls_half =  -nubulk  * (( urad * v)[1:] - ( urad * v)[:-1])\
                 *(across[1:]+across[:-1]) / 2. * rtau_exp #  / (rtau_left + rtau_right)
    # -- photon bulk viscosity
    # dule_half = -((urad)[1:] - (urad)[:-1])\
    #             *(across[1:]+across[:-1]) / 2. * rtau_exp # / (rtau_left + rtau_right)
    
    du = (urad*across/rho)[1:] - (urad*across/rho)[:-1]
    
    # du[1:-1] = (-urad[3:]+8.*urad[2:-1]-8.*urad[1:-2]+urad[:-3])/12.
    
    ttau = taufun(taueff, taumin, taumax)
    dule_half = -du / 3. / dl * ttau
    # (across[1:]+across[:-1]) / 2. /(3.*dl)/ ((rho[1:]+rho[:-1])/2.) # (rtau_exp1 + rtau_exp2)  # * rtau_exp # / (rtau_left + rtau_right)

    # dule_half +=  duls_half * (v[1:]+v[:-1])/2. # adding the viscous energy flux
    # -- radial diffusion

    # flux limiting
    # dule_half = minimum(dule_half, maximum((urad*across/rho)[:-1],(urad*across/rho)[1:])/dl) # radiation going to the right
    dule_half = maximum(dule_half, -maximum((urad*across/rho)[:-1],(urad*across/rho)[1:])/dl) # radiation going to the left
    dule_half = minimum(dule_half, (urad*across)[:-1])
    dule_half = maximum(dule_half, -(urad*across)[1:])
    # smoothing the diffusive flux
    edepth = 0.5
    ddepth = 10
    if edepth > 0.:
        # taul = cumtrapz((rho[1:]+rho[:-1])/2., x=dl, initial = 0.)
        
        for k in arange(ddepth):
            dule_half[1:-1] += edepth * ((dule_half[2:]-dule_half[1:-1]) * exp(-rho[2:-1]/2. * dl[1:-1])+ (dule_half[:-2]-dule_half[1:-1])*exp(-rho[1:-2]/2. * dl[:-2])) * exp(-ttau[1:-1])
        
        '''
        for k in arange(ddepth)+1:
            dule_half[k:-k] += (dule_half[(k+1):-(k-1)]*exp(-taul[])) ((urad*across)[(2*k+1):]-(urad*across)[(k+1):-k])*exp(-abs(taul[k:-k]-taul[(2*k):]))
            dule_half[k:-k] += ((urad*across)[:-(2*k+1)]-(urad*across)[(k+1):-k])*exp(-abs(taul[k:-k]-taul[:-(2*k)]))
        '''

    #n0 = (urad <= 0.).sum()
    #if n0>0:
    ##    print(n0)
    #    print(urad.min())
        # ii = input('N')


    return duls_half, dule_half
           
def fluxes(g, rho, v, u, press):
    '''
    computes the fluxes of conserved quantities, given primitives; 
    radiation diffusion flux is not included, as it is calculated at halfpoints
    inputs:
    rho -- density, v -- velocity, u -- thermal energy density
    g is geometry (structure)
    Note: fluxes do not include diffusion (added separately)
    '''
    phi = gphi(g) # copy(-1./g.r-0.5*(g.r*g.sth*omega)**2)
    s = g.across * (v * rho) # mass flux (identical to momentum per unit length -- can we use it?)
    if not fsplitter:
        p = s * v + press * g.across # momentum flux
    else:
        p = s * v
    fe = g.across * ( (u + press) * v + (v**2/2.+potfrac*phi)*(rho*v)) # energy flux without diffusion    
    return s, p, fe

def qloss_separate(rho, urad, g, gin = False, dt = None):
    '''
    standalone estimate for flux distribution
    '''
    #    tau = rho * g.delta
    #    tauphi = rho * g.across / g.delta / 2. # optical depth in azimuthal direction
    taueff = copy(rho)*0.
    #   print("size rho = "+str(size(rho)))
    #   print("size g = "+str(size(g.delta)))
    if gin: # when we exclude ghost zones
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
    # taueff /= 2. # either we radiate from two sides and use one-half of taueff, or we use the full optical depth and use effectively one side
    # taufac = taufun(taueff, taumin, taumax)    # 1.-exp(-tau)
    # beta = betafun(Fbeta(rho, u, betacoeff))
    # urad = copy(u * (1.-beta)/(1.-beta/2.))
    #  urad = (urad+fabs(urad))/2.    
    if ifthin:
        taufactor = tratfac(taueff, taumin, taumax) / xirad
    else:
        taufactor = taufun(taueff, taumin, taumax) / (xirad*taueff+1.)
        
    if cooltwosides:
        perimeter = 2. * (across/delta)
    else:
        perimeter = 2. * (across/delta+2.*delta)
        
    qloss = copy(urad*perimeter * taufactor)  # diffusion approximation

    '''
    if size(g.l) == size(qloss):
        print("current flux: "+str(trapz(qloss, x = g.l))+'\n')
    else:
        print("current flux: "+str(trapz(qloss, x = g.l[1:-1]))+'\n')
    '''
    
    return qloss

def sources(m, g, rho, v, u, urad, ltot = 0., forcecheck = False, dmsqueeze = 0., desqueeze = 0., dt = None):
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
    ng = size(g.delta) ;    nrho = size(rho)
    if ng > nrho:
        # if the size of geometry variables is larger, it is probably because of the ghost cells included in the geometry arrays
        delta = g.delta[1:-1] ;  across = g.across[1:-1] ; r = g.r[1:-1] ; cth = g.cth[1:-1] ; sth = g.sth[1:-1] ; cosa = g.cosa[1:-1] ; sina = g.sina[1:-1]
        dr = copy(g.r)
        dr[1:-1] = (g.r[2:]-g.r[:-2])/2.
        dr[0] = g.r[1]-g.r[0] ;  dr[-1] = g.r[-1]-g.r[-2]
    else:
        delta = g.delta ;  across = g.across ; r = g.r ; cth = g.cth ; sth = g.sth ; cosa = g.cosa ; sina = g.sina
        dr = copy(r)
        dr[:-1] = r[1:]-r[:-1] ; dr[-1] = dr[-2]
    # taueff = copy(rho)*0.
    # if cooltwosides:
    #     taueff[:] = rho *delta 
    # else:
    #     taueff[:] = rho / (1./delta + 2. * delta /  across) 
    sinsum = 2.*cth / sqrt(3.*cth**2+1.)  # = sina*cth+cosa*sth = sin(theta+alpha)
    # barycentering (does not work out)
    #rc = copy(r)
    #rc[1:-1] = ((m*r)[2:] + 2. * (m*r)[1:-1] + (m*r)[:-2])/m[1:-1]/4.
    #rc[0] = (3. * (m*r)[0] + (m*r)[1])/m[0]/4.
    #rc[-1] = ((m*r)[-2] + 3. * (m*r)[-1])/m[-1]/4.
    
    # force = -sinsum * rho * across / r**2
    # if crank == first:
    #    force[0] = 0.
    force = copy(gforce(sinsum, g, dr)*across*rho) # without irradiation
    # *(1.-eta * ltot * tratfac(rho*delta, taumin, taumax))
                  # +omega**2*r*sth*cosa)*rho*across) # *taufac
    # for k in arange(size(force)):
    #    print(str(force[k])+" = "+str(((-sinsum/r**2*across)*rho)[k]))
    # ii = input("F")
    if eta>0.:
        gammaforce = -copy(force) * eta * ltot * tratfac(rho*delta, taumin, taumax)
        # -copy(gforce(sinsum, g, dr)) * eta * ltot * tratfac(rho*delta, taumin, taumax)
    else:
        gammaforce = 0.
    if(forcecheck):
        network = simps(force/(rho*across), x=g.l)
        return network, (1./r[0]-1./r[-1])
    if not(nocool):
        qloss = qloss_separate(rho, urad, g, gin = True, dt = dt)
    else:
        qloss = 0.
    if squeezemode:
        if dmsqueeze.min() < 0.:
            print("min(dmsq) = "+str(dmsqueeze.min()))
            ii = input("dm")
    dm = copy(rho)*0.-dmsqueeze
    #  dudt = copy(v*force-qloss) # +irradheating # copy
    ds = copy(force+gammaforce - dmsqueeze * v) # lost mass carries away momentum
    de = copy((force*(1.-potfrac)+gammaforce) * v - qloss - desqueeze)  #

    if ifnuloss:
        de -= neu.Qnu(rho, urad, mass1 = m1, separate=False) # neutrino losses according to Beaudet et al. 1967

    return dm, ds, de

def derivo(l_half, s_half, p_half, fe_half, dm, ds, de):
    #, dlleft, dlright,
    #sleft, sright, pleft, pright, feleft, feright):
    '''
    main advance step
    input: l (midpoints), three fluxes (midpoints), three sources
    output: three temporal derivatives later used for the time step
    '''
    # nl=size(dm)
    # dmt=zeros(nl) ; dst=zeros(nl); det=zeros(nl)
    dmt = -(s_half[1:]-s_half[:-1])/(l_half[1:]-l_half[:-1]) + dm
    dst = -(p_half[1:]-p_half[:-1])/(l_half[1:]-l_half[:-1]) + ds
    det = -(fe_half[1:]-fe_half[:-1])/(l_half[1:]-l_half[:-1]) + de
    
    return dmt, dst, det

def RKstep(gnd, lhalf, ahalf, prim, leftpack, rightpack, umagtar = None, ltot = 0., dtq = None, defout = False):
    # BCleft, BCright, 
    # m, s, e, g, ghalf, dl, dlleft, dlright, ltot=0., umagtar = None, momentum_inflow = None, energy_inflow =  None):
    '''
    calculating elementary increments of conserved quantities
    input: geometry, half-step l, primitives (dictionary), data from the left ghost zone, from the right ghost zone
    '''
    #    unpacking primitives:
    rho = prim['rho'] ;  press = prim['press'] ;  v = prim['v'] ; urad = prim['urad'] ; u = prim['u'] ;   beta = prim['beta']
    # ahalf is the cross section vector at the midpoints
    ahalf = concatenate([ahalf, [(gnd.across[-1]+gnd.across[-2])/2.]]) # why is it one step offset?
    ahalf = concatenate([[(gnd.across[0]+gnd.across[1])/2.], ahalf])
    # ahalf = concatenate([ahalf, [gnd.across[-1]]])
    # ahalf = concatenate([[gnd.across[0]], ahalf])

    m, s, e = tocon_separate(rho, v, u, gnd, gin = True) # conserved quantities
    g1 = Gamma1(5./3., beta)
    # sources & sinks:
    if(squeezemode):
        if umagtar is None:
            umagtar = umag * ((1.+3.*gnd.cth**2)/4. * (rstar/gnd.r)**6)[1:-1]
        # step = sstep(press/umagtar-1., 0.001, 10.) * sqrt(g1*umagtar/rho)
        step = sqrt(g1*umagtar/rho*maximum(press/umagtar-1., 0.))
        dmsqueeze = 2. * m * step/gnd.delta[1:-1]
        if squeezeothersides:
            dmsqueeze += 4. * m * step/ (gnd.across[1:-1] / gnd.delta[1:-1])
        if zeroeloss:
            # phi = gphi(gnd, 0.) # copy(-1./gnd.r-0.5*(gnd.r*gnd.sth*omega)**2)        
            desqueeze  = dmsqueeze * e /m
        else:
            desqueeze = dmsqueeze * ((e + press * gnd.across[1:-1]) / m) # (e-u*g.across)/m
        if crank == first:
            dmsqueeze[0] = 0. # no losses from the innermost cell (difficult to fit with the BC)
            desqueeze[0] = 0.
        if venttest: # if we want to suppress mass loss above the NS surface
            if crank == first:
                dmsqueeze[2:]=0.
            else:
                dmsqueese *= 0.
        dmloss = trapz(dmsqueeze, x= gnd.r[1:-1])
    else:
        dmsqueeze = 0.
        desqueeze = 0.
        dmloss = 0.
    #if (dmsqueeze >0.).sum() > 5:        
    #    print("P>Umag in "+str((dmsqueeze >0.).sum())+" points")
    #    ii =input('dm')
    dm, ds, de = sources(m, gnd, rho, v, u, urad, ltot=ltot, dmsqueeze = dmsqueeze, desqueeze = desqueeze, dt = dtq) 

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
        # dv = 1e-3 # dtq * 8. /(gnd.r[1]+gnd.r[0]) /(gnd.r[2]+gnd.r[1])
        # print("dv = "+str(dtq/gnd.r[0]**2))
        # ii = input("L")
        # rho0 = rho[0] ; press0 = press[0] ; u0 = u[0] ; urad0 = urad[0] ; v0 = v[0] ; beta0 = beta[0] #
        # crossfrac = gnd.across[1] / gnd.across[0]
        # crossfrac = 1.
        # rho1 = rho0 * crossfrac
        # u1 = u0 * crossfrac # + rho1 * dr
        # beta1 = betafun(Fbeta(rho1, u1, betacoeff))
        # press1 = u1/3./(1.-beta1/2.)
        # urad1 = u1*(1.-beta1)/(1.-beta1/2.)
        #   betafun_p(Fbeta(rho0, press0, betacoeff))
        # rho1 = rho[1] ; u1 = u[1] ; press1 = press[1] ; urad1 = urad[1] ; beta1 = beta[1]
        rho1 = rho[0] ; u1 = u[0] ; press1 = press[0] ; urad1 = urad[0] ; beta1 = beta[0]     ;   v1 =  -minimum(v[0], 0.)
        rho = concatenate([[rho1], rho])
        v = concatenate([[v1], v]) # inner BC for v
        u = concatenate([[u1], u]) 
        urad = concatenate([[urad1], urad])
            # [3.*(1.-beta0)/(4.-3./2.*beta0)*bernoulli], urad])
        press = concatenate([[press1], press])
            #[bernoulli/(4.-3./2.*beta0)], press])
        beta = concatenate([[beta1], beta])        
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
        # outer BC:
        rhout = -mdot / vout / g.across[-1]
        betaout = betafun(Fbeta(rhout, umagout, betacoeff))
        pressout = umagout/3./(1.-betaout/2.)
        uradout = umagout * (1.-betaout) / (1.-betaout/2.)
        rho = concatenate([rho, [rhout]]) #ahalf[-1]]])   #
        v = concatenate([v, [vout]]) # [minimum(v[-1], 0.)]])
        u = concatenate([u, [umagout]])    #  [u[-1]]])
        urad = concatenate([urad, [uradout]])
        press = concatenate([press, [pressout]])
        beta = concatenate([beta, [betaout]])
        print(crank)
        ii = input("NONE:"+str(crank)) # this should not happen: outer BC is set separately
    # fluxes:
    fm, fs, fe = fluxes(gnd, rho, v, u, press)

    g1 = Gamma1(5./3., beta)
    # g1[:] = 5./3. # stability?
    u, rho, press = regularize(u, rho, press)
    cs = sqrt(g1*press/rho)
    # vl, vm, vr, philm = sigvel_hybrid(v, cs, 4./3., rho, press)
    # # vl, vm, vr = sigvel_roe(v, cs, rho)
    philm = None 
    vl, vm, vr, philm = sigvel_hybrid(v, cs, 5./3., rho, press,
                                      pmode = 'acoustic')
    #if crank == first:
    #    vm[0] = maximum(-vm[1], 0.)
    #    vl[0] = -1.
    #    vr[0] = 1.
    
    if any(vl>vm) or any(vm>vr):
        print("core "+str(crank)+": sigvel (h) = "+str(vl.min())+".."+str(vr.max()))
        print("core "+str(crank)+": dv = "+str((vr-vl).min())+".."+str((vr-vl).max()))
        print("core "+str(crank)+": dv(m) = "+str((vr-vm).min())+".."+str((vm-vl).min()))
        wwrong = where((vl >vm) | (vm>vr))
        nwrong = size(wwrong)
        print(str(nwrong)+" corrupted cell(s)")
        for k in arange(nwrong):
            print("R/R* = "+str((gnd.r[1:])[wwrong[k]]/rstar))
            print("vleft = "+str(vl[wwrong[k]]))
            print("vmed = "+str(vm[wwrong[k]]))
            print("vright = "+str(vr[wwrong[k]]))
            ii = input("K")
            
        #print("rho = "+str((rho[1:])[wwrong]))
        #print("press = "+str((press[1:])[wwrong]))
        #print("vleft = "+str(vl[wwrong]))
        #print("vmed = "+str(vm[wwrong]))
        #print("vright = "+str(vr[wwrong]))
        #print("R = "+str((gnd.r[1:])[wwrong]))
        print("signal velocities crashed -- core "+str(crank))
        # ii=input("cs")
        sys.exit(1) 

    m, s, e = tocon_separate(rho, v, u, gnd) # conserved quantities for the extended mesh
    # print(type(rsolver))
    # ii = input('solver')
    # fm_half, fs_half, fe_half =  solv.HLLC1([fm, fs, fe], [m, s, e], vl, vr, vm, rho, press, v, phi = philm)
    #     fm_half, fs_half, fe_half =  solv.HLLCL([fm, fs, fe], [m, s, e], rho, press, v)
    if 'HLLCL' in rsolver:
        # print('HLLCL')
        fm_half, fs_half, fe_half =  solv.HLLCL([fm, fs, fe], [m, s, e], rho, press, v, gamma = g1)
    else:
        if 'HLLC' in rsolver:
            # print('HLLC')
            fm_half, fs_half, fe_half =solv.HLLC([fm, fs, fe], [m, s, e], vl, vr, vm, rho, press, phi = philm)
            # solv.HLLC1([fm, fs, fe], [m, s, e], vl, vr, vm, rho, press, v, phi = philm)
        else:
            # print('size ahalf = '+str(size(ahalf)))
            # print('size a = '+str(size(gnd.across)))
            # ii = input('a')
            fm_half, fs_half, fe_half =  solv.HLLE([fm, fs, fe], [m, s, e], vl, vr, vm, phi = philm)
    
    if crank == last:
        fm_half[-1] = -mdot # setting mass inflow to mdot (looks safe)
    if(raddiff):
        #        dl = gnd.l[1:]-gnd.l[:-1]
        #  across = gnd.across
        # radial diffusion suppressed, if transverse optical depth is small:
        delta = (gnd.delta[1:]+gnd.delta[:-1])/2.
        across = (gnd.across[1:]+gnd.across[:-1])/2.
        if cooltwosides:
            taueff = delta  * (rho[1:]+rho[:-1])/2.
        else:
            taueff = (rho[1:]+rho[:-1])/2. / (1./delta + 2. * delta /  across)
        duls_half, dule_half = diffuse(rho, urad, v, gnd.l[1:]-gnd.l[:-1], gnd.across, taueff)
        # duls_half *= taufun(taueff, taumin, taumax)
        # dule_half *= taufun(taueff, taumin, taumax)
        if leftpack is None:
            dule_half[0] = 0.
        # duls_half *= 1.-exp(-delta * (rho[1:]+rho[:-1])/2.)
        #  dule_half *= 1.-exp(-delta * (rho[1:]+rho[:-1])/2.)
        fs_half += duls_half ; fe_half += dule_half         

    dmt, dst, det = derivo(lhalf, fm_half, fs_half, fe_half, dm, ds, de)
    # flux splitter: pressure and ram pressure are treated separately
    if fsplitter:
        # press_half = (press[1:]*sqrt(rho[1:])+press[:-1]*sqrt(rho[:-1]))/(sqrt(rho[1:])+sqrt(rho[:-1])) # Roe average
        rhomean = (rho[1:]+rho[:-1])/2.
        # cs  = g1*press/rho
        # g1[:] = 5./3.
        csmean = (sqrt((g1*press/rho)[1:])+sqrt((g1*press/rho)[:-1]))/2.
        rhocmean = (sqrt((g1*press*rho)[1:])+sqrt((g1*press*rho)[:-1]))/2.
        # g1 = 5./3.
        press_half = (press[1:]+press[:-1])/2. - rhomean * csmean * (v[1:]-v[:-1])/2.
        # gl = sqrt(2./(g1+1.)/rho[:-1]/(press[:-1]+(g1-1.)/(g1+1.)*press_half))
        # gr = sqrt(2./(g1+1.)/rho[1:]/(press[1:]+(g1-1.)/(g1+1.)*press_half))
        # press_half = (gl*press[:-1]+gr*press[1:]-(v[1:]-v[:-1]))/(gl+gr)
        # z = (g1-1.)/2./g1
        # press_half = ((cs[:-1]+cs[1:]-(g1-1.)/2.*(v[1:]-v[:-1]))/((cs/press**z)[:-1]+(cs/press**z)[1:]))**(1./z)
        
        press_half = maximum(press_half, minimum(press[1:], press[:-1]))
        dst[:] += gnd.across[1:-1] * (press_half[:-1]-press_half[1:]) / (lhalf[1:]-lhalf[:-1])
    
    if defout:
        # diffusive energy flux (DEF) output
        return {'m': dmt, 's': dst, 'e': det, 'dmloss': dmloss, 'DEF': dule_half}
    else:
        return {'m': dmt, 's': dst, 'e': det, 'dmloss': dmloss}

def updateCon(l, dl, dt, coeffs =  None):
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
        if coeffs is not None:
            for k in range(ndl):
                if k == 0:
                    l1['m'] = dl[k]['m']*coeffs[k]
                    l1['s'] = dl[k]['s']*coeffs[k]
                    l1['e'] = dl[k]['e']*coeffs[k]
                else:
                    l1['m'] += dl[k]['m']*coeffs[k]
                    l1['s'] += dl[k]['s']*coeffs[k]
                    l1['e'] += dl[k]['e']*coeffs[k]
            l1['m'] = l1['m']*dt + l['m'] ;  l1['s'] = l1['s']*dt + l['s']  ;  l1['e'] = l1['e']*dt + l['e']
            return l1
        else:
            for k in range(ndl):
                if k == 0:
                    l1['m'] = dl[k]['m']*dt[k]
                    l1['s'] = dl[k]['s']*dt[k]
                    l1['e'] = dl[k]['e']*dt[k]
                else:
                    l1['m'] += dl[k]['m']*dt[k]
                    l1['s'] += dl[k]['s']*dt[k]
                    l1['e'] += dl[k]['e']*dt[k]
            l1['m'] += l['m'] ;  l1['s'] += l['s']  ;  l1['e'] += l['e']
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

def onedomain(g, ghalf, icon, comm, hfile = None, fflux = None, ftot = None, t=0., nout = 0, thetimer = None, rightpack_save = None, dmlost = 0., ediff = 0., fnu = None):
#(g, lcon, ghostleft, ghostright, dtpipe, outpipe, hfile, t = 0., nout = 0):
    '''
    single domain, calculated by a single core
    arguments: geometry, geometry+halfstep, conserved quantities, MPI communicator
    '''
    con = icon.copy()
    con1 = icon.copy()
    con2 = icon.copy()
    con3 = icon.copy()
    dcon1 = icon.copy()
    dcon2 = icon.copy()
    dcon3 = icon.copy()
    dcon4 = icon.copy()
    
    prim = toprim(con, gnd = g) # primitive from conserved

    # outer BC:
    if (crank == last) & (rightpack_save is None):
        # prim['v'][-1] = vout
        # prim['rho'][-1] = -mdot  / (prim['v'] * g.across)[-1]
        rho0 = -mdot  / vout / g.across[-1]
        p0 = (umagout-0.5*mdot * vout / g.across[-1]) * 0.25
        v0 = vout
        if ifturnoff:
            prim['rho'][-1] *= turnofffactor # reducing the mass flow at the outer limit
        rightpack_save = {'rho': rho0, 'v': v0, 'u': p0*3.}
        if verbose:
            print('setting outer BC\n')
            ## ii = input('OBC')
    
    if ifmdotvar and (crank == last):
        # density variations modify rightpack value of rho; the value is restored in the end
        rho_backup = rightpack_save['rho']
        rightpack_save['rho'] *= 1. + mdotvar_amp * sin(ovar * t)
    
    ltot = 0. # total luminosity of the flow (required for IRR)
    timectr = 0
    
    # basic topology:
    left = crank - 1 ; right = crank + 1
    
    #    t = 0.
    #    print("rank = "+str(crank))
    #    print("tmax = "+str(tmax))

    gleftbound = geometry_local(g, 0)
    grightbound = geometry_local(g, -1)
    if crank == first: # interpolation of the inner ghost zone
        gleftbound.l[0]=g.l[0]-(g.l[1]-g.l[0])
        gleftbound.r[0]=g.r[0] - (g.r[1]-g.r[0]) # energy!
        gleftbound.sth[0]=g.sth[0] 
        gleftbound.across[0] = g.across[0] 
        gleftbound.delta[0] = g.delta[0]
    if crank == last:
        grightbound.l[-1]=g.l[-1]+(g.l[-1]-g.l[-2])
        grightbound.r[-1]=g.r[-1] +  (g.r[-1]-g.r[-2])
        grightbound.sth[-1]=g.sth[-1] + (g.sth[-1]-g.sth[-2])
        grightbound.across[-1] = g.across[-1] + (g.across[-1]-g.across[-2])
        grightbound.delta[-1] = g.delta[-1] + (g.delta[-1]-g.delta[-2])
        

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
    phi = gphi(g) # copy(-1./g.r-0.5*(g.r*g.sth*omega)**2) # gravitational potential (needed by the inner BC)
    
    #    print("nd = "+str(nd)+": "+str(lhalf))
    #    ii = input('lhfl')
    tstore = t # ; nout = 0
    timectr = 0
    # initial conditions 
    if thetimer is not None:
        thetimer.start("total")
        thetimer.start("io")
    outblock = {'nout': nout, 't': t, 'g': g, 'con': con, 'prim': prim, 'dmlost': dmlost, 'ediff': ediff}
    if (crank != first):                
        comm.send(outblock, dest = first, tag = crank)
    else:
        tireouts(hfile, comm, outblock, fflux, ftot, nout = nout, dmlost = dmlost, ediff = ediff, fnu = fnu)
    # nout += 1
    if thetimer is not None:
        thetimer.stop("io")
        
    while(t<(tstore+dtout)):
        if thetimer is not None:
            thetimer.start_comp("BC")
        leftpack_send = {'rho': prim['rho'][0], 'v': prim['v'][0], 'u': prim['u'][0]} # , prim['beta'][0]]
        rightpack_send = {'rho': prim['rho'][-1], 'v': prim['v'][-1], 'u': prim['u'][-1]} #, prim['beta'][-1]]
        leftpack, rightpack = BCsend(leftpack_send, rightpack_send, comm)
        
        if crank == last:
            rightpack = rightpack_save
        
        if thetimer is not None:
            thetimer.stop_comp("BC")        
            thetimer.start_comp("dt")
        # time step: all the domains send dt to first, and then the first sends the minimum value back
        if timectr == 0:
            dt = time_step(prim, g, dl, xirad = xirad, raddiff = raddiff, eta = eta, CFL = CFL, Cdiff = Cdiff, Cth = Cth, taumin = taumin, taumax = taumax, CMloss = CMloss * squeezemode) # this is local dt
            if eta >0.:
                ltot = dt[1]
                dt = dt[0]
            dt = comm.allreduce(dt, op=MPI.MIN) # calculates one minimal dt
        timectr += 1
        if timectr >= timeskip:
            timectr = 0
        if thetimer is not None:
            thetimer.stop_comp("dt")        
            thetimer.start_comp("RKstep")
        dcon1 = RKstep(gext, lhalf, ghalf.across, prim, leftpack, rightpack, umagtar = con['umagtar'], ltot = ltot, defout = raddiff) #, dtq = dt/4.)
        if thetimer is not None:
            thetimer.stop_comp("RKstep")
            thetimer.start_comp("updateCon")
        con1 = updateCon(con, dcon1, dt/2.)
        if (crank == first) & szero:
            # con1['s'][0] = 0.
            con1['s'][0] = 0. # minimum(con1['s'][0]+con1['m'][0]*dt/g.r[0]**2, 0.)
            # con1['e'][0] = con1['e'][1] + (1.-potfrac) * (phi[1] * con1['m'][1]-phi[0] * con1['m'][0]) 
        if thetimer is not None:
            thetimer.stop_comp("updateCon")
            thetimer.start_comp("BC")
        prim = toprim(con1, gnd = g)
        leftpack_send = {'rho': prim['rho'][0], 'v': prim['v'][0], 'u': prim['u'][0]} # , prim['beta'][0]]
        rightpack_send = {'rho': prim['rho'][-1], 'v': prim['v'][-1], 'u': prim['u'][-1]} #, prim['beta'][-1]]
        leftpack, rightpack = BCsend(leftpack_send, rightpack_send, comm)

        if crank == last:
            rightpack = rightpack_save
            # --ensuring fixed physical conditions @ the right boundary

        if thetimer is not None:
            thetimer.stop_comp("BC")
            thetimer.start_comp("RKstep")
        dcon2 = RKstep(gext, lhalf,  ghalf.across, prim, leftpack, rightpack, umagtar = con['umagtar'], ltot = ltot) #, dtq = dt/4.) # , BCfluxleft, BCfluxright)
        if thetimer is not None:
            thetimer.stop_comp("RKstep")
            thetimer.start_comp("updateCon")
        con2 = updateCon(con, dcon2, dt/2.)     
        #  if crank == last:
        #      con2['s'][-1] = -mdot * turnofffactor
        #      con2['e'][-1] = vout*mdot*(1.-potfrac)/2.
            # con2['e'][-1] =  umagout*g.across[-1]-vout*mdot/2.*(1.-potfrac) # (con2['m'] / 2. /g.r)[-1]
        if (crank == first) & szero:
            #  con2['s'][0] = 0.
            con2['s'][0] = 0. # minimum(con2['s'][0]+con2['m'][0]*dt/g.r[0]**2, 0.)
            # con2['e'][0] = con2['e'][1]  + (1.-potfrac) * (phi[1] * con2['m'][1]-phi[0] * con2['m'][0]) 
        if thetimer is not None:
            thetimer.stop_comp("updateCon")
            thetimer.start_comp("BC")
        prim = toprim(con2, gnd = g)
        leftpack_send = {'rho': prim['rho'][0], 'v': prim['v'][0], 'u': prim['u'][0]} # , prim['beta'][0]]
        rightpack_send = {'rho': prim['rho'][-1], 'v': prim['v'][-1], 'u': prim['u'][-1]} #, prim['beta'][-1]]
        leftpack, rightpack = BCsend(leftpack_send, rightpack_send, comm)

        if crank == last:
            rightpack = rightpack_save
            # --ensuring fixed physical conditions @ the right boundary
        if thetimer is not None:
            thetimer.stop_comp("BC")
            thetimer.start_comp("RKstep")
        dcon3 = RKstep(gext, lhalf,  ghalf.across, prim, leftpack, rightpack, umagtar = con['umagtar'], ltot = ltot) # , dtq = dt/2.) #, BCfluxleft, BCfluxright)
        if thetimer is not None:
            thetimer.stop_comp("RKstep")
            thetimer.start_comp("updateCon")
        con3 = updateCon(con, dcon3, dt)
        # if crank == last:
        #    con3['s'][-1] = -mdot * turnofffactor 
        #    con3['e'][-1] = vout*mdot*(1.-potfrac)/2.
            # con3['e'][-1] =  umagout*g.across[-1]-vout*mdot/2.*(1.-potfrac) #(con3['m'] / 2. /g.r)[-1]
        if (crank == first) & szero:
            con3['s'][0] = 0. # minimum(con3['s'][0]+con3['m'][0]*dt/g.r[0]**2, 0.)
            # con3['s'][0] += con3['m'][0]*dt/g.r[0]**2
            #   con3['s'][0] = 0.
            # con3['e'][0] = con3['e'][1] + (1.-potfrac) * (phi[1] * con3['m'][1]-phi[0] * con3['m'][0]) 
        if thetimer is not None:
            thetimer.stop_comp("updateCon")
            thetimer.start_comp("BC")
        prim = toprim(con3, gnd = g)
        leftpack_send = {'rho': prim['rho'][0], 'v': prim['v'][0], 'u': prim['u'][0]} # , prim['beta'][0]]
        rightpack_send = {'rho': prim['rho'][-1], 'v': prim['v'][-1], 'u': prim['u'][-1]} #, prim['beta'][-1]]
        leftpack, rightpack = BCsend(leftpack_send, rightpack_send, comm)
        if crank == last:
            rightpack = rightpack_save

        if thetimer is not None:
            thetimer.stop_comp("BC")
            thetimer.start_comp("RKstep")
        dcon4 = RKstep(gext, lhalf,  ghalf.across, prim, leftpack, rightpack, umagtar = con['umagtar'], ltot = ltot) # , dtq = dt) # , BCfluxleft, BCfluxright)
        if thetimer is not None:
            thetimer.stop_comp("RKstep")
            thetimer.start_comp("updateCon")

        con = updateCon(con, [dcon1, dcon2, dcon3, dcon4], dt,
                        coeffs = [1./6., 1./3., 1./3., 1./6.])
        # con = updateCon(con, dcon2, dt) #!!! temporary
        if squeezemode:
            dmlost += (dcon1['dmloss'] + 2.* dcon2['dmloss'] + 2.*dcon3['dmloss'] + dcon4['dmloss'])/6. * dt
        
        if (crank == first) & szero:
            con['s'][0] = 0. # minimum(con['s'][0]+con['m'][0]*dt/g.r[0]**2, 0.)
            # con['s'][0] += con['m'][0]*dt/g.r[0]**2
            # con['e'][0] = con['e'][1] + (1.-potfrac) * (phi[1] * con['m'][1]-phi[0] * con['m'][0]) 
            #        prim = toprim(con, gnd = g)
        if thetimer is not None:
            thetimer.stop_comp("updateCon")
 
        t += dt
        #        print("nd = "+str(nd)+"; t = "+str(t)+"; dt = "+str(dt))
        prim = toprim(con, gnd = g) # primitive from conserved
#        if cslimit & False:
#            prim['u'] =  maximum(prim['u'], prim['rho']*csqmin)
#            con = tocon(prim, gnd = g)
#            con['umagtar'] = icon['umagtar']
        if thetimer is not None:
            thetimer.lap("step")
    # sending data:
    if thetimer is not None:
        thetimer.stop("step")
        thetimer.start("io")
#    outblock = {'nout': nout, 't': t, 'g': g, 'con': con, 'prim': prim, 'dmlost': dmlost}

#    if (crank != first):                
#        comm.send(outblock, dest = first, tag = crank)
#    else:
#        tireouts(hfile, comm, outblock, fflux, ftot, nout = nout, dmlost = dmlost)
    if thetimer is not None:
        thetimer.stop("io")
    if (thetimer is not None) & (nout%ascalias == 1):
        thetimer.stats("step")
        thetimer.stats("io")
        thetimer.comp_stats()
        thetimer.start("step") #refresh lap counter (avoids IO profiling)
        thetimer.purge_comps()
    nout += 1
    
    if ifmdotvar and (crank == last):
        rightpack_save['rho'] = rho_backup # restoring the unperturbed value of rho
    
    if raddiff:
        return nout, t, con, rightpack_save, dmlost, dcon1['DEF']
    else:
        return nout, t, con, rightpack_save, dmlost, 0.
    
##########################################################
def tireouts(hfile, comm, outblock, fflux, ftot, nout = 0, dmlost = 0., ediff = 0., fnu=None):
    '''
    single-core output 
    '''        
    t = outblock['t'] ; g = outblock['g'] ; con = outblock['con'] ; prim = outblock['prim']
    m = con['m'] ; e = con['e'] ; umagtar = con['umagtar']
    rho = prim['rho'] ; v = prim['v'] ; u = prim['u'] ; urad = prim['urad'] ; beta = prim['beta'] ; press = prim['press']
    r = g.r

    if size(ediff) > 1:
        ediff = (ediff[1:]+ediff[:-1])/2.

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
            if verbose:
                print(str(dmlost)+" += "+str(outblock['dmlost']))
            dmlost += outblock['dmlost']
            if size(ediff) > 1:
                ediff_tmp = outblock['ediff']
                ediff = concatenate([ediff,(ediff_tmp[1:]+ediff_tmp[:-1])/2.])
    # wsort = argsort(r) # restoring the proper order of inputs
    #    r = r[wsort] ;  m = m[wsort] ;  e = e[wsort]
    # rho = rho[wsort] ; v = v[wsort] ; u = u[wsort] ; urad = urad[wsort] ; beta = beta[wsort] ; umagtar = umagtar[wsort]
    qloss = qloss_separate(rho, urad, gglobal)
    ltot = trapz(qloss, x = gglobal.l) 
    mtot = trapz(m, x = gglobal.l)
    etot = trapz(e, x = gglobal.l)
    fflux.write(str(t*tscale)+' '+str(ltot)+'\n')
    print(str(t*tscale)+' '+str(ltot)+'\n')
    #     ii = input("Q")
    ftot.write(str(t*tscale)+' '+str(mtot)+' '+str(etot)+' '+str(dmlost)+' '+str(mdot*t)+' '+str((rho*v*gglobal.across)[-1])+'\n')
    # print("mdot = "+str(mdot))
    print(str(t/tmax)+" calculated\n")
    print(str(t*tscale)+' '+str(ltot)+'\n')
    print(str(t*tscale)+' '+str(mtot)+'\n')
    # ii = input("FT")
    # print("dt = "+str(dt)+'\n')
    dt, dt_CFL, dt_thermal, dt_diff, dt_mloss, mach = timestepdetails(gglobal, rho, press, u, v, urad,  xirad = xirad, raddiff = raddiff, CFL = CFL, Cdiff = Cdiff, Cth = Cth, taumin = taumin, taumax = taumax, CMloss = 0.5)
    qloss = qloss_separate(rho, urad, gglobal, dt=dt)
    if ifnuloss:
        nuloss_A, nuloss_Ph, nuloss_Pl = neu.Qnu(rho, urad, mass1=m1, separate=True)
        Lnu_A =  trapz(nuloss_A * gglobal.across, x= gglobal.l)
        Lnu_Ph =  trapz(nuloss_Ph * gglobal.across, x= gglobal.l)
        Lnu_Pl =  trapz(nuloss_Pl * gglobal.across, x= gglobal.l)
        # neu.Qnu(rho, urad, mass1=m1, separate=True)
        fnu.write(str(t*tscale)+' '+str(Lnu_A)+' '+str(Lnu_Ph)+' '+str(Lnu_Pl)+'\n')
        fnu.flush()
    print("dt = "+str(dt)+" = "+str(dt_CFL)+"; "+str(dt_thermal)+"; "+str(dt_diff)+"; "+str(dt_mloss)+"\n")
    print("characteristic Mach number: "+str(mach))
    fflux.flush() ; ftot.flush()
    if hfile is not None:
        # print("qloss = ", qloss)
        # print("ediff = ", ediff)
        if ifnuloss:
            hdf.dump(hfile, nout, t, rho, v, u, qloss, ediff, nuloss = (nuloss_A, nuloss_Ph, nuloss_Pl))
        else:
            hdf.dump(hfile, nout, t, rho, v, u, qloss, ediff)
        hfile.flush()
    if not(ifhdf) or (nout%ascalias == 0):
        if ifplot:
            plots.vplot(gglobal.r, v, sqrt(4./3.*u/rho), name=outdir+'/vtie{:05d}'.format(nout))
            plots.uplot(gglobal.r, u, rho, gglobal.sth, v, name=outdir+'/utie{:05d}'.format(nout), umagtar = umagtar,
                time = t * tscale)
            #plots.someplots(gglobal.r, [beta, 1.-beta], formatsequence=['r-', 'b-'],
            #                name=outdir+'/beta{:05d}'.format(nout), ytitle=r'$\beta$, $1-\beta$', ylog=True)
            plots.someplots(gglobal.r, [qloss*gglobal.r],
                            name=outdir+'/qloss{:05d}'.format(nout),
                            ytitle=r'$\frac{r{\rm d} E}{{\rm d}l {\rm d} t}$', ylog=False,
                            formatsequence = ['k-', 'r-'])
            plots.someplots(gglobal.r, [-v/sqrt(4./3.*u/rho),v/sqrt(4./3.*u/rho)],
                            name=outdir+'/mach{:05d}'.format(nout),
                            ytitle=r'$\frac{r{\rm d} E}{{\rm d}l {\rm d} t}$', ylog=True,
                            xlog = True, formatsequence = ['k-', 'k:'])
            if size(ediff) > 1:
                # print(size(ediff))
                plots.someplots(gglobal.r, [ediff, v * (u + press + rho * v**2/2.) * gglobal.across],
                                name=outdir+'/ediff{:05d}'.format(nout),
                                ytitle=r'$F_e$', ylog=False, yrange = [ediff.min(), ediff.max()],
                                formatsequence = ['k-', 'r-'])

        # ascii output:
        # print(nout)
        fname=outdir+'/tireout{:05d}'.format(nout)+'.dat'
        if verbose:
            print(" ASCII output to "+fname)
        fstream=open(fname, 'w')
        fstream.write('# t = '+str(t*tscale)+'s\n')
        if size(ediff) > 1:
            fstream.write('# format: r/rstar -- rho -- v -- u/umag -- qloss -- ediff\n')
        else:
            fstream.write('# format: r/rstar -- rho -- v -- u/umag -- qloss \n')
        nx = size(gglobal.r)
        for k in arange(nx):
            if size(ediff) > 1:
                fstream.write(str(gglobal.r[k]/rstar)+' '+str(rho[k])+' '+str(v[k])+' '+str(u[k]/umagtar[k])+' '+str(qloss[k])+' '+str(ediff[k])+'\n')
            else:
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
            print(conf+": t_r = A_\perp u_mag / g / dot{M} = "+str(g.across[0] * umag * rstar**2 / mdot * tscale)+"s")
            ii =input("R")
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
        xs, BSbeta = bs.xis(BSgamma, BSeta, ifbeta = True)
        print("   xi_s = "+str(xs))
        print("   beta_s = "+str(BSbeta))
        cthfun = interp1d(g.r/r[0], g.cth) # we need a function allowing to calculate cos\theta (x)

        print(" t_s = "+str(tscale * rstar**1.5 * m1 * bs.dtint(BSgamma, xs, cthfun)))
        if verbose:
            input("BS")
        # magnetic field energy density:
        umagtar = umag * (1.+3.*g.cth**2)/4. * (rstar/g.r)**6
        #
    
        ##### IC initial conditions: ####
        if not(ifrestart):
            m=zeros(nx) ; s=zeros(nx) ; e=zeros(nx)
            vinit=vout *sqrt(rmax/g.r) # first estimate of initial velocity (not matching the v[0]=0 condition)
            # setting the initial distributions of the primitive variables:
            rho = copy(abs(mdot) / (abs(vout)+abs(vinit)) / g.across)
            #   rho *= 1. + (g.r/rstar)**2
            # total mass
            mass = trapz(rho*g.across, x=g.l)
            meq = (g.across*umag*rstar**2)[0]
            print('meq = '+str(meq)+"\n")
            # ii = input('M')
            rho *= meq/mass * minitfactor # normalizing to the initial mass
            # vinit = vout * sqrt(rmax/g.r) * ((g.r-rstar)/(rmax-rstar)) # to fit the v=0 condition at the surface of the star
            # vinit *= abs(mdot / (rho * vinit * g.across)[-1]) # renormalise for a ~const mdot
            v = copy(vinit)
            #        print("umagout = "+str(umagout))
            #        ii = input("vout * mdot = "+str(vout*mdot/g.across[-1]))
            press =  (umagout-vout*mdot/2./g.across[-1]) * (g.r[-1]/g.r) * (rho/rho[-1]+1.)/2. * 0.25
            # pressure approximately fits the energy density = magnetic energy density
            rhonoize = 1.e-3 * random.random_sample(nx) # noise (entropic)
            rho *= (rhonoize+1.)
            beta = betafun_p(Fbeta_press(rho, press, betacoeff))
            u = press * 3. * (1.-beta/2.)
            u, rho, press = regularize(u, rho, press)

        # restart block:
        # if we want to restart from a stored configuration
        # works so far correctly ONLY if the range of r is identical
        if(ifrestart):
            ifhdf_restart = configactual.getboolean('ifhdf_restart')
            restartn = configactual.getint('restartn')
            nout = restartn
            if(ifhdf_restart):
                # restarting from a HDF5 file
                restartfile = configactual.get('restartfile')
                entryname, t, l1, r1, sth1, rho1, u1, v1, qloss1, glosave, ediff = hdf.read(restartfile, restartn)
                print("restarted from file "+restartfile+", entry "+entryname)
            else:
                # restarting from an ascii output
                restartprefix = configactual.get('restartprefix')
                restartdir = os.path.dirname(restartprefix)
                ascrestartname = restartprefix + hdf.entryname(restartn, ndig=5) + ".dat"
                lines = loadtxt(ascrestartname, comments="#")
                print("restarting from "+ascrestartname)
                # ii = input("R")
                r1 = lines[:,0]
                r1, theta1, alpha1, across1, l1, delta1 = gread(restartdir+"/geo.dat")
                r1 /= rstar
                sth1 = sin(theta1) ; cth1 = cos(theta1)
                r1 = lines[:,0]
                umagtar1 = umag * (1.+3.*cth1**2)/4. * (1./r1)**6
                rho1 = lines[:,1] ; v1 = lines[:,2] ; u1 = lines[:,3] * umagtar1
                # what about t??
                tfile = open(ascrestartname, "r") # linecache.getline(restartfile, 1)
                tline = tfile.readline()
                tfile.close()
                t=double(re.search(r'\d+.\d+', tline).group()) / tscale
                print("restarted from ascii output "+ascrestartname)
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
            else:
                print("restarting with the same resolution")
                rho = copy(rho1) ; v = copy(v1) ; u = copy(u1)
                # r *= rstar
                if verbose:
#                    print(r)
#                    print(r1 * rstar)
                    print("Du_max = "+str(abs(u-u1).max()))
                    print("Dv_max = "+str(abs(v-v1).max()))
                    print("Drho_max = "+str(abs(rho-rho1).max()))
                    #                    ii = input('r')           
            beta = betafun(Fbeta(rho, u, betacoeff))
            press = u / (3.*(1.-beta/2.))
            if ifplot:
                plots.uplot(g.r, u, rho, g.sth, v, name=outdir+'/utie_restart', umagtar = umagtar)
                plots.vplot(g.r, v, sqrt(4./3.*u/rho), name=outdir+'/vtie_restart')
                plots.someplots(g.r, [u/rho**(4./3.)], name=outdir+'/entropy_restart', ytitle=r'$S$', ylog=True)
                plots.someplots(g.r, [beta, 1.-beta], formatsequence=['r-', 'b-'],
                                name=outdir+'/beta_restart', ytitle=r'$\beta$, $1-\beta$', ylog=True)
            print("t = "+str(t*tscale))
            print("nout = "+str(nout))
            #         ii = input("T")
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
    
        rho1, v1, u1, urad1, beta1, press1 = toprim_separate(m, s, e, g) # primitive from conserved
        #    workout, dphi = sources(rho1, v1, u1, g, forcecheck = True) # checking whether the force corresponds to the potential
        if verbose:
            #    print(conf+": potential at the surface = "+str(-workout)+" = "+str(dphi))
            print(str(abs(rho/rho1-1.).max())) 
            print(str(abs(v-v1).max()))
            print(str(abs(u/u1-1.).max())) 
            print(str(abs(press/press1-1.).max())) 
            print("primitive-conserved")
            print(conf+": rhomin = "+str(rho.min())+" = "+str(rho1.min())+" (primitive-conserved and back)")
            print(conf+": umin = "+str(u.min())+" = "+str(u1.min()))
#             ii = input('prim')
        m0=m

        if not(ifrestart):
            t = 0. ; nout = 0 # time = 0 except when we restart
    
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
        if ifnuloss:
            fnu = open(outdir+'/'+'neuloss.dat', 'w')
            fnu.write('# t, s  -- Lnu(A), Ledd/4pi\ -- Lnu(Ph), Ledd/4pi\ -- Lnu(Pl), Ledd/4pi\n')
        else:
            fnu = None
            
        fflux.write("# t, s  --  luminosity, Ledd/4pi\n")
        ftot.write("# t, s -- mass, "+" -- energy -- lost mass -- accreted mass -- current mdot\n")
        ftot.write("#  mass units "+str(massscale)+"g\n")
        ftot.write("#  energy units units "+str(energyscale)+"erg\n")

        ### splitting ###
        inds = parallelfactor
        
        gglobal = g
        l_g = geometry_split(g, inds)
        l_ghalf = geometry_split(ghalf, inds, half = True)

        # data splitting:
        l_m = array_split(m, inds) ; l_e = array_split(e, inds) ; l_s = array_split(s, inds)
        l_con = [{'m': l_m[i], 's': l_s[i], 'e': l_e[i]} for i in range(parallelfactor)] # list of conserved quantities, each item organized as a dictionary
        l_u = array_split(u, inds) ; l_rho = array_split(rho, inds) ; l_v = array_split(v, inds)
        l_umagtar = array_split(umagtar, inds)
        l_press = array_split(press, inds) ;    l_urad = array_split(urad1, inds)
        beta = betafun(Fbeta(rho, u, betacoeff))
        l_beta = array_split(beta1, inds) 

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

    rightpack_save = None

    dmlost = 0.
    ediff = 0.
    
    while (t<tmax):
        if verbose:
            print("t = "+str(t*tscale)+" (crank = "+str(crank)+")")
        if crank ==0:
            nout, t, con, rightpack_save, dmlost1, ediff1 = onedomain(g, ghalf, con, comm, hfile = hfile, fflux = fflux, ftot = ftot, t=t, nout = nout, thetimer = timer, rightpack_save = rightpack_save, dmlost = dmlost, ediff = ediff, fnu = fnu)
        else:
            nout, t, con, rightpack_save, dmlost1, ediff1 = onedomain(g, ghalf, con, comm, t=t, nout = nout, rightpack_save = rightpack_save, dmlost = dmlost, ediff = ediff)
        # print("alltire onedomain "+str(crank)+": mdlost1 = "+str(dmlost1))
        dmlost = dmlost1 ; ediff = ediff1
        if rightpack_save is not None:
            print("vout = "+str(rightpack_save['v']))
if (parallelfactor != csize):
    print("wrong number of processes, "+str(parallelfactor)+" != "+str(csize))
    exit(1)

alltire()
