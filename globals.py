from numpy import *
# All the global parameters used in the code
# let us assume GM=1, c=1, kappa=1; this implies Ledd=4.*pi

nx=10000 # the actual number of points in use
nx0=nx*20 # first we make a finer mesh for interpolation
logmesh=False

# physical parameters:
mu30 = 130. # magnetic moment, 1e30 units
m1=1.4
mdot = 6291.12 * 1.734 * 4.*pi /m1 # mass accretion rate 
mdotsink = 0. # mass sink rate at the inner edge
# 1e21g/s --> 6291.12*4.*pi/m1
# acc=True # true if we are going to zero the mass and energy fluxes through the outer boundary in actual equations
rstar=6.8/m1 # GM/c**2 units
# 10km --> 6.77159 for 1Msun
b12=2.*mu30*(rstar*m1/6.8)**(-3) # dipolar magnetic field on the pole, 1e12Gs units

# BC modes:
galyamode = False # if on, sets the internal energy density to MF energy density at the inner boundary
coolNS = False # if on (and galyamode is off), internal energy is kept zero at the surface of the NS
ufixed = True # if on, fixes the internal energy at the outer rim, otherwise fixes the heat flux

eta=0.0 # self-illumination efficiency 
mfloor=1e-15  # crash floor for mass per unit length
rhofloor=1e-15 # crash floor for density
ufloor=1e-15 # crash floor for energy density
afac=0.5 # part of the longitudes subtended by the flow
re = 4376.31 * (mu30**2/mdot)**(2./7.)*m1**(-10./7.) # magnetospheric radius
dre=minimum(1.5*mdot/(4.*pi), re*0.5) # radial extent of the flow at re
print("magnetospheric radius re = "+str(re)+" = "+str(re/rstar)+"stellar radii")
print("Delta re = "+str(dre))

# conversion to CGS units:
tscale=4.92594e-06*m1 # GMsun/c**3
rscale=1.47676e5*m1 # GMsun/c**2
rhoscale=1.93474e-05/m1 # c**2 / GMsun kappa, for kappa=0.35 (Solar metallicity, complete ionization)
uscale=1.73886e16/m1 # c**4/GMsun kappa
mdotscale=1.26492e16*m1 # G Msun / c kappa
lscale=1.13685e37*m1 # G Msun c / kappa luminosity scale
massscale=6.23091e10*m1**2 # (GMsun/c**2)**2/kappa
#
tmax=1000./tscale # maximal time in tscales
dtout=0.001/tscale # output time step in tscales
omega=0.*sqrt(0.6)*re**(-1.5) # in Keplerian units on the outer rim
print("spin period "+str(2.*pi/omega*tscale)+"s")
umag=b12**2*2.29e6*m1 # magnetic energy density at the surface, for a 1.4Msun accretor
umagout=0.5**2*umag*(rstar/re)**6 # magnetic field pressure at the outer rim of the disc (1/2 factor from equatorial plane)
vout=-1./sqrt(re) * 1./15. # initial poloidal velocity at the outer boundary ; set to scale with magnetic pressure. 

xirad=0.2 # radiation loss scaling

# plotting options:
ifplot = True
plotalias = 10 # plot every Nth output step 
ascalias = 1 # make an ascii file every Nth output step

# output options:
ifhdf = True # if we are writing to HDF5 instead of ascii (flux is always outputted as ascii)
