from numpy import *
# All the global parameters used in the code
# let us assume GM=1, c=1, kappa=1; this implies Ledd=4.*pi

nx=3000 # the actual number of points in use
nx0=nx*20 # first we make a finer mesh for interpolation
logmesh=True

# physical parameters:
b12=260.
m1=1.4
mdot=10597.1*4.*pi/m1 # mass accretion rate
mdotsink = 0. # mass sink rate at the inner edge
# 1e21g/s --> 10597.1*4.*pi
# acc=True # true if we are going to zero the mass and energy fluxes through the outer boundary in actual equations
rstar=6.8/m1 # GM/c**2 units
# 10km --> 6.77159 for 1Msun
galyamode = True # if on, sets the internal energy density to MF energy density at the inner boundary
eta=0.0 # self-illumination efficiency 
mfloor=1e-15  # crash floor for mass per unit length
rhofloor=1e-15 # crash floor for density
ufloor=1e-15 # crash floor for energy density
afac=0.5 # part of the longitudes subtended by the flow
re = 122.4 * ((b12*rstar**3)**2/mdot)**(2./7.)*m1**(2./7.) # magnetospheric radius
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
dtout=0.01/tscale # output time step in tscales

omega=sqrt(0.6)*re**(-1.5) # in Keplerian units on the outer rim
print("spin period "+str(2.*pi/omega*tscale))
umag=b12**2*3.2e6 # magnetic energy density at the surface, for a 1.4Msun accretor
umagout=0.5**2*umag*(rstar/re)**6 # magnetic field pressure at the outer rim of the disc (1/2 factor from equatorial plane)
pmagout = umagout/3.
vout=-.5*umagout*4.*pi*re*dre*afac/mdot # initial poloidal velocity at the outer boundary ; set to scale with magnetic pressure. 

xirad=0.2 # radiation loss scaling

# plotting options:
ifplot = True
plotalias = 10 # plot every Nth output step 
ascalias = 100 # make an ascii file every Nth output step

# output options:
ifhdf = True # if we are writing to HDF5 instead of ascii (flux is always outputted as ascii)
