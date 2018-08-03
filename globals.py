from numpy import *
# All the global parameters used in the code
# let us assume GM=1, c=1, kappa=1; this implies Ledd=4.*pi

nx=3000 # the actual number of points in use
nx0=nx*20 # first we make a finer mesh for interpolation

b12=1.
m1=1.4
mdot=100.*4.*pi # mass accretion rate
rstar=6. # GM/c**2 units
# vout=-0.5/sqrt(re) # initial poloidal velocity at the outer boundary 
eta=0.0 # self-illumination efficiency 
mfloor=1e-25  # crash floor for mass per unit length
rhofloor=1e-25 # crash floor for density
ufloor=1e-25 # crash floor for energy density
afac=0.5 # part of the longitudes subtended by the flow
re = 122. * ((b12*rstar**3)**2/mdot)**(2./7.)*m1**(2./7.) # magnetospheric radius
dre=minimum(1.5*mdot/(4.*pi), re*0.5) # radial extent of the flow at re
print("magnetospheric radius re = "+str(re))
print("Delta re = "+str(dre))
tscale=4.92594e-06*m1
tmax=100./tscale # maximal time in tscales
dtout=100. # output time step in tscales

omega=0.99*re**(-1.5) # in Keplerian units on the outer rim
umag=b12**2*3.2e6 # magnetic energy density at the surface, for a 1.4Msun accretor
pmagout=umag*(rstar/re)**6 # magnetic field pressure at the outer rim of the disc
vout=-0.5*pmagout*4.*pi*re*dre*afac/mdot # initial poloidal velocity at the outer boundary ; set to scale with magnetic pressure. 

xirad=0.25 # radiation loss scaling

# plotting options:
ifplot = True
plotalias = 50 # plot every Nth output step 

# output options:
ifhdf = True # if we are writing to HDF5 instead of ascii (flux is always outputted as ascii)