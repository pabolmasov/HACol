from numpy import *
# All the global parameters used in the code
# let us assume GM=1, c=1, kappa=1; this implies Ledd=4.*pi

nx=1000 # the actual number of points in use
nx0=nx*50 # first we make a finer mesh for interpolation
logmesh=True
rbasefactor = 0.5 #  
CFL = 0.5 # CFL = 0.5 is still noisy!
Cth = 0.25 # thermal timescale factor
Cdiff = 0.25

# physical parameters:
mu30 = 0.1 # magnetic moment, 1e30 units
m1=1.4
mdot = 10. * 4. * pi
# 6291.12 * 1.734 * 4.*pi /m1 # mass accretion rate 
mdotsink = 0. # mass sink rate at the inner edge
# 1e21g/s --> 6291.12*4.*pi/m1
rstar = 6.8/m1 # GM/c**2 units
# 10km --> 6.77159 for 1Msun
b12 = 2.*mu30*(rstar*m1/6.8)**(-3) # dipolar magnetic field on the pole, 1e12Gs units
mow = 0.6 # molecular weight
betacoeff = 1.788e-5 * (m1)**(-0.25)/mow # coefficient used to calculate gas-to-total pressure ratio

# BC modes:
galyamode = False # if on, limits the internal energy density by MF energy density at the inner boundary
coolNS = False # if on (and galyamode is off), internal energy is constant at the inner boundary
# a test with coolNS converges well, but it is completely unphysical
ufixed = True # if on, fixes the internal energy at the outer rim, otherwise fixes the heat flux
# (setting ufixed = False leads to unpredictable results if v changes at the outer boundary, as the heat flux is (u+p)v)
squeezemode = True # if on, press>umag at the inner boundary leads to mass loss

# radiation transfer treatment:
raddiff = False # if we include radiation diffusion along the field line
xirad = 1. # radiation diffusion scaling
taumin = 1e-4 # minimal optical depth to consider the low-tau limit
taumax = 1e2 # maximal optical depth

mfloor = 1e-15  # crash floor for mass per unit length
rhofloor = 1e-15 # crash floor for density
ufloor = 1e-15 # crash floor for energy density
csqmin = 1e-16
nubulk = 0.5 # bulk viscosity coeff. Coughlin & Begelman (2014) give 8/81, Loeb & Laor (1992) 40/81 -- check which one is correct! Maybe the definition of zeta is different

eta = 0.0 # self-illumination efficiency 
heatingeff = 0.1 # additional heating scaling with mdot
afac = 0.1 # part of the longitudes subtended by the flow
xifac = 0.5 # magnetospheric radius in Alfven units
r_e = 4376.31 * (mu30**2/mdot)**(2./7.)*m1**(-10./7.) * xifac # magnetospheric radius
dr_e = minimum(1.5*mdot/(4.*pi), r_e*0.05) # radial extent of the flow at r_e
print("Alfven = "+str(r_e/xifac / rstar)+"stellar radii")
print("magnetospheric radius r_e = "+str(r_e)+" = "+str(r_e/rstar)+"stellar radii")
print("Delta re = "+str(dr_e))

# conversion to CGS units:
tscale = 4.92594e-06*m1 # GMsun/c**3
rscale = 1.47676e5*m1 # GMsun/c**2
rhoscale = 1.93474e-05/m1 # c**2 / GMsun kappa, for kappa=0.35 (Solar metallicity, complete ionization)
uscale = 1.73886e16/m1 # c**4/GMsun kappa
mdotscale = 1.26492e16*m1 # G Msun / c kappa
lscale = 1.13685e37*m1 # G Msun c / kappa luminosity scale
massscale = 6.23091e10*m1**2 # (GMsun/c**2)**2/kappa
#
tmax = 1000./tscale # maximal time in tscales
dtout = 0.0001/tscale # output time step in tscales
omega = sqrt(0.0)*r_e**(-1.5) # in Keplerian units on the outer rim
print("spin period "+str(2.*pi/omega*tscale)+"s")
umag = b12**2*2.29e6*m1 # magnetic energy density at the surface, for a 1.4Msun accretorvtie00010.png
umagout = 0.5**2*umag*(rstar/r_e)**6 # magnetic field pressure at the outer rim of the disc (1/2 factor from equatorial plane)
vout = -1./sqrt(r_e) / 15.  # initial poloidal velocity at the outer boundary ; set to scale with magnetic pressure. 

# plotting options:
ifplot = True
plotalias = 10 # plot every Nth output step 
ascalias = 10 # make an ascii file every Nth output step

# output options:
ifhdf = True # if we are writing to HDF5 instead of ascii (flux is always outputted as ascii)
outdir = "out/"

# restart options
ifrestart = False
ifhdf_restart = True # if we are restarting from a hdf file (or an ascii snapshot)
restartfile = outdir + 'tireout2.hdf5'
restartn = 2580
restartprefix = outdir+'tireout' # used if we restart from ascii output

# estimating optimal N for a linear grid
print("nopt(lin) = "+str(r_e/dr_e * (r_e/rstar)**2/5))
print("nopt(log) = "+str(rstar/dr_e * (r_e/rstar)**2/5))

# estimated heat flux at the outer boundary:
print("heat coming from the outer BC "+str(-vout * 4.*pi*r_e*dr_e * 3.*umagout))
print("compare to "+str(mdot/rstar))
