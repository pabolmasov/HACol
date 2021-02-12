from scipy.interpolate import interp1d
from numpy import *

from tauexp import *

# routines calculating the adjustable time step for tire_MPI

def time_step(prim, g, dl, xirad = 1.5, raddiff = True, eta = 0., CFL = 0.5, Cdiff = 0.5, Cth = 0.5, taumin = 0.01, taumax = 100.):
    # time step adjustment:
    # also outputs total luminosity of the fraction of the flow, if global eta>0.
    csqest = 4./3.*prim['press']/prim['rho']
    dt_CFL = CFL * (dl / (sqrt(csqest)+abs(prim['v']))[1:-1]).min()
    taueff = prim['rho']/(1./g.delta + 2.*g.delta/g.across)
    qloss = 2.*prim['urad']/(1.+xirad*taueff)* (g.across/g.delta + 2.*g.delta) # * taufun(taueff, taumin, taumax) # here, we ignore the exponential factor 1-e^-tau: it is important for radiative losses but computationally heavy and unnecessary to estimate the time scale
    #    qloss = 2.*prim['urad']/prim['rho'] / xirad * (g.across/g.delta + 2.*g.delta)**2/g.across
    # approximate qloss

    #    wpos = ((qloss) > 0.) & (prim['rho'] > 0.)
    #    wpos = wpos[1:-1]
    dt_thermal = Cth * abs(((prim['u']*g.across)/qloss)).min()
    
    if(raddiff):
        ctmp = 3.*(prim['rho'][1:-1] * dl +1.) * dl
        dt_diff = Cdiff * min(ctmp[ctmp>0.]*(taueff[1:-1] > .1)) # (dx^2/D)
        # if the optical depth is small, diffusion works wrong anyway
    else:
        dt_diff = dt_CFL * 5. # effectively infinity ;)

    # outputs luminosity if irradiation is included
    if eta>0.:
        ltot = trapz(qloss * taufun(taueff, taumin, taumax), x = g.l)
        return minimum(dt_CFL, minimum(dt_diff, dt_thermal)), ltot
    else:
        return minimum(dt_CFL, minimum(dt_diff, dt_thermal))# 1./(1./dt_CFL + 1./dt_thermal + 1./dt_diff)

def timestepdetails(g, rho, press, u, v, urad,  xirad = 1.5, raddiff = True, CFL = 0.5, Cdiff = 0.5, Cth = 0.5, taumin = 0.01, taumax = 100.):
    dl = g.l[1:]-g.l[:-1]
    rho_half = (rho[1:]+rho[:-1])/2. ; press_half = (press[1:]+press[:-1])/2. ; u_half = (u[1:]+u[:-1])/2. ; v_half = (v[1:]+v[:-1])/2. ; urad_half = (urad[1:]+urad[:-1])/2.
    csqest = 4./3.*press_half/rho_half
    dt_CFL = CFL * (dl / (sqrt(csqest)+abs(v_half))).min()
    taueff = rho/(1./g.delta + 2.*g.delta/g.across)
    qloss = 2.*urad/(1.+xirad*taueff)* (g.across/g.delta + 2.*g.delta) * taufun(taueff, taumin, taumax)
    #    qloss = 2.*urad/rho / xirad * (g.across/g.delta + 2.*g.delta)**2/g.across
    wpos = ((qloss) > 0.) & (rho > 0.)
    dt_thermal = Cth * abs((u*g.across)[wpos]/qloss[wpos]).min()
    if(raddiff):
        ctmp = dl**2 * 3.*rho_half
        dt_diff = Cdiff * min(ctmp[ctmp>0.]) # (dx^2/D)
    else:
        dt_diff = dt_CFL * 5. # effectively infinity ;)
    return minimum(dt_CFL, minimum(dt_diff, dt_thermal)), dt_CFL, dt_thermal, dt_diff
