from numpy import *

lmcor = True
malim = 1.0 # low-Mach limit

def sigvel_mean(v, cs):
    '''
    signal velocities used by Davis 1988 ``Simplified second-order Godunov-type methods''
    '''
    vstar = (v[1:]+v[:-1])/2.
    astar = (cs[:-1] + cs[1:])/2.
    vl = minimum((v-cs)[:-1], vstar-astar)
    vr = maximum((v+cs)[1:], vstar+astar)
    return vl, vstar, vr

def sigvel_roe(v, cs, rho):
    '''
    Einfeldt 1988 ``On Godunov-type methods for gas dynamics''

    '''
    csq = cs**2
    vstar = ((v*sqrt(rho))[1:]+(v*sqrt(rho))[:-1])/(sqrt(rho)[1:]+sqrt(rho)[:-1])
    astar = sqrt(((csq*sqrt(rho))[:-1] + (csq*sqrt(rho))[1:])/(sqrt(rho)[1:]+sqrt(rho)[:-1])+\
                 sqrt(rho[1:]*rho[:-1])/(sqrt(rho[1:])+sqrt(rho[:-1]))*(v[1:]-v[:-1])**2/2.)
    vl = minimum((v-cs)[:-1], vstar-astar)
    vr = maximum((v+cs)[1:], vstar+astar)
    return vl, vstar, vr

def sigvel_isentropic(v, cs, g1, csqmin = 0.):
    '''
    isentropic signal velocity estimates (see Toro 1994; section 3.1)
    '''
    vstar = vmean + (cs/(g1-1.))[:-1]-(cs/(g1-1.))[1:] # estimates for radiation-pressure-dominated case
    astar = amean + ((v*(g1-1.))[:-1]-(v*(g1-1.))[1:])/4. # see Toro et al. (1994), eq (10)
    if any(astar <= csqmin):
        w=where(astar <= csqmin)
        astar[w] = csqmin
    #        vr=(v+cs) ; vl=(v-cs)
    vl = minimum((v-cs)[:-1], vstar-astar)
    vr = maximum((v+cs)[1:], vstar+astar)
    return vl, vstar, vr

def sigvel_linearized(v, cs, g1, rho, p):
    '''
    linearized signal velocity estimates (see Toro 1994; section 3.2)
    '''
    rhomean = (rho[1:]+rho[:-1])/2. ; csmean = (cs[1:]+cs[:-1])/2. 
    pstar = (p[1:]+p[:-1])/2. - rhomean * csmean * (v[1:]-v[:-1])/2.
    vstar = (v[1:]+v[:-1])/2. + (p[1:]-p[:-1])/rhomean/csmean/2.
    rhostarleft = rho[:-1] + (v[:-1]-vstar)*rhomean / csmean
    rhostarright = rho[1:] + (vstar-v[1:])*rhomean / csmean
    rhostarleft = maximum(rhostarleft, minimum(rho[1:], rho[:-1]))
    rhostarright = maximum(rhostarright, minimum(rho[1:], rho[:-1]))
    pstar = maximum(pstar, minimum(p[1:], p[:-1]))
    astarleft = sqrt(g1[:-1]*pstar/rhostarleft) ; astarright = sqrt(g1[1:]*pstar/rhostarright)
    vl = minimum((v-cs)[:-1], vstar-astarleft)
    vr = maximum((v+cs)[1:], vstar+astarright)
    return vl, vstar, vr

def sigvel_hybrid(v, cs, g1, rho, p, pmode = 'acoustic'):
    '''
    hybrid estimates by Toro (1994)
    pmode -- mode for pstar. Could be: 'acoustic' (default; Toro 1991), 'TR' (two rarefactions), 'TS' (two shocks) 
    '''
    if pmode == 'TR':
        z = (g1-1.)/2./g1
        pstar = ((cs[:-1]+cs[1:]-(g1-1.)/2.*(v[1:]-v[:-1]))/((cs/p**z)[:-1]+(cs/p**z)[1:]))**(1./z)
    else:
        if pmode == 'TS':
            gl = sqrt(2./(g1+1.)/rho[:-1]/(p[:-1]+(g1-1.)/(g1+1.)*pstar))
            gr = sqrt(2./(g1+1.)/rho[1:]/(p[1:]+(g1-1.)/(g1+1.)*pstar))
            pstar = (gl*p[:-1]+gr*p[1:]-(v[1:]-v[:-1]))/(gl+gr)
        else:
            rhomean = (rho[1:]+rho[:-1])/2. ; csmean = (cs[1:]+cs[:-1])/2.
            pstar = (p[1:]+p[:-1])/2. - rhomean * csmean * (v[1:]-v[:-1])/2.
        pstar = maximum(pstar, 0.)
    
    hsleft = pstar/p[:-1] ; hsright = pstar / p[1:]
    qleft = sqrt(1.+(g1+1.)/2./g1*maximum(hsleft-1.,0.))
    qright = sqrt(1.+(g1+1.)/2./g1*maximum(hsright-1.,0.))
        
    if any(qleft) < 0.:
        print("qleft = "+str(qleft[where(qleft<0.)]))
    if any(qright) < 0.:
        print("qright = "+str(qright[where(qright<0.)]))
    vl = v[:-1]-cs[:-1]*qleft  ; vr = v[1:]+cs[1:]*qright
    # vstar = (v[:-1]+v[1:])/2. - (p[1:]-p[:-1])/2./rhomean/csmean
    vstar = ((rho*cs*v)[:-1]*qleft+(rho*cs*v)[:-1]*qright + p[:-1]-p[1:])/((rho*cs)[:-1]*qleft+(rho*cs)[1:]*qright)   # where did this come from? 
    # low-Mach correction
    if lmcor:
        # Miczek's thesis, 2012 (p. 51)
        philm = minimum(1., maximum(fabs((v/cs)[:-1])/malim, fabs((v/cs)[1:]))/malim)
        # philm = sqrt(sin(pi/2.*philm))
        
    # if there are points where vl > vr (there could be in a shock wave), let us swap the velocities 
    winv = where(vl > vr)
    if size(winv) > 0:
        print("sigvel_hybrid: "+str(size(winv))+" inverted points")
    #    a = 2.*((vl-vr)/(qleft*cs[:-1]+cs[1:]*qright))[winv]-1.
    #    vl[winv] -= a * (cs[:-1]*qleft)[winv]
    #    vr[winv] += a * (cs[1:]*qright)[winv]
        v1 = maximum(vl, vr)
        v2 = minimum(vl, vr)
        vl = minimum(v1, v2) ; vr = maximum(v1,v2)     
    vstar = minimum(maximum(vstar, vl*0.9999+vr*0.0001), vr*0.9999+vl*0.0001) # affects the shock front
    
    if lmcor:
        return vl, vstar, vr, philm
    else:
        return vl, vstar, vr, None
    
def sigvel_toro(m, s, e, p, fe, sl, sr, across_half, r_half, sth_half):
    '''
    (recursive)
    making a (spatial-)half-step set of sound velocities
    '''
    m_half = (sr * m[1:] - sl * m[:-1] - (s[1:]-s[:-1]))/(sr-sl)
    s_half = (sr * s[1:] - sl * s[:-1] - (p[1:]-p[:-1]))/(sr-sl)
    e_half = (sr * e[1:] - sl * e[:-1] - (fe[1:]-fe[:-1]))/(sr-sl)
    rho_half, v_half, u_half, urad_half, beta_half, press_half = toprim(m_half, s_half, e_half, across_half, r_half, sth_half)
    if(m_half.min()<mfloor):
        am = m_half.argmin()
        print("rho_half.min = " + str(rho_half.min()))
        print("m_half.min = " + str(m_half.min()))
        print("(m.min = " + str(m.min())+")")
        print("sl = "+str(sl[am]))
        print("sr = "+str(sr[am]))
        print("m = "+str((m[1:])[am]))
        print("m = "+str((m[:-1])[am]))
        print("s = "+str((s[1:])[am]))
        print("s = "+str((s[:-1])[am]))
        print("v = "+str((s[1:]/m[1:])[am]))
        print("v = "+str((s[:-1]/m[:-1])[am]))
        print("u_half = "+str(u_half[am]))     
        print(str(((sr * m[1:] - sl * m[:-1] -  (s[1:]-s[:-1]))/(sr-sl))[am]))        
        exit()
    g1 = Gamma1(5./3., beta_half)
    cs=sqrt(g1*press_half/(rho_half)) # slightly under-estimating the SOS to get stable signal velocities; exact for u<< rho
    vl = minimum(sl, v_half-cs)
    vr = maximum(sr, v_half+cs)
    return vl, v_half, vr
