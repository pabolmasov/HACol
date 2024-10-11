from numpy import *
from scipy.special import erf

import solvers as solv
from sigvel import *

mfloor = 1e-15

def stepfun(x):
    return (1.+erf(x/sqrt(2.)))/2.

##########################################################################
def shocktester():
    # Sod's shock test (Sod 1978)
    x1 = 0.; x2 = 1. ; nx = 1000
    x=arange(nx)/double(nx-1)*(x2-x1)+x1

    gamma = 5./3.
    
    drho = 0.125 ; x0 = 0.5 ; sx = 1./double(nx)
    dp = 10.
    cent = 0.01
    
    m = (drho-1.)*stepfun((x0-x)/sx)+1.
    v = 0. * (random.rand(nx)-0.5)
    s = m * v
    press = (dp-1.)*stepfun((x0-x)/sx)+1.
    e = press / (gamma-1.) + m*v**2/2.
    p = m * v**2 + press
    fe = (e+press + m*v**2/2.) * v 

    etot = trapezoid(e, x=x)
    mtot = trapezoid(m, x=x)
    
    m_C = copy(m) ; s_C = copy(s) ; e_C = copy(e)
    m_E = copy(m) ; s_E = copy(s) ; e_E = copy(e)
    
    tmax = 10. ; t=0.
    dtout = 0.1 ; tstore =0. ;  nout = 0
    
    dx = 1./double(nx)
    dt = dx * 0.5 # CFL multiplier
    dtdx = dt/dx
    CFL = 0.2
    
    while(t< tmax):
        mpos = (m_C+mfloor + abs(m_C-mfloor))/2.
        v_C = s_C/mpos
        press_C = mpos**gamma * cent
        u_C = press_C / (gamma-1.) 
        p_C = m_C * v_C**2 + press_C
        fe_C = (e_C+press_C) * v_C
        rho_C = m_C
        cs = sqrt(gamma * press_C/mpos)

        # dt from real CFL?
        dt = CFL * dx / cs.max()
        dtdx = dt / dx
        
        sl_C, sm_C, sr_C, philm = sigvel_hybrid(v_C, cs, gamma, rho_C, press_C)
        #  sl_C = -1. ; sr_C = 1. ; sm_C = (v_C[1:]+v_C[:-1])/2.
        s_halfC, p_halfC, fe_halfC = solv.HLLC1([s_C, p_C, fe_C], [m_C, s_C, e_C], sl_C, sr_C, sm_C, mpos, press_C, v_C) # , phi = philm)
        # s_halfC, p_halfC, fe_halfC = solv.HLLC([s_C, p_C, fe_C], [m_C, s_C, e_C], sl_C, sr_C, sm_C, mpos, presspos)
        m_C[1:-1] += -(s_halfC[1:]-s_halfC[:-1])*dtdx
        m_C[-1] += -(0.-s_halfC[-1])*dtdx ; m_C[0] += -(s_halfC[0]-0.)*dtdx
        s_C[1:-1] += -(p_halfC[1:]-p_halfC[:-1])*dtdx
        s_C[-1] = 0. ; s_C[0] = 0.
        e_C[1:-1] += -(fe_halfC[1:]-fe_halfC[:-1])*dtdx
        e_C[-1] += -(0.-fe_halfC[-1])*dtdx ; e_C[0] += -(fe_halfC[0]-0.)*dtdx
        # print(s_halfC)
        # i = input('s')
        
        mpos = (m_E+mfloor + abs(m_E-mfloor))/2.
        v_E = s_E/mpos
        press_E = mpos**gamma * cent
        presspos = (press_E + abs(press_E))/2.
        p_E = m_E * v_E**2 + press_E
        rho_E = m_E
        fe_E = (e_E+press_E) * v_E
        cs = sqrt(gamma * presspos/mpos)
        #        sl_E = (v_E - cs)[:-1] ; sr_E = (v_E + cs)[1:] ; sm_E = (v_E[1:]+v_E[:-1])/2.
        # sl_E, sm_E, sr_E = sigvel_isentropic(v_E, cs, gamma)
        sl_E, sm_E, sr_E, philm_E = sigvel_hybrid(v_E, cs, gamma, rho_E, presspos)
        s_halfE, p_halfE, fe_halfE = solv.HLLE([s_E, p_E, fe_E], [m_E,s_E,e_E], sl_E, sr_E, sm_E) # , phi = philm_E)
        m_E[1:-1] += -(s_halfE[1:]-s_halfE[:-1])*dtdx
        m_E[-1] += -(0.-s_halfE[-1])*dtdx ; m_E[0] += -(s_halfE[0]-0.)*dtdx
        s_E[1:-1] += -(p_halfE[1:]-p_halfE[:-1])*dtdx
        s_E[-1] = 0. ; s_E[0] = 0.
        e_E[1:-1] += -(fe_halfE[1:]-fe_halfE[:-1])*dtdx
        e_E[-1] += -(0.-fe_halfE[-1])*dtdx ; e_E[0] += -(fe_halfE[0]-0.)*dtdx
        
        t += dt
        
        if(t > tstore):
            print("dt = ", dt)
            foutm = open("shocktestm."+str(nout), 'w')
            foutv = open("shocktestv."+str(nout), 'w')
            mpos = (m_E+mfloor + abs(m_E-mfloor))/2.
            v_E = s_E/mpos
            mpos = (m_C+mfloor + abs(m_C-mfloor))/2.
            v_C = s_C/mpos
            for k in arange(nx):
                foutm.write(str(x[k])+" "+str(m_C[k])+" "+str(m_E[k])+"\n")
                foutv.write(str(x[k])+" "+str(v_C[k])+" "+str(v_E[k])+"\n")
            foutm.flush() ; foutv.flush()
            foutm.close(); foutv.close()
            print("v(C) = "+str(v_C.min())+".."+str(v_C.max()))
            print("v(E) = "+str(v_E.min())+".."+str(v_E.max()))
            # print("vR(C) = "+str(sr_C))
            # total mass and energy
            print("DM(total, C)/Minit = "+str(trapezoid(m_C/mtot-1., x=x)))
            print("DM(total, E)/Minit = "+str(trapezoid(m_E/mtot-1., x=x)))
            print("DE(total, C)/Einit = "+str(trapezoid(e_C/etot-1., x=x)))
            print("DE(total, E)/Einit = "+str(trapezoid(e_E/etot-1., x=x)))
            nout += 1 ; tstore += dtout
