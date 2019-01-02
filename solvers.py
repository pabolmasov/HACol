from numpy import *

def HLLE(fs, qs, sl, sr, sm):
    '''
    makes a proxy for a half-step flux, HLLE-like
    flux of quantity q, density of quantity q, sound velocity to the left, sound velocity to the right
    '''
    #    sr=1.+vshift[1:] ; sl=-1.+vshift[:-1]
    f1,f2,f3 = fs  ;  q1,q2,q3 = qs
    sl1 = minimum(sl, 0.) ; sr1 = maximum(sr, 0.)
    ds=sr1-sl1 # see Einfeldt et al. 1991 eq. 4.4
    #    print(ds)
    #    i = input("ds")
    # wreg = where((sr[1:]>=0.)&(sl[:-1]<=0.)&(ds>0.))
    wreg = where(ds>0.)
    #    wleft=where(sr[1:]<0.) ; wright=where(sl[:-1]>0.)
    fhalf1=(f1[1:]+f1[:-1])/2.  ;  fhalf2=(f2[1:]+f2[:-1])/2.  ;  fhalf3=(f3[1:]+f3[:-1])/2.
    if(size(wreg)>0):
        fhalf1[wreg] = ((sr1*f1[:-1]-sl1*f1[1:]+sl1*sr1*(q1[1:]-q1[:-1]))/ds)[wreg] # classic HLLE
        fhalf2[wreg] = ((sr1*f2[:-1]-sl1*f2[1:]+sl1*sr1*(q2[1:]-q2[:-1]))/ds)[wreg] # classic HLLE
        fhalf3[wreg] = ((sr1*f3[:-1]-sl1*f3[1:]+sl1*sr1*(q3[1:]-q3[:-1]))/ds)[wreg] # classic HLLE
    return fhalf1, fhalf2, fhalf3

def HLLC(fs, qs, sl, sr, sm):
    '''
    makes a proxy for a half-step flux,
    following the basic framework of Toro et al. 1994
    flux of quantity q, density of quantity q, sound velocity to the left, sound velocity to the right, velocity of the contact discontinuity
    '''
    f1, f2, f3 = fs  ;  q1, q2, q3 = qs
    ds = sr - sl
    #    for k in arange(size(sl)):
    #        print(str(sl[k])+" "+str(sm[k])+" "+str(sr[k])+"\n")
    #    i=input('r')
    
    fhalf1=(f1[1:]+f1[:-1])/2.  ;  fhalf2=(f2[1:]+f2[:-1])/2.  ;  fhalf3=(f3[1:]+f3[:-1])/2.

    qstar1_left = (q1[:-1] * sl - f1[:-1]) / (sl - sm) ;   qstar1_right = (q1[1:] * sr - f1[1:]) / (sr - sm)
    qstar2_left = qstar1_left * sm ; qstar2_right = qstar1_right * sm
    pleft = sm * (sl * q1[:-1] - f1[:-1]) - (q2[:-1]*sl-f2[:-1])
    pright = sm * (sr * q1[1:] - f1[1:]) - (q2[1:]*sr-f2[1:])
    qstar3_left = ( sl * q3[:-1] - f3[:-1] + sm * pleft)/(sl - sm)
    qstar3_right = ( sr * q3[1:] - f3[1:] + sm * pright)/(sr - sm)

    fluxleft1 = f1[:-1] + sl * (qstar1_left-q1[:-1]) # Toro eq 18
    fluxleft2 = f2[:-1] + sl * (qstar2_left-q2[:-1])
    fluxleft3 = f3[:-1] + sl * (qstar3_left-q3[:-1])
    fluxright1 = f1[1:] + sr * (qstar1_right-q1[1:]) # Toro er 19
    fluxright2 = f2[1:] + sr * (qstar2_right-q2[1:])
    fluxright3 = f3[1:] + sr * (qstar3_right-q3[1:])
    
    wsuperleft = where((sr<0.) & (ds >0.))
    wsubleft = where((sr>=0.) & (sm<=0.) & (ds >0.))
    wsubright = where((sl<=0.) & (sm>=0.) & (ds >0.))
    wsuperright = where((sl>0.) & (ds >0.))
        
    if(size(wsubleft)>0):
        fhalf1[wsubleft] = fluxleft1[wsubleft]
        fhalf2[wsubleft] = fluxleft2[wsubleft]
        fhalf3[wsubleft] = fluxleft3[wsubleft]
    if(size(wsubright)>0):
        fhalf1[wsubright] = fluxright1[wsubright]
        fhalf2[wsubright] = fluxright2[wsubright]
        fhalf3[wsubright] = fluxright3[wsubright]
    if(size(wsuperleft)>0): # Toro eq 29
        fhalf1[wsuperleft] = ((sm * ( sr * (q1[1:]-q1[:-1]) + (sr/sl) * f1[:-1] - f1[1:]) +
                              sr * (1. - sm/sl) * fluxleft1)/(sr - sm))[wsuperleft]
        fhalf2[wsuperleft] = ((sm * ( sr * (q2[1:]-q2[:-1]) + (sr/sl) * f2[:-1] - f2[1:]) +
                              sr * (1. - sm/sl) * fluxleft2)/(sr - sm))[wsuperleft]
        fhalf3[wsuperleft] = ((sm * ( sr * (q3[1:]-q3[:-1]) + (sr/sl) * f3[:-1] - f3[1:]) +
                              sr * (1. - sm/sl) * fluxleft3)/(sr - sm))[wsuperleft]
    if(size(wsuperright)>0): # Toro eq 30
        fhalf1[wsuperright] = ((sm * ( sl * (q1[1:]-q1[:-1]) - (sl/sr) * f1[1:] + f1[:-1]) -
                              sl * (1. - sm/sr) * fluxright1)/(sm - sl))[wsuperright]
        fhalf2[wsuperright] = ((sm * ( sl * (q2[1:]-q2[:-1]) - (sl/sr) * f2[1:] + f2[:-1]) -
                              sl * (1. - sm/sr) * fluxright2)/(sm - sl))[wsuperright]
        fhalf3[wsuperright] = ((sm * ( sl * (q3[1:]-q3[:-1]) - (sl/sr) * f3[1:] + f3[:-1]) -
                              sl * (1. - sm/sr) * fluxright3)/(sm - sl))[wsuperright]
    wcool = where(ds<=0.)
    if(size(wcool)>0):
        wcoolright = where((sl>0.) & (ds<=0.))
        wcoolleft = where((sr<0.) & (ds<=0.))
        if(size(wcoolright)>0):
            fhalf1[wcoolright] = (f1[:-1])[wcoolright]
            fhalf2[wcoolright] = (f2[:-1])[wcoolright]
            fhalf3[wcoolright] = (f3[:-1])[wcoolright]
        if(size(wcoolleft)>0):
            fhalf1[wcoolleft] = (f1[1:])[wcoolleft]
            fhalf2[wcoolleft] = (f2[1:])[wcoolleft]
            fhalf3[wcoolleft] = (f3[1:])[wcoolleft]
       
    return fhalf1, fhalf2, fhalf3

##########################################################################
def shocktester():

    x1 = 0.; x2 = 1. ; nx = 10000
    x=arange(nx)/double(nx-1)*(x2-x1)+x1

    gamma = 5./3.
    
    drho = 10. ; x0 = 0.5 ; sx = 0.01
    cent = 0.01
    
    m = x*0.+1.+drho*double(abs(x-x0)<sx)
    v = 0.
    s = m * v
    press = m ** gamma * cent
    e = press / (gamma-1.) + m*v**2/2.
    p = m * v**2 + press
    fe = (e+p + m*v**2/2.) * v 

    m_C = m ; s_C = s ; e_C = e
    m_E = m ; s_E = s ; e_E = e
    
    tmax = 1000. ; t=0.
    dtout = 0.1 ; tstore =0. ;  nout = 0
    
    dx = 1./double(nx)
    dt = dx * 0.5 # CFL multiplier
    dtdx = dt/dx

    while(t< tmax):
        v_C = s_C/m_C
        press_C = m_C**gamma * cent
        p_C = m_C * v_C**2 + press_C
        fe_C = (e_C+p_C + m_C*v_C**2/2.) * v_C
        cs = sqrt(gamma * press_C/m_C)
        sl = (v_C - cs)[:-1] ; sr = (v_C + cs)[1:] ; sm = (v_C[1:]+v_C[:-1])/2.
        s_halfC, p_halfC, fe_halfC = HLLC([s_C, p_C, fe_C], [m_C,s_C,e_C], sl, sr, sm)
        m_C[1:-1] += -(s_halfC[1:]-s_halfC[:-1])*dtdx
        m_C[-1] += (s_halfC[-1]-0.)*dtdx ; m_C[0] += (0.-s_halfC[0])*dtdx
        s_C[1:-1] += -(p_halfC[1:]-p_halfC[:-1])*dtdx
        s_C[-1] = 0. ; s_C[0] = 0.
        e_C[1:-1] += -(fe_halfC[1:]-fe_halfC[:-1])*dtdx
        e_C[-1] = -(0.-fe_halfC[-1])*dtdx ; e_C[0] = -(fe_halfC[0]-0.)*dtdx
        
        v_E = s_E/m_E
        press_E = m_E**gamma * cent
        p_E = m_E * v_E**2 + press_E
        fe_E = (e_E+p_E + m_E*v_E**2/2.) * v_E
        cs = sqrt(gamma * press_E/m_E)
        sl = (v_E - cs)[:-1] ; sr = (v_E + cs)[1:] ; sm = (v_E[1:]+v_E[:-1])/2.
        s_halfE, p_halfE, fe_halfE = HLLE([s_E, p_E, fe_E], [m_E,s_E,e_E], sl, sr, sm)
        m_E[1:-1] += -(s_halfE[1:]-s_halfE[:-1])*dtdx
        m_E[-1] += (s_halfE[-1]-0.)*dtdx ; m_E[0] += (0.-s_halfE[0])*dtdx
        s_E[1:-1] += -(p_halfE[1:]-p_halfE[:-1])*dtdx
        s_E[-1] = 0. ; s_E[0] = 0.
        e_E[1:-1] += -(fe_halfE[1:]-fe_halfE[:-1])*dtdx
        e_E[-1] = -(0.-fe_halfE[-1])*dtdx ; e_E[0] = -(fe_halfE[0]-0.)*dtdx
        
        t += dt
        
        if(t > tstore):
            foutm = open("shocktestm."+str(nout), 'w')
            foutv = open("shocktestv."+str(nout), 'w')
            for k in arange(nx):
                foutm.write(str(x[k])+" "+str(m_C[k])+" "+str(m_E[k])+"\n")
                foutv.write(str(x[k])+" "+str(s_C[k]/m_C[k])+" "+str(s_E[k]/m_E[k])+"\n")
            foutm.flush() ; foutv.flush()
            foutm.close(); foutv.close()
            nout += 1 ; tstore += dtout

            
