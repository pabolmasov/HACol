from numpy import *

def HLLE(fs, qs, sl, sr):
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
    
    fhalf1=(f1[1:]+f1[:-1])/2.  ;  fhalf2=(f2[1:]+f2[:-1])/2.  ;  fhalf3=(f3[1:]+f3[:-1])/2.

    qstar1_left = (q1[:-1] * sl - f1[:-1]) / (sm - sl) ;   qstar1_right = (q1[1:] * sr - f1[1:]) / (sm - sr)
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
    
    wsuperleft = where((sr<0.))
    wsubleft = where((sr>=0.) & (sm<=0.))
    wsubright = where((sl<=0.) & (sm>=0.))
    wsuperright = where((sl>0.))
        
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
        for k in arange(size(sl)):
            print(str(sl[k])+" < "+str(sm[k])+" < "+str(sr[k]))
        i=input('r')
       
    return fhalf1, fhalf2, fhalf3
