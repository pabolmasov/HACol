from numpy import *

debug = False

def HLLE(fs, qs, sl, sr, sm, phi = None):
    '''
    makes a proxy for a half-step flux, HLLE-like
    flux of quantity q, density of quantity q, sound velocity to the left, sound velocity to the right
    Sod's test passed!
    '''
    f1,f2,f3 = fs  ;  q1,q2,q3 = qs
    if phi is not None:
        sl *= phi ; sr *= phi
    sl1 = minimum(sl, 0.) ; sr1 = maximum(sr, 0.)
    ds = sr1-sl1 # see Einfeldt et al. 1991 eq. 4.4
    #    print(ds)
    #    i = input("ds")
    # wreg = where((sr[1:]>=0.)&(sl[:-1]<=0.)&(ds>0.))
    wreg = where(ds > 0.)
    w0 = where(ds <= 0.)
    #    wleft=where(sr[1:]<0.) ; wright=where(sl[:-1]>0.)
    fhalf1=copy(f1[1:])  ;  fhalf2=copy(f2[1:])  ;  fhalf3=copy(f3[1:])    
    
    if(size(wreg)>0):
        fhalf1[wreg] = (((sr1*f1[:-1]-sl1*f1[1:])/ds+sl1*sr1*(q1[1:]-q1[:-1])/ds))[wreg] # classic HLLE
        fhalf2[wreg] = (((sr1*f2[:-1]-sl1*f2[1:])/ds+sl1*sr1*(q2[1:]-q2[:-1])/ds))[wreg] # classic HLLE
        fhalf3[wreg] = (((sr1*f3[:-1]-sl1*f3[1:])/ds+sl1*sr1*(q3[1:]-q3[:-1])/ds))[wreg] # classic HLLE
    if size(w0)>0.:
        wpos = where((ds <=0.) & (sm >= 0.))
        wneg = where((ds <=0.) & (sm <= 0.))
        if size(wpos)>0.:
            fhalf1[wpos] = (f1[:-1])[wpos]
            fhalf2[wpos] = (f2[:-1])[wpos]
            fhalf3[wpos] = (f3[:-1])[wpos]
        if size(wneg)>0.:
            fhalf1[wneg] = (f1[1:])[wneg]
            fhalf2[wneg] = (f2[1:])[wneg]
            fhalf3[wneg] = (f3[1:])[wneg]
 
    return fhalf1, fhalf2, fhalf3

def HLLC(fs, qs, sl, sr, sm, rho, press, phi = None):
    '''
    makes a proxy for a half-step flux,
    following the basic framework of Toro et al. 1994
    flux of quantity q, density of quantity q, sound velocity to the left, sound velocity to the right, velocity of the contact discontinuity
    works for Sod
    does not work well for tire
    '''
    f1, f2, f3 = fs  ;  q1, q2, q3 = qs
    ds = sr - sl

    v = 0. # f1/q1
    
    nx=size(q1)
    fhalf1=zeros(nx-1, dtype=double)  ;  fhalf2=zeros(nx-1, dtype=double)  ;  fhalf3=zeros(nx-1, dtype=double)    
    if phi is not None:
        sl = sl * phi
        sr = sr * phi
        sm = sm
    fluxleft1 = sm*(sl*q1[:-1] - f1[:-1])/(sl-sm)
    fluxleft2 = (sm*(sl*q2[:-1] - f2[:-1])+sl*(press[:-1] + rho[:-1]*(sl-v[:-1])*(sm-v[:-1])))/(sl-sm)
    fluxleft3 = (sm*(sl*q3[:-1] - f3[:-1])+sm*sl*(press[:-1] + rho[:-1]*(sl-v[:-1])*(sm-v[:-1])))/(sl-sm)
    fluxright1 = sm*(sr*q1[1:] - f1[1:])/(sr-sm)
    fluxright2 = (sm*(sr*q2[1:] - f2[1:])+sr*(press[1:] + rho[1:]*(sr-v[1:])*(sm-v[1:])))/(sr-sm)
    fluxright3 = (sm*(sr*q3[1:] - f3[1:])+sm*sr*(press[1:] + rho[1:]*(sr-v[1:])*(sm-v[1:])))/(sr-sm)
    
    wsuperleft = where(sr<=0.)
    wsubleft = where((sl<0.) & (sm>0.) & (sr>0.))
    wsubright = where((sl<0.) & (sm<=0.) & (sr>0.))
    wsuperright = where(sl>=0.)

    if(debug):
        print("sr = "+str(sr.min())+" to "+str(sr.max()))
        print("sl = "+str(sl.min())+" to "+str(sl.max()))
        print("sm = "+str(sm.min())+" to "+str(sm.max()))
        print(str(size( wsuperleft))+" + "+str(size( wsubleft))+" + "+str(size( wsubright))+
              " + "+str(size( wsuperright))+" + "+str(size( where(sl>= sr)))+ " =  "+str(nx-1))
        j = input('j')
    if(size(wsubleft)>0):
        fhalf1[wsubleft] = fluxleft1[wsubleft]
        fhalf2[wsubleft] = fluxleft2[wsubleft]
        fhalf3[wsubleft] = fluxleft3[wsubleft]
    if(size(wsubright)>0):
        fhalf1[wsubright] = fluxright1[wsubright]
        fhalf2[wsubright] = fluxright2[wsubright]
        fhalf3[wsubright] = fluxright3[wsubright]
    if(size(wsuperleft)>0): # Toro eq 29
        fhalf1[wsuperleft] = (f1[1:])[wsuperleft]
        fhalf2[wsuperleft] = (f2[1:])[wsuperleft]
        fhalf3[wsuperleft] = (f3[1:])[wsuperleft]       
    if(size(wsuperright)>0): # Toro eq 30
        fhalf1[wsuperright] = (f1[:-1])[wsuperright]
        fhalf2[wsuperright] = (f2[:-1])[wsuperright]
        fhalf3[wsuperright] = (f3[:-1])[wsuperright]        
    wcool = where(ds<=0.)
    if(size(wcool)>0):
        #        print(str(size(wcool))+" cool points")
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

def HLLC1(fs, qs, sl, sr, sm, rho, press, v, phi = None):
    '''
    second version of HLLC, according to Fleischmann et al.(2020)
    apparently, works
    phi is low-Mach correction
    '''
    f1, f2, f3 = fs  ;  q1, q2, q3 = qs
    #    ds = sr - sl
    # v = f1/q1
    
    nx=size(q1)
    #     fhalf1=zeros(nx-1, dtype=double)  ;  fhalf2=zeros(nx-1, dtype=double)  ;  fhalf3=zeros(nx-1, dtype=double)    
    if phi is not None:
        sl1 = sl * phi
        sr1 = sr * phi
        sm1 = sm
    else:
        sl1 = sl
        sr1 = sr
        sm1 = sm

    q1star_left = q1[:-1] * (sl-v[:-1])/(sl-sm)
    q2star_left = q1[:-1] * (sl-v[:-1])/(sl-sm) * sm
    q3star_left = (sl-v[:-1])/(sl-sm) * (q3[:-1] + (sm-v[:-1])*(sm+(press/rho)[:-1]/(sl-v[:-1])) * q1[:-1])
    q1star_right = q1[1:] * (sr-v[1:])/(sr-sm)
    q2star_right = q1[1:] * (sr-v[1:])/(sr-sm) * sm
    q3star_right = (sr-v[1:])/(sr-sm) * (q3[1:] + (sm-v[1:])*(sm+(press/rho)[1:]/(sr-v[1:])) * q1[1:]) 
        
    # compact form from Fleischmann 2021 
    f1half = (1.+sign(sm1))/2. * (f1[:-1]+minimum(sl1, 0.)*(q1star_left-q1[:-1])) \
             +  (1.-sign(sm1))/2. * (f1[1:]+maximum(sr1, 0.)*(q1star_right-q1[1:]))
    f2half = (1.+sign(sm1))/2. * (f2[:-1]+minimum(sl1, 0.)*(q2star_left-q2[:-1])) \
             +  (1.-sign(sm1))/2. * (f2[1:]+maximum(sr1, 0.)*(q2star_right-q2[1:]))
    f3half = (1.+sign(sm1))/2. * (f3[:-1]+minimum(sl1, 0.)*(q3star_left-q3[:-1])) \
             +  (1.-sign(sm1))/2. * (f3[1:]+maximum(sr1, 0.)*(q3star_right-q3[1:]))

    if False:
        gamma = 4./3.
        cs = sqrt(gamma * press/rho)
        rhomean = (rho[1:]+rho[:-1])/2. ; csmean = (cs[1:]+cs[:-1])/2.
        f2half += (1.-phi) * rhomean * csmean / 2. * fabs(v[1:]-v[:-1])
        # f1half -= phi * rhomean / 2. * fabs(v[1:]-v[:-1])
    
    return f1half, f2half, f3half


def HLLCL(fs, qs, rho, press, v, gamma = None):
    '''
    Kitamura & Shima 2019
    '''
    f1, f2, f3 = fs  ;  q1, q2, q3 = qs
    #    ds = sr - sl
    # v = f1/q1
    
    nx=size(q1)

    if gamma is None:
        gamma = 4./3.

    cl = sqrt(gamma * press/rho)[:-1] ; cr = sqrt(gamma * press/rho)[1:]
    machl = minimum(1.0, 1.0 * sqrt((v[:-1]/cl)**2+0.01))
    machr = minimum(1.0, 1.0 * sqrt((v[1:]/cr)**2+0.01))
    phil = machl * (2.-machl) ;   phir = machr * (2.-machr)
    
    # cl1 = cl * phil ; cr1 = cr * phir
    cl1 = cl ; cr1 = cr
    sl = minimum(v[:-1]-cl1, v[1:]-cr1)
    sr = maximum(v[:-1]+cl1, v[1:]+cr1)
    sl1 = minimum(0., sl) ; sr1 = maximum(0., sr)
    sl0 = minimum(v[:-1]-cl, v[1:]-cr) ;  sr0 = maximum(v[:-1]+cl, v[1:]+cr)
    al = rho[:-1] * (v[:-1]-sl) ; ar = rho[1:] * (sr-v[1:])
    al0 = rho[:-1] * (v[:-1]-sl0) ; ar0 = rho[1:] * (sr0-v[1:])
    sm = (al * v[:-1] + ar * v[1:]) / (al+ar) - (press[1:]-press[:-1]) / (al0+ar0)

    p1 = (press[1:]+press[:-1] + al * (v[:-1]-sm) - ar * (v[1:]-sm))/2. # pstar
    p2 = copy(p1) # ptilde

    f1half=zeros(nx-1, dtype=double)  ;  f2half=zeros(nx-1, dtype=double)  ;  f3half=zeros(nx-1, dtype=double) 
    ds = sr1-sl1
    wreg = where(ds > 0.)
    w0 = where(ds <= 0.)
    
    f1half[wreg] = ((sr1 * f1[:-1] - sr1 * f1[1:])/(sr1-sl1) + sr1 * sl1 * (q1[1:]-q1[:-1])/(sr1-sl1))[wreg]
    f2half[wreg] = ((sr1*f2[:-1]-sl1*f2[1:])/(sr1-sl1)+sl1*sr1*(q2[1:]-q2[:-1])/(sr1-sl1))[wreg]
    f3half[wreg] = ((sr1*f3[:-1]-sl1*f3[1:])/(sr1-sl1)+sl1*sr1*(q3[1:]-q3[:-1])/(sr1-sl1))[wreg]
    # as in classic HLLE
    if size(w0)>0.:
        wpos = where((ds <=0.) & (sm >= 0.))
        wneg = where((ds <=0.) & (sm <= 0.))
        if size(wpos)>0.:
            fhalf1[wpos] = f1[:-1]
            fhalf2[wpos] = f2[:-1]
            fhalf3[wpos] = f3[:-1]
        if size(wneg)>0.:
            fhalf1[wneg] = f1[1:]
            fhalf2[wneg] = f2[1:]
            fhalf3[wneg] = f3[1:]

    return f1half, f2half, f3half
