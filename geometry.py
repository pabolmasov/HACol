from numpy import *
from scipy.integrate import cumtrapz

class geometry:
    r = None # radial (spherical) coordinate
    l = None # coordinate along the line
    sth = None # sin(theta), where theta is polar angle
    cth = None # cos(theta)
    across = None # cross-section
    sina = None # sin(alpha), where alpha sets the orientation of the tangent to the field line
    cosa = None # cos(alpha)
    delta = None # tangent (polar-angle) thickness of the tube

##############################################################################
def geometry_initialize(r, r_e, dr_e, writeout=None, afac = 1.):
    '''
    computes all the geometrical quantities. Sufficient to run once before the start of the simulation.
    Output: sin(theta), cos(theta), sin(alpha), cos(alpha), tangential cross-section area, 
    l (zero at the surface, growing with radius), and tangential thickness delta
    adding nontrivial writeout key allows to write the geometry to an ascii file 
    '''
    #    theta=arcsin(sqrt(r/r_e))
    #    sth=sin(theta) ; cth=cos(theta)
    sth=sqrt(r/r_e) ; cth=sqrt(1.-r/r_e) # OK
    across=4.*pi*afac*dr_e*r_e*(r/r_e)**3/sqrt(1.+3.*cth**2) # follows from Galja's formula (17)
    alpha=arctan((cth**2-1./3.)/sth/cth) # Galja's formula (3)
    sina=sin(alpha) ; cosa=cos(alpha)
    l=cumtrapz(sqrt(1.+3.*cth**2)/2./cth, x=r, initial=0.) # coordinate along the field line
    delta = r * sth/sqrt(1.+3.*cth**2) * dr_e/r_e
    # transverse thickness of the flow (Galya's formula 17)
    # dl diverges near R=Re, hence the maximal radius should be smaller than Re
    # ascii output:
    if(writeout is not None):
        theta=arctan(sth/cth)
        fgeo=open(writeout, 'w')
        fgeo.write('# format: r -- theta -- alpha -- across -- l -- delta \n')
        for k in arange(size(l)):
            fgeo.write(str(r[k])+' '+str(theta[k])+' '+str(alpha[k])+' '+str(across[k])+' '+str(l[k])+' '+str(delta[k])+'\n')
        fgeo.close()

    g = geometry()
    g.r = r ; g.l = l ; g.sth = sth ; g.cth = cth
    g.sina = sina ; g.cosa = cosa ; g.across = across; g.delta = delta
    
    return g

def geometry_split(ginit, np, half = False):
    '''
    replaces the single solid mesh with np chunks of equal length
    '''
    if np <= 1:
        return [ginit]
    else:
        if not(half):
            nchunk = size(ginit.r) // np
        else:
            nchunk = (size(ginit.r)+1) // np
        if ((nchunk * np) != size(ginit.r)) and not(half):
            print("geometry_split: dimensions unequal")
            exit(1)
        glist = []
        for k in arange(np):
            start = nchunk*k
            if not(half):
                finish = start+nchunk
            else:
                finish = start+nchunk-1
            g = geometry()
            g.r = ginit.r[start:finish]
            g.l = ginit.l[start:finish]
            g.sth = ginit.sth[start:finish]
            g.cth = ginit.cth[start:finish]
            g.sina = ginit.sina[start:finish]
            g.cosa = ginit.cosa[start:finish]
            g.across = ginit.across[start:finish]
            g.delta = ginit.delta[start:finish]
            glist.append(g)
        return glist

def dlbounds_define(glist):
    np = size(glist)
    dlleft = zeros(np) ; dlright = zeros(np)

    #    dlleft[0] = glist[0].l[1]-glist[0].l[0]
    #    dlright[-1] = glist[-1].l[-1]-glist[-1].l[-2]
    
    for k in range(np):
        if k > 0:
            dlleft[k] = glist[k].l[0]-glist[k-1].l[-1]
        else:
            dlleft[k] = glist[k].l[1]-glist[k].l[0]
        if k<(np-1):
            dlright[k] = glist[k+1].l[0]-glist[k].l[-1]
        else:
            dlright[k] = glist[k].l[-1]-glist[k].l[-2]
            
    return dlleft, dlright

def geometry_local(g, n):
    # defines a local, single-point, piece of geometry
    g1 = geometry()
    g1.l = asarray([g.l[n]])  
    g1.r = asarray([g.r[n]]) ; g1.sth = asarray([g.sth[n]]) ; g1.cth = asarray([g.cth[n]])
    g1.across = asarray([g.across[n]]) ; g1.delta = asarray([g.delta[n]])
    g1.cosa = asarray([g.cosa[n]]) ; g1.sina = asarray([g.sina[n]])
    return g1

def geometry_add(g1, g2):
    # merges two geometries
    gadd = geometry()
    #    nx1 = size(g1.r) ; nx2 = size(g2.r)
    gadd.l = concatenate([g1.l[:], g2.l[:]])
    gadd.r = concatenate([g1.r, g2.r]) ; gadd.sth = concatenate([g1.sth, g2.sth]) ; gadd.cth = concatenate([g1.cth, g2.cth])
    gadd.across = concatenate([g1.across, g2.across]) 
    gadd.delta = concatenate([g1.delta, g2.delta]) 
    gadd.cosa = concatenate([g1.cosa, g2.cosa]) 
    gadd.sina = concatenate([g1.sina, g2.sina]) 
    return gadd
    
def gread(geofile):
    ## reads a geometry file written by geometry-initialize
    lines = loadtxt(geofile, comments="#")
    r = lines[:,0] ; theta = lines[:,1] ; alpha = lines[:,2] 
    across = lines[:,3] ; l = lines[:,4] ; delta = lines[:,5]
    return r, theta, alpha, across, l, delta

