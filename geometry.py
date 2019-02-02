from numpy import *
from scipy.integrate import cumtrapz

class geometry:
    r: double # radial (spherical) coordinate
    l: double # coordinate along the line
    sth: double # sin(theta), where theta is polar angle
    cth: double # cos(theta)
    across: double # cross-section
    sina: double # sin(alpha), where alpha sets the orientation of the tangent to the field line
    cosa: double # cos(alpha)
    delta: double # tangent (polar-angle) thickness of the tube

##############################################################################
def geometry_initialize(r, r_e, dr_e, writeout=None, afac = 1.):
    '''
    computes all the geometrical quantities. Sufficient to run once before the start of the simulation.
    Output: sin(theta), cos(theta), sin(alpha), cos(alpha), tangential cross-section area, 
    l (zero at the surface, growing with radius), and tagnential thickness delta
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
    g.sina = sina ; g.cosa = cosa ; g. across = across; g.delta = delta
    
    return g
