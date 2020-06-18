from numpy import *
import os

import plots
import beta
# from globals import umag
import geometry as geo

'''
ASCII reading and writing routines
'''

def readtireout(infile, ncol = 0):
    '''
    reading a tireout ASCII file
    ncol = 1 for rho
    ncol = 2 for v
    ncol = 3 for U/Umag
    '''
    lines = loadtxt(infile+'.dat', comments="#")
    r = squeeze(lines[:,0])
    if size(ncol) <= 1:
        q = squeeze(lines[:,ncol])
    else:
        q = []
        for k in arange(size(ncol)):
            q.append(squeeze(lines[:,ncol[k]]))
    return r, q

def galjaread(infile):
    '''
    reads the results of BS-type stationary solution and outputs x = R/RNS and
    u=u/umag
    '''
    lines = loadtxt(infile+'.dat', comments="#", delimiter="\t", unpack=False)
    
    x = squeeze(lines[:,0]) ; u = 3.*squeeze(lines[:,3]) ; v = squeeze(lines[:,5])
    prat =  squeeze(lines[:,11])

    #    plots.someplots(x, [v], name=infile+'_u', ylog=True, formatsequence=['k-'])
    
    return x, u, v, prat

def comparer(ingalja, inpasha, nentry = 1000, ifhdf = False, vnorm = None):

    xg, ug, vg, pratg = galjaread(ingalja)
    if vnorm is not None:
        vg *= vnorm 

    if ifhdf:
        entry, t, l, xp, sth, rho, up, vp, qloss  = read(inhdf, nentry)
    else:
        xp, qp = readtireout(inpasha, ncol = [3, 2, 1])
        up, vp, rhop = qp
    geofile = os.path.dirname(inpasha)+"/geo.dat"
    r, theta, alpha, across, l, delta = geo.gread(geofile)

    umagtar = umag * (1.+3.*cos(theta)**2)/4. * xp**(-6.)
    
    betap = beta.Fbeta(rhop, up * umagtar )
    pratp = betap / (1.-betap)
    
    plots.someplots([xg, xp], [ug, up], name='BScompare_u', ylog=True, formatsequence=['k-', 'r--'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$U/U_{\rm mag}$', multix = True)
    plots.someplots([xg, xp], [vg, vp], name='BScompare_v', ylog=False, formatsequence=['k-', 'r--'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$v/c$', multix = True)
    plots.someplots([xg, xp], [pratg, pratp], name='BScompare_p', ylog=False, formatsequence=['k-', 'r--'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$P_{\rm gas} / P_{\rm rad}$', multix = True)
# comparer('galia_F/BS_solution_F', 'titania_fidu/tireout01000')
