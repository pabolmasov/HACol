from numpy import *
from scipy.integrate import *
from scipy.interpolate import *
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import os

import hdfoutput as hdf
import geometry as geo
import bassun as bs
from beta import *

import configparser as cp
conffile = 'globals.conf'
config = cp.ConfigParser(inline_comment_prefixes="#")
config.read(conffile)

ifplot =  config['DEFAULT'].getboolean('ifplot')

if ifplot:
    import plots
    from matplotlib.pyplot import ioff
    ioff()

def BSphysical(conf='DEFAULT'):
    '''
    outputs physical parameters of the BS solution from the surface up to the shock wave
    '''
    
    varkappa = 0.35 # temporary!!!
    
    rstar = config[conf].getfloat('rstar')
    # print('Rstar = ', rstar)
    #r_e = config[conf].getfloat('r_e')
    # print('Re = ', r_e)
    m1 = config[conf].getfloat('m1')
    mu30 = config[conf].getfloat('mu30')
    mdot = config[conf].getfloat('mdot') * 4.*pi # now it is in G M / c kappa
    xifac = config[conf].getfloat('xifac')
    r_e = config[conf].getfloat('r_e_coeff') * (mu30**2/mdot)**(2./7.)*m1**(-10./7.) * xifac # magnetosphere radius
    dr_e = config[conf].getfloat('drrat') * r_e
    afac = config[conf].getfloat('afac')
    mass1 = config[conf].getfloat('m1')
    tscale = config[conf].getfloat('tscale') * mass1
    rhoscale = config[conf].getfloat('rhoscale') / mass1
    rscale = config[conf].getfloat('rscale') * mass1
    realxirad = config[conf].getfloat('xirad')
    mow = config[conf].getfloat('mow')
    b12 = 2.*mu30*(rstar*m1/6.8)**(-3) # dipolar magnetic field on the pole, 1e12Gs units
    umag = b12**2*2.29e6*m1 # magnetic energy density at the bottom
    betacoeff = config[conf].getfloat('betacoeff') * (m1)**(-0.25)/mow

    # setting geometry:
    nx=100
    g = geo.geometry_initialize(asarray(rstar * (r_e / rstar)**(arange(nx)/double(nx))), r_e, dr_e, afac=afac) # radial-equidistant mesh
    
    BSgamma = (g.across/g.delta**2)[0]/mdot*rstar / (realxirad/1.5) * (0.35/varkappa)
    BSeta = (8./21./sqrt(2.) * (varkappa/0.35) * umag *3. * (realxirad/1.5))**0.25*sqrt(g.delta[0])/(rstar)**0.125
    
    x, v, u = bs.BSsolution(BSgamma, BSeta)

    v/= sqrt(rstar)

    print("u0 = ", b12**2/8./pi*3.*1e24,"erg/cm**3")
    # ii = input("u")
    # u *= 3. * umag

    g = geo.geometry_initialize(rstar * x, r_e, dr_e, afac=afac) # radial-equidistant mesh with dipolar geometry, 1000 elements, and extended up to xs

    rho = mdot / (v * g.across)  # code units
    
    # perimeter = 2. * (g.across/g.delta) # cooling from two sides
    taueff = rho * g.delta
    # taufactor = taufun(taueff, taumin, taumax) / (realxirad*taueff+1.)
    qloss = u / (realxirad*taueff+1.) # u in u[0]


    rho = mdot / (v * g.across) * rhoscale * 0.35/varkappa# physical density in g /cm^3
    
    # teff = 4.75 * mass1**(-0.25) * qloss**0.25 # keV
    teff = 242.88 * b12**0.5 * qloss**0.25 # keV

    tc = 171.742 * b12**0.5 * u**0.25 # temperature in keV

    r = g.r * rscale # physical length
    delta = g.delta * rscale

    bloc = b12/2. * sqrt(3.*g.cth**2+1.) * (rstar/g.r)**3 # local magnetic field, 10^12 G

    plots.someplots(r, [delta], name='BSdelta', xtitle='$R$, cm',
                    ytitle=r'$\delta$, cm', ylog=True,
                    formatsequence = ['k-'])
    plots.someplots(r, [rho], name='BSrho', xtitle='$R$, cm',
                    ytitle=r'$\rho,$ g cm$^{-3}$', ylog=True,
                    formatsequence = ['k-'])
    plots.someplots(r, [v], name='BSv', xtitle='$R$, cm',
                    ytitle=r'$v/c$', ylog=True,
                    formatsequence = ['k-'])
    plots.someplots(r, [teff], name='BSteff', xtitle='$R$, cm',
                    ytitle=r'$T_{\rm eff}$, keV', ylog=False,
                    formatsequence = ['k-'])
    plots.someplots(r, [bloc], name='BSB', xtitle='$R$, cm',
                    ytitle=r'$B$, $10^{12}$G', ylog=True,
                    formatsequence = ['k-'])
    # TODO: make an ASCII-saving module
    fout = open('BSphys.dat', 'w')
    fout.write("# gamma = "+str(BSgamma)+"; eta = "+str(BSeta)+"\n")
    fout.write("# R[cm]  -- delta[cm]  -- v/c -- tau -- rho [CGS] -- Teff [keV] -- B[10^12G]\n")
    nx = size(r)
    for k in arange(nx):
        s = str(r[k]) + " " + str(g.delta[k] * rscale) + " " + str(v[k]) + " " + str(taueff[k]) + " " + str(rho[k]) + " " + str(teff[k]) + " " + str(bloc[k]) + "\n"
        # print(s)
        fout.write(s)
    fout.flush()
    fout.close()
    
    w2 = abs(r-2e6).argmin()
    
    print("tc(2R_*) = ", tc[w2])
    print("l/delta [2R_*] = ", (g.across/g.delta**2)[w2]/2.)
    s = str(r[w2]) + " " + str(g.delta[w2] * rscale) + " " + str(v[w2]) + " " + str(taueff[w2]) + " " + str(rho[w2]) + " " + str(teff[w2]) + " " + str(bloc[w2]) + "\n"
    print("# R[cm]  -- delta[cm]  -- v/c -- tau -- rho [CGS] -- Teff [keV] -- B[10^12G]\n")
    print(s)

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
    
    x = squeeze(lines[:,0]) ; u = 3.*squeeze(lines[:,2]) ; v = squeeze(lines[:,3]) 
    rho = squeeze(lines[:,4]) ; prat =  squeeze(lines[:,6])

    #    plots.someplots(x, [v], name=infile+'_u', ylog=True, formatsequence=['k-'])
    
    return x, u, v, rho, prat

def acomparer(infile, nentry =1000, ifhdf = True, conf = 'DEFAULT', nocalc = False, trange = None, savetheta = False):
    '''
    compares the structure of the flow to the analytic solution by B&S76
    savetheta -- produces an additional output file with theta -- u/umag output
    '''
    
    sintry = size(nentry)
    
    rstar = config[conf].getfloat('rstar')
    rstarg = rstar
    m1 = config[conf].getfloat('m1')
    mu30 = config[conf].getfloat('mu30')
    mdot = config[conf].getfloat('mdot') * 4.*pi # TODO: should it be multiplied by mass1?
    afac = config[conf].getfloat('afac')
    mass1 = config[conf].getfloat('m1')
    tscale = config[conf].getfloat('tscale') * mass1
    rhoscale = config[conf].getfloat('rhoscale') / mass1
    realxirad = config[conf].getfloat('xirad')
    mow = config[conf].getfloat('mow')
    b12 = 2.*mu30*(rstar*m1/6.8)**(-3) # dipolar magnetic field on the pole, 1e12Gs units
    umag = b12**2*2.29e6*m1
    betacoeff = config[conf].getfloat('betacoeff') * (m1)**(-0.25)/mow

    if not nocalc:
        if ifhdf:
            inhdf = infile + '.hdf5'
            sintry = size(nentry)
            if sintry <= 1:
                entry, t, l, xp, sth, rhop, up, vp, qloss, glo, ediff  = hdf.read(inhdf, nentry)
                betap = Fbeta(rhop, up, betacoeff)
                dv = vp *0.
                du = up *0.
                drho = rhop*0.
                dbeta = betap*0.
                dqloss = qloss * 0.
            else:
                entry, t, l, xp, sth, rhop, up, vp, qloss, glo, ediff = hdf.read(inhdf, nentry[0])
                beta1 = Fbeta(rhop, up, betacoeff)
                betap = copy(beta1)
                nentries = 0 # nentry[1]-nentry[0]
                dv = copy(vp)**2
                du = copy(up)**2
                drho = copy(rhop)**2
                dbeta = copy(beta1)**2
                dqloss = copy(qloss)**2
                for k in arange(nentry[1]-nentry[0])+nentry[0]+1:
                    entry1, t, l, xp, sth, rho1, up1, vp1, qloss1, glo1, ediff1  = hdf.read(inhdf, k)
                    beta1 = Fbeta(rho1, up1, betacoeff)
                    t *= tscale
                    # print("t="+str(t))
                    # print(k)
                    if (trange is None) or ((t-trange[0])*(t-trange[1]) <= 0.):
                        rhop += rho1 ; up += up1 ; vp += vp1 ; qloss += qloss1 ; betap += beta1
                        dv += vp1**2
                        du += up1**2
                        drho += rho1**2
                        dbeta += beta1**2
                        dqloss += qloss1**2
                        if k == nentry[0]+1:
                            tstart = t
                        if k == nentry[1]-1:
                            tend = t
                        nentries += 1
                betap /= double(nentries)
                rhop /= double(nentries)
                up /= double(nentries)
                vp /= double(nentries)
                qloss /= double(nentries)
                dv = sqrt(dv/double(nentries) - vp**2)
                du = sqrt(du/double(nentries) - up**2)
                drho = sqrt(drho/double(nentries) - rhop**2)
                dbeta = sqrt(dbeta/double(nentries) - betap**2)
                dqloss = sqrt(dqloss/double(nentries) - qloss**2)

                dv[isnan(dv)] = 0. ; du[isnan(du)] = 0.
                # print("time range = "+str(tstart*tscale)+".."+str(tend*tscale)+"s")
                # ii = input("T")
        else:
            sintry=0
            xp, qp = readtireout(infile, ncol = [3, 2, 1])
            up, vp, rhop = qp
    else:
        lines = loadtxt(os.path.dirname(infile) + '/avprofile.dat', comments = '#')
        ls = shape(lines)
        # print(ls)
        xp = lines[:,0] ; vp = lines[:,1] ; up = lines[:,2] ; betap = lines[:,3] ; tempp = lines[:,4] ; rhop = lines[:,5] ; qloss = lines[:,6]
        dv = lines[:,7] ; du = lines[:,8] ; dbeta = lines[:,minimum(10, ls[1]-1)] ; dqloss = lines[:,minimum(10, ls[1]-1)]
        sintry = 0
        #  betap = pratp / (1.+pratp)

    geofile = os.path.dirname(infile)+"/geo.dat"
    r, theta, alpha, across, l, delta = geo.gread(geofile)
    perimeter = 2. * (across/delta + 2.*delta)
    
    qloss /= perimeter ; dqloss /= perimeter # flux from unit surface area

    umagtar = umag * (1.+3.*cos(theta)**2)/4. * (r/rstar)**(-6.)

    BSgamma = (across/delta**2)[0]/mdot*rstar / (realxirad/1.5)
    # umag is magnetic pressure
    b12 = 2.*mu30*(rstar*m1/6.8)**(-3) # dipolar magnetic field on the pole, 1e12Gs units
    # umag = b12**2*2.29e6*m1
    
    BSeta = (8./21./sqrt(2.)*umag*3. * (realxirad/1.5))**0.25*sqrt(delta[0])/(rstar)**0.125
    print("BSgamma = "+str(BSgamma))
    print("BSeta = "+str(BSeta))
    xs, BSbeta = bs.xis(BSgamma, BSeta, x0=4., ifbeta = True)

    BSr, BSv, BSu = bs.BSsolution(BSgamma, BSeta)
    BSv *= -1./sqrt(rstar)
    BSumagtar = umag * BSr**(-6.)

    BSu = 3. * BSu/BSu[0] * BSr**6
    tempg = (BSu * BSumagtar / mass1)**(0.25) * 3.35523 # keV
    if nocalc:
        tempp = (up * umagtar / mass1)**(0.25) * 3.35523 # keV
    else:
        tempp = (up / mass1)**(0.25) * 3.35523 # keV
    acrossfun = interp1d(r/rstar, across, bounds_error=False)
    BSacross = acrossfun(BSr)
    BSrho = -mdot / BSacross / BSv

    betag = Fbeta(BSrho, BSu * BSumagtar, betacoeff)
    pratg = copy(betag / (1.-betag))
    
    # virialbetaP =  (up+pressp) * umagtar / rhop * rstar
    # virialbetaBS = (8.+5.*pratg)/(6.+3.*pratg) * BSu * BSumagtar / (BSrho/rhoscale) * rstar

    dirname = os.path.dirname(infile)

    if not nocalc:
        # we need to save the result as an ASCII file, then
        fout = open(dirname + '/avprofile.dat', 'w')
        if savetheta:
            fth = open(dirname + '/thprofile.dat','w')
            fth.write('# theta -- u/umag \n')
        fout.write("# R  -- v -- u -- Prat -- T -- rho -- qloss -- dv -- du -- dbeta -- dqloss \n")
        nx = size(xp)
        for k in arange(nx):
            s = str(xp[k]) + " " + str(vp[k]) + " " + str((up/umagtar)[k]) + " " + str(betap[k]) + " " + str(tempp[k]) + " " + str(rhop[k]) + " " + str(qloss[k]) + " " + str(dv[k])+ " "+str((du/umagtar)[k])+" "+str(dbeta[k])+" "+str(dqloss[k])+"\n"
            if savetheta:
                fth.write(str(theta[k])+' '+ str((up/umagtar)[k]) + '\n')
            print(s)
            fout.write(s)
            fout.flush()
        fout.close()
        if savetheta:
            fth.close()
    if ifplot:
        if nocalc:
            press = up / 3. / (1.-betap/2.)
        else:
            press = up / 3. / (1.-betap/2.) / (umagtar)
        if (sintry > 1) or nocalc:
            plots.someplots([r/rstar, BSr, r/rstar, r/rstar, r/rstar, r/rstar], [-vp, -BSv, 1./sqrt(r), 1./sqrt(r)/7., -vp+dv, -vp-dv], name=dirname+'/acompare_v', ylog=True, formatsequence=['r--', 'k-', 'b:', 'b:', 'r:', 'r:'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$-v/c$', multix = True, yrange= [-BSv.max()/2., -BSv.min()*7.*2.], inchsize=[5,3])
            print(str(sintry)+' = sintry')
            # print(du.max())
            # plots pressure:
            plots.someplots([BSr, r/rstar, r/rstar, r/rstar, r/rstar], [BSu/3., press, up*0.+1., press+du/up*press, press-du/up*press], name=dirname+'/acompare_u', ylog=False, formatsequence=['k-', 'r--', 'b:', 'r:', 'r:'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$p/p_{\rm mag}$', multix = True, inchsize=[5,3], yrange = [BSu.min()/10., maximum(press.max()*2.,5.)])
            # yrange = [BSu.min()/10., maximum(BSu.max()*2.,5.)], inchsize=[5,3])
            plots.someplots([BSr, r/rstar, r/rstar, r/rstar], [betag, betap, betap+dbeta, betap-dbeta], name=dirname+'/acompare_p', ylog=False, formatsequence=['k-', 'r--', 'r:', 'r:'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$\beta$', multix = True, inchsize=[5,3])
            #     plots.someplots([BSr, r/rstar, r/rstar, r/rstar], [BSrho, rhop, rhop+drho, rhop-drho], name=dirname+'/acompare_rho', ylog=True, formatsequence=['k-', 'r--', 'r:', 'r:'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$\rho/\rho^*$', multix = True)
        else:
            plots.someplots([r/rstar, BSr, r/rstar, r/rstar], [-vp, -BSv, 1./sqrt(r), 1./sqrt(r)/7.], name=dirname+'/acompare_v', ylog=True, formatsequence=['r--', 'k-', 'b:', 'b:', 'r:', 'r:'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$-v/c$', multix = True, yrange= [-BSv.max()/2., -BSv.min()*7.*2.], inchsize=[5,3])
            press = up / 3. / (1.-betap/2.)
            # plots pressure:
            plots.someplots([BSr, r/rstar, r/rstar], [BSu/3., press, up*0.+1.], name=dirname+'/acompare_u', ylog=True, formatsequence=['k-', 'r--', 'b:'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$p/p_{\rm mag}$', multix = True, yrange = [BSu.min()/10., maximum(up.max()*2.,5.)], inchsize=[5,3])
            plots.someplots([BSr, r/rstar], [BSrho, rhop], name=dirname+'/acompare_rho', ylog=True, formatsequence=['k-', 'r--'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$\rho/\rho^*$', multix = True, inchsize=[5,3])
        plots.someplots([BSr, r/rstar], [tempg, tempp], name=dirname+'/acompare_T', ylog=True, formatsequence=['k-', 'r--'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$T$, keV', multix = True, inchsize=[5,3])
        teff = 4.75 * mass1**(-0.25) * qloss**0.25 # keV
        dteff = dqloss / qloss * teff * 0.25
        plots.someplots([r/rstar], [teff], name=dirname+'/acompare_Teff', ylog=False, formatsequence=['k-', 'r--'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$T_{\rm eff}$, keV', multix = True, inchsize=[5,3])
 #        plots.someplots([BSr, r/rstar], [betag, betap], name=dirname+'/acompare_p', ylog=True, formatsequence=['k-', 'r--'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$\beta$', multix = True)

    print(dirname)

def avcompare_list(dirlist, rrange = None, conf = 'DEFAULT', tauscale = None):
    '''
    tauscale is the number of the profile used to calculate the optical depth
    '''
    mass1 = config[conf].getfloat('m1')

    nf = size(dirlist)
    
    xlist = [] ; vlist = [] ; ulist = [] ; plist = [] ; qlist = [] ; tefflist = []
    presslist = []
    
    if tauscale is not None:
        taulist = []
    
    for k in arange(nf):
        lines = loadtxt(dirlist[k] + '/avprofile.dat', comments = '#')
        xp = lines[:,0] ; vp = lines[:,1] ; up = lines[:,2] ; pratp = lines[:,3] ; tempp = lines[:,4] ; rhop = lines[:,5] ; qloss = lines[:,6]
        teff = 4.75 * mass1**(-0.25) * qloss**0.25 # keV
        xlist.append(xp) ; vlist.append(vp) ; ulist.append(up)  ; plist.append(pratp/(1.+pratp))
        presslist.append(up/3./(1.-pratp/(1.+pratp)))
        qlist.append(qloss) ; tefflist.append(teff)
        
        if tauscale is not None:
            geofile = dirlist[k]+'/geo.dat'
            r, theta, alpha, across, l, delta = geo.gread(geofile)
            # r/r[0] shd coincide with xp
            tau = cumulative_trapezoid(rhop, x=r)
            # position of the shock front:
            dvdl = abs((vp[1:]-vp[:-1])/(l[1:]-l[:-1]))
            wfront = dvdl.argmax()
            tau -= tau[wfront]
            taulist.append(tau)
            if k == tauscale:
                tau2x = interp1d(tau, (xp[1:]+xp[:-1])/2., bounds_error = False, fill_value = (tau[0], tau[-1]))
                x2tau = interp1d((xp[1:]+xp[:-1])/2., tau, bounds_error = False, fill_value = (xp[0], xp[-1]))

    if rrange is None:
        plots.someplots(xlist, vlist, multix=True, formatsequence=['r--', 'k-', 'g:', 'b-.'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$-v/c$', name='avtwo_v', inchsize=[5,3])
        plots.someplots(xlist, presslist, multix=True, formatsequence=['r--', 'k-', 'g:', 'b-.'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$P/P_{\rm mag}$', name='avtwo_u', ylog=True, inchsize=[5,3])
        plots.someplots(xlist, plist, multix=True, formatsequence=['r--', 'k-', 'g:', 'b-.'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$\beta$', name='avtwo_p', ylog=True, inchsize=[5,3])
        plots.someplots(xlist, qlist, multix=True, formatsequence=['r--', 'k-', 'g:', 'b-.'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$Q^-$', name = 'avtwo_q', xlog = True, ylog = True, inchsize=[5,3])
        plots.someplots(xlist, tefflist, multix=True, formatsequence=['r--', 'k-', 'g:', 'b-.'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$T_{\rm eff}$, keV', name = 'avtwo_teff', xlog = True, ylog = False, inchsize=[5,3])
    else:
        plots.someplots(xlist, vlist, multix=True, formatsequence=['r--', 'k-', 'g:', 'b-.'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$-v/c$', name='avtwo_v', xrange = rrange, xlog = False, inchsize=[5,3])
        urange = [ulist[-1][(xlist[-1]>rrange[0]) & (xlist[-1]<rrange[1])].min()*0., ulist[-1][(xlist[-1]>rrange[0]) & (xlist[-1]<rrange[1])].max()*1.1]
        plots.someplots(xlist, ulist, multix=True, formatsequence=['r--', 'k-', 'g:', 'b-.'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$U/U_{\rm mag}$', name='avtwo_u', xrange = rrange, xlog = False, yrange = urange, inchsize=[5,3])
        qrange = [qlist[-1][(xlist[-1]>rrange[0]) & (xlist[-1]<rrange[1])].min()*0., qlist[-1][(xlist[-1]>rrange[0]) & (xlist[-1]<rrange[1])].max()*1.1]
        teffrange = [tefflist[-1][(xlist[-1]>rrange[0]) & (xlist[-1]<rrange[1])].min()*0., tefflist[-1][(xlist[-1]>rrange[0]) & (xlist[-1]<rrange[1])].max()*1.1]
        plots.someplots(xlist, qlist, multix=True, formatsequence=['r--', 'k-', 'g:', 'b-.'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$Q^-$', name = 'avtwo_q', xrange = rrange, yrange = qrange, xlog = False, inchsize=[5,3])
        if tauscale is not None:
            plots.someplots(xlist, tefflist, multix=True, formatsequence=['r--', 'k-', 'g:', 'b-.'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$T_{\rm eff}$, keV', name = 'avtwo_teff', xrange = rrange, yrange = teffrange, xlog = False, inchsize=[5,3], secaxfunpair = (x2tau, tau2x))
            print(tau2x(-1.), tau2x(0.), tau2x(1.))
        else:
            plots.someplots(xlist, tefflist, multix=True, formatsequence=['r--', 'k-', 'g:', 'b-.'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$T_{\rm eff}$, keV', name = 'avtwo_teff', xrange = rrange, xlog = False, inchsize=[5,3])

def comparer(ingalja, inpasha, nentry = 1000, ifhdf = True, conf = 'DEFAULT', vone = None, nocalc = False):

    if ifplot:
        xg, ug, vg, rhog, pratg = galjaread(ingalja)
        if vone is not None:
            vg *= vone
        betag = pratg / (1.+pratg)
    
    rstar = config[conf].getfloat('rstar')
    rstarg = rstar
    m1 = config[conf].getfloat('m1')
    mu30 = config[conf].getfloat('mu30')
    mdot = config[conf].getfloat('mdot') * 4.*pi # TODO: *mass1?
    afac = config[conf].getfloat('afac')
    mass1 = config[conf].getfloat('m1')
    tscale = config[conf].getfloat('tscale') * mass1
    rhoscale = config[conf].getfloat('rhoscale') / mass1
    realxirad = config[conf].getfloat('xirad')
    mow = config[conf].getfloat('mow')
    b12 = 2.*mu30*(rstar*m1/6.8)**(-3) # dipolar magnetic field on the pole, 1e12Gs units
    umag = b12**2*2.29e6*m1
    betacoeff = config[conf].getfloat('betacoeff') * (m1)**(-0.25)/mow

    if not nocalc:
        if ifhdf:
            inhdf = inpasha + '.hdf5'
            sintry = size(nentry)
            if sintry <= 1:
                entry, t, l, xp, sth, rhop, up, vp, qloss, glo  = hdf.read(inhdf, nentry)
                dv = 0.*vp
            else:
                entry, t, l, xp, sth, rhop, up, vp, qloss, glo  = hdf.read(inhdf, nentry[0])
                nentries = nentry[1]-nentry[0]
                dv = copy(vp) * 0.
                for k in arange(nentries-1)+nentry[0]+1:
                    entry1, t, l, xp, sth, rho1, up1, vp1, qloss1, glo1  = hdf.read(inhdf, k)
                    rhop += rho1 ; up += up1 ; vp += vp1 ; qloss += qloss1
                    dv += vp1**2
                    if k == nentry[0]+1:
                        tstart = t
                    if k == nentry[1]-1:
                        tend = t
                rhop /= double(nentries)
                up /= double(nentries)
                vp /= double(nentries)
                qloss /= double(nentries)
                dv = sqrt(dv/double(nentries) - vp**2)
                print("time range = "+str(tstart*tscale)+".."+str(tend*tscale)+"s")
        else:
            sintry=0
            xp, qp = readtireout(inpasha, ncol = [3, 2, 1])
            up, vp, rhop = qp
        geofile = os.path.dirname(inpasha)+"/geo.dat"
        r, theta, alpha, across, l, delta = geo.gread(geofile)
    # lfun = interp1d(r,l)
    # xl = lfun(xp*rstar)/rstar
    else:
        lines = loadtxt(os.path.dirname(inpasha) + '/avprofile.dat', comments = '#')
        xp = lines[:,0] ; vp = lines[:,1] ; up = lines[:,2] ; pratp = lines[:,3] ; tempp = lines[:,4] ; rhop = lines[:,5]
        virialbetaP = lines[:,6]
        sintry = 0
        betap = pratp / (1.+pratp)
    
    if not nocalc:
        umagtar = umag * (1.+3.*cos(theta)**2)/4. * xp**(-6.)
        if ifhdf:
            up /= umagtar
    
        betap = Fbeta(rhop, up * umagtar, betacoeff)
        pratp = betap / (1.-betap)
        pressp = up / 3. / (1.-betap/2.)

        # internal temperatures:
        tempp = (up * umagtar / mass1)**(0.25) * 3.35523 # keV
        uscale = rhoscale*0.898755
        print("umagscale = "+str(rhoscale*0.898755)+"x10^{21}erg/c^3")
        print("physical energy density on the pole and on the column foot:")
        print(str(umag * uscale*3.)+"; "+str(umagtar[0]*uscale*3.)+"x10^{21}erg/cm^3")
        print("compare to 4.774648")
    
        # let us estimate post-factum beta:
        virialbetaP =  (up+pressp) * umagtar / rhop * rstar
        print("measured in situ (Pasha) betaBS = "+str(virialbetaP[0:5]))
        rstarg = rstar
    
    if ifplot:

        umagtar_g = umag * (1.+3.*(1.-xg/xp.max()))/4. * xg**(-6.)
        tempg = (ug * umagtar_g / mass1)**(0.25) * 3.35523 # keV

        virialbetaG = (8.+5.*pratg)/(6.+3.*pratg) * ug * umagtar_g / (rhog/rhoscale) * rstarg
        print("measured in situ (Galja) betaBS = "+str(virialbetaG[0:5]))
        print("pratg = "+str(pratg[0:5]))

        outdir = os.path.dirname(ingalja)+'/'
        print('writing to '+outdir)
        plots.someplots([xg, xp, xp], [ug, up, up*0.+3.], name=outdir+'BScompare_u', ylog=True, formatsequence=['k-', 'r--', 'b:'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$U/U_{\rm mag}$', multix = True)
        if sintry >= 1:
            plots.someplots([xp, xg, xp, xp, xp, xp], [-vp, -vg, 1./sqrt(rstar*xp), 1./sqrt(rstar*xp)/7., -vp+dv, -vp-dv], name=outdir+'BScompare_v', ylog=True, formatsequence=['r--', 'k-', 'b:', 'b:', 'r:', 'r:'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$-v/c$', multix = True)
        else:
            plots.someplots([xp, xg, xp, xp], [-vp, -vg, 1./sqrt(rstar*xp), 1./sqrt(rstar*xp)/7.], name=outdir+'BScompare_v', ylog=True, formatsequence=['r--', 'k-', 'b:', 'b:', 'r:', 'r:'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$-v/c$', multix = True)
        plots.someplots([xg, xp], [betag, betap], name=outdir+'BScompare_p', ylog=True, formatsequence=['k-', 'r--'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$\beta$', multix = True)
        plots.someplots([xg, xp], [tempg, tempp], name=outdir+'BScompare_T', ylog=True, formatsequence=['k-', 'r--'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$T$, keV', multix = True)
        plots.someplots([xg, xp], [rhog/rhoscale, rhop], name=outdir+'BScompare_rho', ylog=True, formatsequence=['k-', 'r--'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$\rho/\rho^*$', multix = True)
        plots.someplots([xg, xp], [virialbetaG, virialbetaP], name=outdir+'BScompare_virbeta', ylog=False, formatsequence=['k-', 'r--'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$\beta_{\rm BS}$', multix = True)
    if not nocalc:
        # we need to save the result as an ASCII file, then
        dirname = os.path.dirname(inpasha)
        fout = open(dirname + '/avprofile.dat', 'w')
        fout.write("# R  -- v -- u -- Prat -- rho -- betaBS \n")
        nx = size(xp)
        for k in arange(nx):
            s = str(xp[k]) + " " + str(vp[k]) + " " + str(up[k]) + " " + str(pratp[k]) + " " + str(tempp[k]) + " " + str(rhop[k]) + " " + str(virialbetaP[k]) + "\n"
            print(s)
            fout.write(s)
            fout.flush()
        fout.close()
        
# comparer('galia_F/BS_solution_F', 'titania_bs/tireout', vone = -9.778880e+05/3e10, nentry = [4000,5000])
# comparer('galia_N/BS_solution_N', 'titania_narrow2/tireout', nentry = [4000,5000], vone = -8.194837e+06/3e10, nocalc = True)
# comparer('galia_M100/BS_solution_M100', 'titania_mdot100/tireout06000', vone = -1.957280e+06/3e10)
    
def rcoolfun(geometry, mdot):
    '''
    calculates the cooling radius from the known geometry
    '''
    r = geometry[:,0] ; across = geometry[:,3] ; delta = geometry[:,5]

    f = delta**2/across * mdot - r
    ffun = interp1d(f, r, bounds_error = False)
    return ffun(0.)
    
def pds(infile='out/flux', binning=None, binlogscale=False):
    '''
    makes a power spectrum plot;
    input infile+'.dat' is an ascii, 2+column dat-file with an optional comment sign of #
    if binning is set, it should be the number of frequency bins; binlogscale makes frequency binning logarithmic
    '''
    lines = loadtxt(infile+".dat", comments="#", delimiter=" ", unpack=False)
    t=lines[:,0] ; l=lines[:,1]
    print("mean flux "+str(l.mean())+"+/-"+str(l.std()))
    # remove linear trend!
    linfit = polyfit(t, l, 1)
    f=fft.rfft((l-linfit[0]*t-linfit[1])/l.std(), norm="ortho")
    freq=fft.rfftfreq(size(t),t[1]-t[0])
    
    pds=abs(f)**2
    if ifplot:
        plots.pdsplot(freq, pds, outfile=infile+'_pds')
    
    # additional ascii output:
    fpds=open(infile+'_pds.dat', 'w')
    for k in arange(size(freq)-1)+1:
        fpds.write(str(freq[k])+' '+str(pds[k])+'\n')
    fpds.close()

    if binning != None:
        if binlogscale:
            binfreq=(freq.max()/freq[freq>0.].min())**(arange(binning+1)/double(binning))*freq[freq>0.].min()
            binfreq[0]=0.
        else:
            binfreq=linspace(freq[freq>0.].min(), freq.max(), binning+1)
        binflux=zeros(binning) ; dbinflux=zeros(binning)
        binfreqc=(binfreq[1:]+binfreq[:-1])/2. # bin center
        binfreqs=(binfreq[1:]-binfreq[:-1])/2. # bin size
        for k in arange(binning):
            win=((freq<binfreq[k+1]) & (freq>=binfreq[k]))
            binflux[k]=pds[win].mean() ; dbinflux[k]=pds[win].std()/sqrt(double(win.sum()))

        fpds=open(infile+'_pdsbinned.dat', 'w')
        for k in arange(binning):
            fpds.write(str(binfreq[k])+' '+str(binfreq[k+1])+' '+str(binflux[k])+' '+str(dbinflux[k])+'\n')
        fpds.close()
        if ifplot:
            plots.binplot_short(binfreqc, binfreqs, binflux, dbinflux, outfile=infile+'_pdsbinned')



def dynspec(infile='out/flux', ntimes=10, nbins=100, binlogscale=False, deline = False, ncol = 5, iffront = False, stnorm = False, fosccol = None, simfreq = None, conf = 'DEFAULT', trange = None, smoothing = False):
    '''
    makes a dynamic spectrum by making Fourier in each of the "ntimes" time bins. Fourier PDS is binned to "nbins" bins
    "ncol" is the number of data column in the input file (the last one is taken by default)
    
    '''
    rstar = config[conf].getfloat('rstar')
    m1 = config[conf].getfloat('m1')
    tscale = config[conf].getfloat('tscale') * m1
    mdot = config[conf].getfloat('mdot') * 4.*pi
    mu30 = config[conf].getfloat('mu30')
    realxirad = config[conf].getfloat('xirad')

    lines = loadtxt(infile+".dat", comments="#", delimiter=" ", unpack=False)
    slines = shape(lines)
    if ncol >= slines[1]:
        ncol = -1
    t=lines[:,0] ; l=lines[:,ncol]
    if fosccol is not None:
        fosc = lines[:, fosccol]
    if simfreq is not None:
        l = l.mean() * sin(2.*pi*t*simfreq)*exp(-t)
        
    if smoothing:
        period = 0.5
        phase = (t / period) % 1.
        w = phase.argsort()
        w = w[t[w]>period]
        l_spline = splrep(phase[w], l[w], s=0.5,k=5,per=True)
    
    if trange is not None:
        w = (t> trange[0]) * (t<trange[1])
        t=t[w] ;  l=l[w]
        if fosccol is not None:
            fosc = fosc[w]
        # print(t.min(), t.max())

    if smoothing:
        # smoothing and subtraction:
        plots.someplots(t, [l, BSpline(*l_spline)(phase)], name='spline', formatsequence=['k.', 'r-'], xtitle = r'$t$, s', ytitle =  r'$L/L_{\rm Edd}$', xlog=False, ylog=False)
        plots.someplots(t, [l-BSpline(*l_spline)(phase)], name='spline_diff',formatsequence=['k-'], xtitle = r'$t$, s', ytitle =  r'$\Delta L/L_{\rm Edd}$', xlog=False, ylog=False)
        plots.someplots(t, [l-BSpline(*l_spline)(phase)], name='spline_diff_detail',formatsequence=['k-'], xtitle = r'$t$, s', ytitle =  r'$\Delta L/L_{\rm Edd}$', xlog=False, ylog=False, xrange=[0.2,0.50], yrange=[-0.1,0.1])
       # ii =input('spline')
        l = l - BSpline(*l_spline)(phase)
    
    if iffront:
        if trange is not None:
            xs = (lines[:,1])[w] # if we want to correlate the maximum with the mean front position
        else:
            xs = (lines[:,1])
    else:
        xs = l # or, we can calculate a bin-average flux instead
    nsize=size(t)
    tbin=linspace(t.min(), t.max(), ntimes+1)
    tcenter=(tbin[1:]+tbin[:-1])/2.
    tsize = (tbin[1:]-tbin[:-1])/2.
    freq1=1./(t.max()-t.min())*double(ntimes)/2. ; freq2 = minimum(1./median(t[1:]-t[:-1])/2., 2000.)
    # print(freq1, freq2)
    # ii =input('F')
    if(binlogscale):
        binfreq=logspace(log10(freq1), log10(freq2), num=nbins+1)
    else:
        binfreq=linspace(freq1, freq2, nbins+1)
    freq1 = 0.
    binfreqc=(binfreq[1:]+binfreq[:-1])/2.
    binfreqs=(binfreq[1:]-binfreq[:-1])/2.
    pds2=zeros([ntimes, nbins]) ;   dpds2=zeros([ntimes, nbins])
    t2=zeros([ntimes+1, nbins+1], dtype=double)
    nbin=zeros([ntimes, nbins], dtype=double)
    binfreq2=zeros([ntimes+1, nbins+1], dtype=double)
    fmax = zeros(ntimes) ;   dfmax = zeros(ntimes)
    xmean = zeros(ntimes) ; xstd = zeros(ntimes)
    # average PDS
    taver = [0.25, 0.4] # averaging interval for meansp
    
    if fosccol is not None:
        foscmean = zeros(ntimes) ; foscstd = zeros(ntimes)
    fdyns=open(infile+'_dyns.dat', 'w')
    ffreqmax = open(infile+'_fmax.dat', 'w')
    for kt in arange(ntimes):
        wt=(t<tbin[kt+1]) & (t>=tbin[kt])
        lt=xs[wt]
        pfit = polyfit(t[wt], lt, 1)
        if deline:
            fsp=fft.rfft((lt-pfit[0]*t[wt]-pfit[1])) #, norm="ortho")
        else:
            fsp=fft.rfft((lt-lt.mean())) #, norm="ortho")
        if stnorm:
            fsp /= lt.std()
        else:
            fsp *= 2./lt.sum() # Miyamoto normalization, see Nowak et al.(1999) The Astrophysical Journal, Volume 510, Issue 2, pp. 874-891
        nt=size(lt)
        # print("nt ="+str(nt))
        dt = median((t[wt])[1:]-(t[wt])[:-1])
        # print("dt = "+str(dt))
        freq = fft.rfftfreq(nt, dt)
        pds=real(fsp*freq)**2+imag(fsp*freq)**2
        # print(pds)
        # ii = input('P')
        # print(freq)
        # print(binfreq)
        # ii=input('P')
        t2[kt,:]=tbin[kt] ; t2[kt+1,:]=tbin[kt+1] 
        binfreq2[kt,:]=binfreq[:] ; binfreq2[kt+1,:]=binfreq[:] 
        for kb in arange(nbins):
            wb=((freq>binfreq[kb]) & (freq<=binfreq[kb+1]))
            nbin[kt,kb] = wb.sum()
            #            print("size(f) = "+str(size(freq)))
            print("size(pds) = "+str(size(pds)))
            if wb.sum() > 1:
                pds2[kt, kb]=pds[wb].mean() ; dpds2[kt, kb]=pds[wb].std()
            # ascii output:
            fdyns.write(str(tcenter[kt])+' '+str(binfreq[kb])+' '+str(binfreq[kb+1])+' '+str(pds2[kt,kb])+' '+str(dpds2[kt,kb])+" "+str(nbin[kt,kb])+"\n")
            print(str(tcenter[kt])+' '+str(binfreq[kb])+' '+str(binfreq[kb+1])+' '+str(pds2[kt,kb])+' '+str(dpds2[kt,kb])+" "+str(nbin[kt,kb])+"\n")
            # ii = input('P')
        # finding maximum:
        nfmax = ((pds2*(nbin>1)*(arange(nbins)>0))[kt,:]).argmax()
        # print((pds2*(nbin>1)*(arange(nbins)>0)))
        # print("nfmax = "+str(nfmax))
        fmax[kt] = (binfreq2[kt,nfmax]+binfreq2[kt,nfmax+1])/2. # frequency of the maximum
        dfmax[kt] = (-binfreq2[kt,nfmax]+binfreq2[kt,nfmax+1])/2.
        ffreqmax.write(str(tcenter[kt])+" "+str(tsize[kt])+" "+str(fmax[kt])+" "+str(dfmax[kt])+"\n")
        print(str(tcenter[kt])+" "+str(tsize[kt])+" "+str(fmax[kt])+" "+str(dfmax[kt])+"\n")
        xmean[kt] = xs[wt].mean() ; xstd[kt] = xs[wt].std()
        if fosccol is not None:
            foscmean[kt]=fosc[wt].mean() ; foscstd[kt]=fosc[wt].std()
    fdyns.close()
    ffreqmax.close()
    print(t2.max())
    if ifplot:
        frange = plots.plot_dynspec(t2, binfreq2, pds2, outfile=infile+'_dyns', nbin=nbin, logscale = True)
        plots.errorplot(tcenter, tsize, fmax, dfmax, outfile = infile + '_ffmax', xtitle = '$t$, s', ytitle = '$f$, Hz')
        tfilter  = (t2[1:,1:]<taver[1])&(t2[:-1,1:]>taver[0])
        pds1 = (pds2* tfilter).sum(axis = 0) / (tfilter).sum(axis = 0)
        dpds1 = (pds2* tfilter).std(axis = 0) / sqrt((tfilter).sum(axis = 0))
        plots.errorplot(binfreqc, binfreqs, pds1/binfreqc**2, dpds1/binfreqc**2, outfile = infile + '_msp', xtitle = '$f$, Hz', ytitle = 'PDS', ylog = True)
        if iffront:
            # we need geometry:
            outdir = os.path.dirname(infile)
            geometry = loadtxt(outdir+"/geo.dat", comments="#", delimiter=" ", unpack=False)
            geo_r = geometry[:,0]  ; across = geometry[:,3]  ;   delta = geometry[:,5]
            th = geometry[:,1]
            cthfun = interp1d(geo_r/geo_r[0], cos(th))
            across0 = across[0] ; delta0 = delta[0]
            #         deltafun = interp1d(geo_r, delta)
            #  delta_s = deltafun(xmean)
            # acrossfun = interp1d(geo_r, across)
            # across_s = acrossfun(xmean)
            BSgamma = (across0/delta0**2)/mdot*rstar / (realxirad/1.5)
            # umag is magnetic pressure
            b12 = 2.*mu30*(rstar*m1/6.8)**(-3) # dipolar magnetic field on the pole, 1e12Gs units
            umag = b12**2*2.29e6*m1
            BSeta = (8./21./sqrt(2.)*umag*3. * (realxirad/1.5))**0.25*sqrt(delta0)/(rstar)**0.125
            print("BSgamma = "+str(BSgamma))
            print("BSeta = "+str(BSeta))
            xs, BSbeta = bs.xis(BSgamma, BSeta, x0=4., ifbeta = True)
            xtmp = (xmean.max()-xmean.min())*arange(100)/double(99)+xmean.min()
            xtmp = xmean
            tth = tscale * rstar**1.5 * m1 * bs.dtint(BSgamma, xtmp, cthfun)
            print(tth)
            # * mdot *  7.* sqrt(rstar * xmean) * delta_s**2/across_s * (1.+2.*across_s/delta_s**2)
            fth = 1./tth
            # 0.0227364 / xirad / sqrt(rstar*xmean)/rstar / mdot * (1./delta_s+2.*delta_s / across_s)**2 * 2.03e5/m1 # Hz
            if (size(unique(xmean))>5):
                goodx = (xmean > 1.*xstd) * (tcenter> tcenter[2])
                pfit, pcov = polyfit(log(xmean[goodx]), log(fmax[goodx]), 1, cov = True)
                print("dln(f)/dln(R_s) = "+str(pfit[0])+"+/-"+str(pcov[0,0]))
            if fosccol is not None:
                print(foscmean)
                fth = foscmean
                fth[xmean > 10.] = sqrt(-1.)
                xtmp = xmean
            plots.errorplot(xmean, xstd, fmax, dfmax, outfile = infile + '_xfmax', xtitle = r'$R_{\rm shock}/R_{*}$', ytitle = '$f$, Hz', yrange = frange, xrange = [maximum(quantile(xmean-xstd, 0.2), 1.), minimum((xmean+xstd)[xmean<xmean.max()].max(), geo_r.max()/rstar)], addline = [xtmp,fth], xlog=True, ylog=False, lticks = [4, 5, 6, 8, 10])
        else:
            plots.errorplot(xmean, xstd, fmax, dfmax, outfile = infile + '_lfmax', xtitle = r'$L/L_{\rm Edd}$', ytitle = '$f$, Hz')
            
#############################################
def fhist(infile = "out/flux"):
    '''
    histogram of flux distribution (reads a two-column ascii file)
    '''
    lines = loadtxt(infile+".dat", comments="#", delimiter=" ", unpack=False)
    t=lines[:,0] ; l=lines[:,1]
    nsize=size(t)
    fn, binedges = histogram(l, bins='auto')
    binc = (binedges[1:]+binedges[:-1])/2.
    bins = (binedges[1:]-binedges[:-1])/2.
    dfn = sqrt(fn)

    medianl = median(l)
    significant = (fn > (dfn*3.)) # only high signal-to-noize points
    w = significant * (binc>(medianl*3.))
    p, cov = polyfit(log(binc[w]), log(fn[w]), 1, w = 1./dfn[w], cov=True)
    print("median flux = "+str(medianl))
    print("best-fit slope "+str(p[0])+"+/-"+str(sqrt(cov[0,0])))

    if ifplot:
        plots.binplot(binedges, fn, dfn, fname=infile+"_hist", fit = exp(p[0]*log(binedges)+p[1]))

#####################################################################
def shock_hdf(n, infile = "out/tireout.hdf5", kleap = 5, uvcheck = False, uvcheckfile = 'uvcheck'):
    '''
    finds the position of the shock in a given entry of the infile
    kleap allows to measure the velocity difference using several cells
    '''
    entryname, t, l, r, sth, rho, u, v, qloss, glo, ediff = hdf.read(infile, n)
    n=size(r)
    #    v=medfilt(v, kernel_size=3)
    # v1=savgol_filter(copy(v), 2*kleap+1, 1) # Savitzky-Golay filter
    #find maximal compression:
    dvdl = (v[kleap:]-v[:-kleap])/(l[kleap:]-l[:-kleap])
    wcomp = (dvdl).argmin()
    wcomp1 = maximum((wcomp-kleap),0)
    wcomp2 = minimum((wcomp+kleap),n-1)
    print("maximal compression found at r="+str((r[wcomp1]+r[wcomp2])/2.)+" +/- "+str((-r[wcomp1]+r[wcomp2])/2.))
    # print("element No "+str(wcomp1)+" out of "+str(n))
    if isnan(r[wcomp]):
        print("t = "+str(t))
        print(dvdl.min(), dvdl.max())
        print(dvdl[wcomp])
        if ifplot:
            plots.someplots(r[1:], [v[1:], v1[1:], dvdl], name = "shocknan", xtitle=r'$r$', ytitle=r'$v$', xlog=False, formatsequence = ['k.', 'r-', 'b'])
        ii=input('r')

    ltot = trapezoid(qloss[1:], x=l[1:])
    if wcomp1 > 2:
        lbelowshock = trapezoid(qloss[1:wcomp1], x = l[1:wcomp1])
        #    wnearshock = maximum(wcomp1-10,1):minimum(wcomp1+10, n-1)
        lonshock = trapezoid(qloss[maximum(wcomp1-10,1):minimum(wcomp1+10, n-1)], x = l[maximum(wcomp1-10,1):minimum(wcomp1+10, n-1)])
    else:
        lbelowshock = 0.
        lonshock = 0.

    if uvcheck:
        # let us check BS's equation (31):
        s= -rho * v
        uv = -u * v
        w = where(uv > 0.)
        w = (r > (r[wcomp1]/2.)) & (r< (r[wcomp2]*1.5))
        rstar = glo['rstar']
        if ifplot:
            plots.someplots(r[w], [uv[w], 0.75*(s/rstar/r)[w]], name = uvcheckfile,
                            xtitle=r'$r$', ytitle=r'$uv$',
                            xlog=False, ylog = True, formatsequence = ['k.', 'b-'],
                            vertical = (r[wcomp1]+r[wcomp2])/2.)
            
    return t, (r[wcomp1]+r[wcomp2])/2.,(-r[wcomp1]+r[wcomp2])/2., v[wcomp1], v[wcomp2], ltot, lbelowshock, lonshock,  -((u*4./3.+v**2/2.)*v)[-1]  

def shock_dat(n, prefix = "out/tireout", kleap = 1):
    '''
    finds the position of the shock and the velocity leap 
    from a given dat-file ID
    kleap allows to measure the velocity difference using several cells
    '''
    fname = prefix + hdf.entryname(n, ndig=5) + ".dat"
    lines = loadtxt(fname, comments="#")
    r = lines[:,0] ; v = lines[:,2]
    #find maximal compression:
    dvdl = (v[1:]-v[:-1])/(r[1:]-r[:-1])
    wcomp = (dvdl).argmin()
    #    print("maximal compression found at r="+str(r[wcomp])+".. "+str(r[wcomp+1])+"rstar")
    return (r[wcomp]+r[wcomp+1])/2., (r[wcomp+1]-r[wcomp])/2.,v[maximum(wcomp-kleap,00)], v[minimum(wcomp+1+kleap, size(r)-1)], 
    
def multishock(n1, n2, dn, prefix = "out/tireout", dat = False, conf = None, kleap = 5, xest = 4.):
    '''
    draws the motion of the shock front with time, for a given set of HDF5 entries or ascii outputs
    '''
    if conf is None:
        conf = 'DEFAULT'
    rstar = config[conf].getfloat('rstar')
    m1 = config[conf].getfloat('m1')
    tscale = config[conf].getfloat('tscale') * m1
    mu30 = config[conf].getfloat('mu30')
    mdot = config[conf].getfloat('mdot') * 4.*pi
    afac = config[conf].getfloat('afac')
    drrat = config[conf].getfloat('drrat')
    realxirad = config[conf].getfloat('xirad')
    b12 = 2.*mu30*(rstar*m1/6.8)**(-3) # dipolar magnetic field on the pole, 1e12Gs units

    n=arange(n1, n2, dn, dtype=int)
    t = zeros(size(n), dtype=double)
    s=arange(size(n), dtype=double)
    ds=arange(size(n), dtype=double)
    dv=arange(size(n), dtype=double)
    v2=arange(size(n), dtype=double)
    v1=arange(size(n), dtype=double)
    lc_out =arange(size(n), dtype=double) # heat advected from the outer edge
    lc_tot = arange(size(n), dtype=double) ; lc_part = arange(size(n), dtype=double)
    lc_nearshock = arange(size(n), dtype=double)
    compression=arange(size(n), dtype=double)
    print(size(n))
    outdir = os.path.dirname(prefix)
    fluxlines = loadtxt(outdir+"/flux.dat", comments="#", delimiter=" ", unpack=False)
    geometry = loadtxt(outdir+"/geo.dat", comments="#", delimiter=" ", unpack=False)
    tf=fluxlines[:,0] ;  ff=fluxlines[:,1]
    th = geometry[:,1] ; r = geometry[:,0]
    cthfun = interp1d(r/r[0], cos(th)) # we need a function allowing to calculate cos\theta (x)
    across0 = geometry[0,3]  ;   delta0 = geometry[0,5]
    acrosslast =  geometry[-1,3]
    #   acrosslast = pi*4.*afac*drrat * geometry[-1,0]**2 
    BSgamma = (across0/delta0**2)/mdot*rstar / (realxirad/1.5)
    umag = b12**2*2.29e6*m1 * (1.+3.*cos(th[0])**2)/4.
    # umag is magnetic pressure
    BSeta = (8./21./sqrt(2.) * umag * 3. * (realxirad/1.5))**0.25*sqrt(delta0)/(rstar)**0.125
    xs, BSbeta = bs.xis(BSgamma, BSeta, x0=xest, ifbeta = True)
    print("eta = "+str(BSeta))
    print("gamma = "+str(BSgamma))
    print("eta gamma^{1/4} = "+str(BSeta*BSgamma**0.25))
    print("beta = "+str(BSbeta))
    print("predicted shock position = "+str(xs))
    # dt_BS = tscale * rstar**1.5 * bs.dtint(BSgamma, xs, cthfun)
    ii = input("xs")
    # spherization radius
    rsph =1.5*mdot/4./pi
    eqlum = mdot/rstar*(1.-BSbeta)
    if size(eqlum) > 0:
        eqlum = eqlum[0]
    print("m1 = "+str(m1))
    print("mdot = "+str(mdot))
    print("rstar = "+str(rstar))
    # iterating to find the cooling radius
    rcool = rcoolfun(geometry, mdot)
    for k in arange(size(n)):
        if(dat):
            stmp, dstmp, v1tmp, v2tmp = shock_dat(n[k], prefix=prefix, kleap = kleap)
        else:
            ttmp, stmp, dstmp, v1tmp, v2tmp, ltot, lpart, lshock, uvtmp = shock_hdf(n[k], infile = prefix+".hdf5", kleap = kleap, uvcheck = (k == (size(n)-1)), uvcheckfile = outdir+"/uvcheck")
        s[k] = stmp ; ds[k] = dstmp
        v1[k] = v1tmp   ; v2[k] =  v2tmp
        dv[k] = v1tmp - v2tmp
        compression[k] = v2tmp/v1tmp
        lc_tot[k] = ltot ; lc_part[k] = lpart ; lc_nearshock[k] = lshock
        t[k] = ttmp
        lc_out[k] = uvtmp * acrosslast

    print("predicted shock position: xs = "+str(xs)+" (rstar)")
    #    print("cooling limit: rcool/rstar = "+str(rcool/rstar))
    print("flux array size "+str(size(lc_tot)))
    ff /= 4.*pi  ; eqlum /= 4.*pi ; lc_tot /= 4.*pi ; lc_part /= 4.*pi  ; lc_out /= 4.*pi ; lc_nearshock /= 4.*pi
    t *= tscale

    dt_current = tscale * rstar**1.5 * m1 * bs.dtint(BSgamma, s, cthfun)
    
    if(ifplot):
        ws=where((s>1.0) & (lc_part > lc_part.min()))
        n=ws
        plots.someplots(t[ws], [lc_tot[ws], lc_part[ws], lc_out[ws], lc_tot[ws]-lc_part[ws], t[ws]*0.+mdot/rstar/4./pi*(1.-BSbeta), t[ws]*0.+mdot/rstar/4./pi, lc_tot[ws]-mdot/rstar/4./pi*(1.-BSbeta)], name = outdir+"/lumshocks", xtitle=r'$t$, s', ytitle=r'$L/L_{\rm Edd}$', formatsequence = ['k-', 'r-', 'b:', 'g-.', 'k:', 'k--', 'm-'], legendsequence = [r'$L_{tot}$', r'$L_{X}$', r'$L_{out}$', r'$L_{tot}-L_{X}$', r'$(1-\beta_{\rm BS})L_{acc}$', r'$L_{acc}$', r'$L_{tot}-(1-\beta_{\rm BS})L_{acc}$'],inchsize = [5, 4], ylog = True)
        plots.someplots(t[ws], [s[ws], s[ws]*0.+xs], name = outdir+"/shockfront", xtitle=r'$t$, s', ytitle=r'$R_{\rm shock}/R_*$', xlog=False, formatsequence = ['k-', 'r-', 'b:'], vertical = t.max()*0.9, verticalformatsequence = 'b:', inchsize = [5,4])        
        plots.someplots(lc_part[ws], [s[ws], s[ws]*0.+xs], name=outdir+"/fluxshock", xtitle=r'$L/L_{\rm Edd}$', ytitle=r'$R_{\rm shock}/R_*$', xlog= (lc_tot[ws].max()/median(lc_tot[ws])) > 10., ylog= (s[ws].max()/s[ws].min()> 10.), formatsequence = ['k-', 'r-', 'b-'], vertical = eqlum, verticalformatsequence = 'r-', inchsize = [5,4])
        # plots.someplots(t[ws], [ff[n], lc_part[ws], ff[n]*0.+eqlum], name = outdir+"/flux", xtitle=r'$t$, s', ytitle=r'$L/L_{\rm Edd}$', xlog=False, ylog=False, formatsequence = ['k:', 'k-', 'r-'])
        plots.someplots(t[ws], [-v1[ws], -v2[ws], sqrt(2./s[ws]/rstar), sqrt(2./s[ws]/rstar)/7.], name = outdir+"/vleap",xtitle=r'$t$, s', ytitle=r'$ v /c$', xlog=False, formatsequence = ['k-', 'b:', 'r-', 'r-'])
        plots.someplots(s[ws], [1./dt_current[ws], 1./(tscale * rstar**1.5 * s[ws]**3.5)], xtitle = r'$R_{\rm shock}/R_*$', xlog = True, ylog = True, formatsequence = ['ro', 'k-'], name = outdir + '/ux', ytitle = r'$f$, Hz')

    print("effective compression factor "+str(compression[isfinite(compression)].mean()))
    # ascii output
    fout = open(outdir+'/sfront.dat', 'w')
    #    foutflux = open(outdir+'/sflux.dat', 'w')
    fout.write("# time -- shock position -- downstream velocity -- upstream velocity -- total flux -- partial flux -- osc. freq.\n")
    for k in arange(size(n)):
        fout.write(str(t[k])+" "+str(s[k])+" "+str(v1[k])+" "+str(v2[k])+" "+str(lc_tot[k])+" "+str(lc_part[k])+" "+str(1./dt_current[k])+"\n")
    fout.close()
    fglo = open(outdir + '/sfrontglo.dat', 'w') # BS shock position and equilibrium flux
    fglo.write('# equilibrium luminosity -- BS shock front position / rstar \n')

    if isscalar(xs):
        fglo.write(str(eqlum)+' '+str(xs)+'\n')
    else:
        fglo.write(str(eqlum)+' '+str(xs[0])+'\n')
    fglo.close()
    # last 10% average shock position
    wlaten = where(t > (t.max()*0.9))
    wlate = where(tf > (tf.max()*0.9))
    xmean = s[wlaten].mean() ; xrms = s[wlaten].std()+ds[wlaten].mean()
    print("s/RNS = "+str(xmean)+"+/-"+str(xrms)+"\n")
    fmean = ff[wlate].mean() ; frms = ff[wlate].std()
    print("flux = "+str(fmean)+"+/-"+str(frms)+"\n")
    print("total flux = "+str(lc_tot[wlaten].mean())+"+/-"+str(lc_tot[wlaten].std())+"\n below the shock: "+str(lc_part[wlaten].mean())+"+/-"+str(lc_part[wlaten].std())+"\n")
    print("lc_out = "+str(lc_out[wlaten].mean())+"\n")
        
###############################

# power-law fit to the decay tail
def tailfitfun(x, p, n, x0, y0):
    return ((x-x0)**2)**(p/2.)*n+y0

# exponential tail to the decay tail
def tailexpfun(x, xdec, n, x0, y0):
    return exp(-abs(x-x0)/xdec)*n+y0

def tailfit(prefix = 'out/flux', trange = None, ifexp = False, ncol = -1):
    '''
    fits the later shape of the light curve (or shock front position, if prefix is a sfront file, and ncol = 1), assuming it approaches equilibrium. 
    trange sets the range of time (s), ifexp switches between two fitting functions (power-law and exponential), ncol sets the column of the data file to be used. 
    '''
    fluxlines = loadtxt(prefix+".dat", comments="#", delimiter=" ", unpack=False)
    
    t=fluxlines[:,0] ; f=fluxlines[:,ncol]
    if(trange is None):
        t1 = t; f1 = f
    else:
        t1=t[(t>trange[0])&(t<trange[1])]
        f1=f[(t>trange[0])&(t<trange[1])]
    if ifexp:
        par, pcov = curve_fit(tailexpfun, t1, f1, p0=[t1.max()/5.,f1.max()-f1.min(), 0., f1.min()])
    else:
        par, pcov = curve_fit(tailfitfun, t1, f1, p0=[-2.,5., trange[0], f1.min()])
    if ifplot:
        if ifexp:
            plots.someplots(t, [f, tailexpfun(t, par[0], par[1], par[2], par[3]), t*0.+par[3]], name = prefix+"_fit", xtitle=r'$t$, s', ytitle=r'$L$', xlog=False, ylog=False, formatsequence=['k.', 'r-', 'g:'], yrange=[f[f>0.].min(), f.max()])
            if par[1]>0.:
                plots.someplots(t, [f-par[3], tailexpfun(t, par[0], par[1], par[2], 0.)], name = prefix+"_dfit", xtitle=r'$t$, s', ytitle=r'$\Delta L$', formatsequence=['k.', 'r-'], ylog = True, xlog = False)
            else:
                plots.someplots(t, [par[3]-f, -tailexpfun(t, par[0], par[1], par[2], 0.)], name = prefix+"_dfit", xtitle=r'$t$, s', ytitle=r'$\Delta L$', formatsequence=['k.', 'r-'], ylog = True, xlog = False)
        else:
            plots.someplots(t, [f, tailfitfun(t, par[0], par[1], par[2], par[3]), t*0.+par[3]], name = prefix+"_fit", xtitle=r'$t$, s', ytitle=r'$L$', xlog=False, ylog=False, formatsequence=['k.', 'r-', 'g:'], yrange=[f[f>0.].min(), f.max()])
            plots.someplots(t, [f-par[3], tailfitfun(t, par[0], par[1], par[2], 0.)], name = prefix+"_dfit", xtitle=r'$t$, s', ytitle=r'$\Delta L$', formatsequence=['k.', 'r-'], ylog = True, xlog = False)
    
    if ifexp:
        print("decay time = "+str(par[0])+"+/-"+str(sqrt(pcov[0,0])))
    else:
        print("slope ="+str(par[0])+"+/-"+str(sqrt(pcov[0,0])))
    print("y0 ="+str(par[3])+"+/-"+str(sqrt(pcov[3,3])))
    print("norm ="+str(par[1])+"+/-"+str(sqrt(pcov[1,1])))
       
def taus(n, prefix = 'out/tireout', ifhdf = True, conf = 'DEFAULT'):
    '''
    calculates the optical depths along and across the flow
    '''

    rstar = config[conf].getfloat('rstar')
    m1 = config[conf].getfloat('m1')
    mu30 = config[conf].getfloat('mu30')
    mdot = config[conf].getfloat('mdot') * 4.*pi
    afac = config[conf].getfloat('afac')
    realxirad = config[conf].getfloat('xirad')

    geofile = os.path.dirname(prefix)+"/geo.dat"
    print(geofile)
    r, theta, alpha, across, l, delta = geo.gread(geofile) 
    if(ifhdf):
        hname = prefix + ".hdf5"
        entryname, t, l, r, sth, rho, u, v, qloss, glo = hdf.read(hname, n)
    else:
        lines = loadtxt( os.path.dirname(prefix) + '/avprofile.dat', comments = '#')
        r = lines[:,0] ; v = lines[:,1] ; u = lines[:,2] ; prat = lines[:,3] ; temp = lines[:,4] ; rho = lines[:,5]
        # entryname = hdf.entryname(n, ndig=5)
        # fname = prefix + entryname + ".dat"
        # lines = loadtxt(fname, comments="#")
        # rho = lines[:,1]
    taucross = delta * rho  # across the flow
    dr = (r[1:]-r[:-1])  ;   rc = (r[1:]+r[:-1])/2.
    taualong = (rho[1:]+rho[:-1])/2. * dr
    taucrossfun = interp1d(r, taucross, kind = 'linear')
    if(ifplot):
        plots.someplots(rc-1., [taucrossfun(rc), taualong], name = prefix+"_tau", xtitle=r'$r/R_{\rm NS}-1$', ytitle=r'$\tau$', xlog=True, ylog=True, formatsequence=['k-', 'r-'])
        plots.someplots(rc-1., [2./(rho[1:]+rho[:-1])/rc/rstar, (delta[1:]+delta[:-1])/2./rc/rstar, dr/rc], name = prefix+"_d", xtitle=r'$r/R_{\rm NS}-1$', ytitle=r'$\delta/R$, $1/\varkappa \rho R$', xlog=True, ylog=True, formatsequence=['k-', 'r--', 'b:'])
    else:
        # ASCII output
        fout = open(os.path.dirname(prefix)+'/tauprofile.dat', 'w')
        fout.write("# R  -- tperp -- tparallel \n")
        nx = size(rc)
        for k in arange(nx):
            s = str(rc[k]) + " " + str(taucrossfun(rc[k])) + " " + str(taualong[k]) + "\n"
            print(s)
            fout.write(s)
            fout.flush()
        fout.close()

def filteredflux(hfile, n1, n2, rfraction = 0.9, conf = 'DEFAULT'):
    '''
    calculates the flux excluding several outer points affected by the outer BC
    hfile is the input HDF5 file
    n1 is the number of the first entry
    n2 is the last one
    rfraction is the rangle of radii where the flux is being calculated
    '''
    geofile = os.path.dirname(hfile)+"/geo.dat"
    r, theta, alpha, across, l, delta = geo.gread(geofile)
    wr = r < (r.max()*rfraction)

    tscale = config[conf].getfloat('tscale') * config[conf].getfloat('m1')

    lint = zeros(n2-n1)
    ltot = zeros(n2-n1)
    tar = zeros(n2-n1)
    
    for k in arange(n2-n1)+n1:
        entryname, t, l, r, sth, rho, u, v, qloss, glo = hdf.read(hfile, k)
        mdot = glo['mdot']
        lint[k] = simps(qloss[wr], x=l[wr])
        ltot[k] = simps(qloss, x=l)
        tar[k] = t
    ltot /= 4.*pi ; lint /= 4.*pi # convert to Eddington units
    tar *= tscale
    if(ifplot):
        # overplotting with the total flux
        plots.someplots(tar, [lint, ltot, ltot-lint, lint*0.+mdot*0.2], xlog=False, formatsequence = ['k-', 'g--', 'b--', 'r-'], xtitle='t, s', ytitle=r'$L/L_{\rm Edd}$', name= os.path.dirname(hfile)+'/cutflux')

    # ascii output:
    fout = open(os.path.dirname(hfile)+"/cutflux.dat", "w")
    for k in arange(n2-n1)+n1:
        fout.write(str(tar[k])+" "+str(lint[k])+" "+str(ltot[k])+"\n")
        fout.flush()
    fout.close()
    # flux during the last 10% of the curve:
    w = (tar > (tar.max()*0.9))
    ffilteredmean = lint[w].mean()
    fmean = ltot[w].mean()
    fstd = ltot[w].std()
    print("Fmean = "+str(fmean)+"+/-"+str(fstd)+" ("+str(fmean-ffilteredmean)+")")
        
def lplot():

    prefices = ['titania_mdot1', 'titania_mdot3', 'titania_fidu', 'titania_mdot30', 'titania_dhuge', 'titania_nod']

    confs = ['M1', 'M3', 'FIDU', 'M30', 'DHUGE', 'NOD']

    xests = [1.5, 2., 4., 8., 4., 4.]
    
    npref = size(prefices)

    lrad = zeros(npref) ; dlrad = zeros(npref) ; ltot = zeros(npref) ; gammas = zeros(npref) ; betas = zeros(npref)

    for k in arange(npref):
        rstar = config[confs[k]].getfloat('rstar')
        m1 = config[confs[k]].getfloat('m1')
        mu30 = config[confs[k]].getfloat('mu30')
        mdot = config[confs[k]].getfloat('mdot') * 4.*pi
        afac = config[confs[k]].getfloat('afac')
        realxirad = config[confs[k]].getfloat('xirad')
        b12 = 2.*mu30*(rstar*m1/6.8)**(-3) # dipolar magnetic field on the pole, 1e12Gs units
        umag = b12**2*2.29e6*m1
        geometry = loadtxt(prefices[k]+"/geo.dat", comments="#", delimiter=" ", unpack=False)
        across0 = geometry[0,3]  ;   delta0 = geometry[0,5]
        BSgamma = (across0/delta0**2)/mdot*rstar / (realxirad/1.5)
        # umag is magnetic pressure
        BSeta = (8./21./sqrt(2.)*umag*3. * (realxirad/1.5))**0.25*sqrt(delta0)/(rstar)**0.125
        xs, BSbeta = bs.xis(BSgamma, BSeta, x0=xests[k], ifbeta = True)

        frontfile = prefices[k]+'/sfront'
        frontlines = loadtxt(frontfile+'.dat', comments="#", delimiter=" ", unpack=False)

        f = frontlines[:,5] # partial flux from below the shock
        nt = size(f)
        lrad[k] = f[round(nt*0.9):].mean()
        dlrad[k] = f[round(nt*0.9):].std()
        ltot[k] = mdot * 0.205 / 4./pi
        gammas[k] = BSgamma
        betas[k] = BSbeta
        print(ltot[k])

    plots.errorplot(1./gammas, gammas*0., lrad/ltot, dlrad/ltot, outfile = 'bsfig2',
                    xtitle = r'$\gamma^{-1}$', ytitle = r'$L_{\rm s}/L_{\rm tot}$',
                    addline  = 1.-betas, xlog = True, pointlabels = confs)


def energytest(infile, n1, n2, dn, conf = 'DEFAULT'):
    '''
    tracks the evolution of different types of energy
    '''
    geofile = os.path.dirname(infile)+"/geo.dat"
    r, theta, alpha, across, l, delta = geo.gread(geofile)

    n = arange(n1, n2, dn)
    nt = size(n)
    
    etot = zeros(nt) ;    ekin = zeros(nt) ;    epot = zeros(nt)
    eheat = zeros(nt) ; tar = zeros(nt) ; ltot = zeros(nt)
    mass = zeros(nt)
    
    for k in arange(nt):
        entryname, t, l, r, sth, rho, u, v, qloss, glo = hdf.read(infile, n[k])
        ekin[k] = trapezoid(rho * v**2/2. * across, x = l)
        epot[k] = -trapezoid(rho / r * across, x = l)
        eheat[k] = trapezoid(u * across, x = l)
        etot[k] = ekin[k] + epot[k] + eheat[k]
        tar[k] = t
        ltot[k] = trapezoid(qloss, x=l)
        mass[k] = trapezoid(rho * across, x = l)

    llost = cumulative_trapezoid(ltot, x=tar, initial = 0.)
        
    m1 = config[conf].getfloat('m1')
    tscale = config[conf].getfloat('tscale') * m1
    tar *= tscale
    plots.someplots(tar, [etot, ekin, epot, eheat, etot+llost], 
                    name = os.path.dirname(infile)+'/energytest',
                    formatsequence = ['k-', 'b--', 'r:', 'g-.', 'm--'],
                    xtitle = r'$t$, s', ytitle = r'$E$')
    plots.someplots(mass, [etot, ekin, epot, eheat, llost], 
                    name = os.path.dirname(infile)+'/energytest_m',
                    formatsequence = ['k-', 'b--', 'r:', 'g-.', 'm--'],
                    xtitle = r'$M$', ytitle = r'$E$')

    dedt = (etot[1:]-etot[:-1])/(tar[1:]-tar[:-1]) * tscale
    dedt_p = (epot[1:]-epot[:-1])/(tar[1:]-tar[:-1]) * tscale

    plots.someplots(tar[1:], [-dedt, ltot[1:], dedt_p, dedt_p+ltot[1:]-dedt], formatsequence = ['k-', 'b--', 'k:', 'r:'], 
                    name = os.path.dirname(infile)+'/energytest_d', xlog = False, 
                    xtitle = r'$t$, s', ytitle = r'$dE/dt$', yrange = [-ltot.max(), ltot.max()*2.])

def masstest(indir, conf='DEFAULT'):

    mass1 = config[conf].getfloat('m1')
    massscale = config[conf].getfloat('massscale') * mass1
    mdot = config[conf].getfloat('mdot')

    massfile = indir + '/totals.dat'
    
    masslines = loadtxt(massfile, comments="#", delimiter=" ", unpack=False)
    
    t=masslines[:,0] ; m=masslines[:,1] ; mlost=masslines[:,3] ; macc=masslines[:,4] ; mdotcurrent = -masslines[:,5]

    plots.someplots(t, [m-m[0], macc-mlost-(macc[0]-mlost[0]), (macc-macc[0])*1e-2, (mlost-mlost[0])*1e-2], name = 'mbalance', formatsequence = ['k-', 'r:', 'b--', 'm-', 'c-'],
                    xlog = False, ylog = False, xtitle = r'$t$, s', ytitle = r'$M$')
    print("mass change / mass injected = "+str((m[-1]-m[0])/(macc[-1]-macc[0])))
    
    print(mdot)
    print("mdot/mdotd = ", mdotcurrent[400:].mean()/mdot/4./pi, "+/-", mdotcurrent[400:].std()/mdot/4./pi)
    
    # plots.someplots((t[1:]+t[:-1])/2., [(mlost[1:]-mlost[:-1])/(t[1:]-t[:-1])], name='mdot', xtitle = r'$t$, s', ytitle = r'$\dot{M}$, g s$^{-1}$', xlog = False, ylog = False)

    plots.someplots(t, [mdotcurrent/4./pi, macc*0.+mdot], name='mdot', xtitle = r'$t$, s', ytitle = r'$\dot{M}c^2/L_{\rm Edd}$', xlog = False, ylog = False, formatsequence=['k-','r:'])

def quasi2d_nocalc(infile, conf = 'DEFAULT', trange = None):
    
    outdir = os.path.dirname(infile)

    mdot = config[conf].getfloat('mdot')
    mass1 = config[conf].getfloat('m1')

    lines = loadtxt(infile, comments="#", delimiter=" ", unpack=False)
    
    t = lines[:,0] ;  r = lines[:,1] ;  v = lines[:,2] ;  lurel = lines[:,3];  m = lines[:,4]

    sl = shape(lines)
    if sl[1]>5:
        q = lines[:,5] #  ; qd = lines[:,6]
        q *= r**3
        

    nt = size(unique(t)) ; nr = size(unique(r))
    
    t = unique(t) ; r = unique(r)
    v = reshape(v, [nt, nr]) ;  lurel = reshape(lurel, [nt, nr]) ;  m = reshape(m, [nt, nr])
    if sl[1]> 5:
        q = reshape(q, [nt, nr])
    u = 10.**lurel
        
    if ifplot:
        nv = 30
        #  plots.somemap(r, t, log10(4.*pi *q*r**2/mdot), name=outdir+'/q2d_q', inchsize = [4, 12], cbtitle = r'$R^2 Q^- / L_{\rm Edd}$', xrange = trange, transpose=True) #, levels = 3.*arange(nv)/double(nv-2)-1.)
        plots.somemap(r, t, v, name=outdir+'/q2d_v', inchsize = [5, 10], cbtitle = r'$v/c$',  xrange = trange,transpose=True, levels = 0.125 * (arange(nv)/double(nv-1)-0.7))
        plots.somemap(r, t, lurel - log(3.), name=outdir+'/q2d_u', inchsize = [5, 10], cbtitle = r'$\log_{10} p/p_{\rm mag}$', xrange = trange, addcontour = [u/3./0.99, u/3./0.9, u/3./0.8],transpose=True)
        plots.somemap(r, t, m, name=outdir+'/q2d_m', inchsize = [5, 10], cbtitle = r'$s / \dot{M}$', xrange = trange, transpose=True, levels = 3.*arange(nv)/double(nv-2)-1.)
        teff = 4.75 * mass1**(-0.25) * (q/r**3)**0.25 # keV
        plots.somemap(r, t, q, name=outdir+'/q2d_q', inchsize = [5, 10], cbtitle = r'$Q R^3$', xrange = trange, transpose=True)
        plots.somemap(r, t, teff, name=outdir+'/q2d_teff', inchsize = [5, 10], cbtitle = r'$Teff$, keV', xrange = trange, transpose=True)

#############################################
def quasi2d(hname, n1, n2, conf = 'DEFAULT', step = 1, kleap = 5, trange = None, iflog=False, ifnu=False):
    '''
    makes quasi-2D Rt plots or an RT table
    '''
    outdir = os.path.dirname(hname)
    
    betafun = betafun_define() # defines the interpolated function for beta

    nt=int(floor((n2-n1)/step))
    # geometry:
    geofile = outdir+"/geo.dat"
    print(geofile)
    r, theta, alpha, across, l, delta = geo.gread(geofile)
    perimeter = 2. * (across/delta + 2.*delta)
    # first frame
    entryname, t, l, r, sth, rho, u, v, qloss, glo, ediff = hdf.read(hname, n1)
    cth = sqrt(1.-sth**2)

    rstar = glo['rstar']
    umag = glo['umag']
    
    m1 = config[conf].getfloat('m1')
    tscale = config[conf].getfloat('tscale')*m1
    rstar = config[conf].getfloat('rstar')
    mu30 = config[conf].getfloat('mu30')
    mdot = config[conf].getfloat('mdot') * 4.*pi
    afac = config[conf].getfloat('afac')
    realxirad = config[conf].getfloat('xirad')
    mow = config[conf].getfloat('mow')
    b12 = 2.*mu30*(rstar*m1/6.8)**(-3) # dipolar magnetic field on the pole, 1e12Gs units
    umag1 = b12**2*2.29e6*m1
    print("mdot = "+str(mdot))
    # ii = input("M")
    print("Umag = "+str(umag)+" = "+str(umag1)+"\n")
    betacoeff = config[conf].getfloat('betacoeff') * (m1)**(-0.25)/mow
    
    mcol = across[0] * rstar**2 * umag / m1 * (1.+3.*cth[0]**2)/4.
    tr = mcol / mdot * tscale
    print("tr = "+str(tr))
    # ii =input('tr')
    
    nr=size(r)
    nrnew = 300 # radial mesh interpolated to nrnew
    rnew = (r.max()/r.min())**(arange(nrnew)/double(nrnew-1))*r.min()
    sthfun = interp1d(r, sth)
    sthnew = sthfun(rnew)
    umagtar = umag * (1.+3.*(1.-sth**2))/4. / (r)**6 # r is already in rstar units
    umagtarnew = umag * (1.+3.*(1.-sthnew**2))/4. / (rnew)**6 # r is already in rstar units
    var = zeros([nt, nrnew], dtype=double)
    uar = zeros([nt, nrnew], dtype=double)
    par = zeros([nt, nrnew], dtype=double)
    betar = zeros([nt, nrnew], dtype = double) # this is pgas/ptot
    qar = zeros([nt, nrnew], dtype=double)
    ear = zeros([nt, nrnew], dtype=double)
    lurel = zeros([nt, nrnew], dtype=double)
    mdar = zeros([nt, nrnew], dtype=double)
    tar = zeros(nt, dtype=double)
    rvent = zeros(nt, dtype=double)
    maxprat = zeros(nt, dtype=double)
    drvent = zeros(nt, dtype=double)
    rshock = zeros(nt, dtype=double)
    betavent = zeros(nt, dtype = double) # this one is BS's beta
    betaeff = zeros(nt, dtype = double) # and this one, too
    betaeff_m = zeros(nt, dtype = double) # this, too
    if ifnu:
        qnuA = zeros([nt, nrnew], dtype = double)
        qnuPh = zeros([nt, nrnew], dtype = double)
        qnuPl = zeros([nt, nrnew], dtype = double)
    #    var[0,:] = v[:] ; uar[0,:] = u[:] ; tar[0] = t
    for k in arange(nt):
        if ifnu:
            entryname, t, l, r, sth, rho, u, v, qloss, glo, ediff, qnuA_1, qnuPh_1, qnuPl_1 = hdf.read(hname, k*step+n1, ifnu=True) # hdf5 read
            qafun = interp1d(r, qnuA_1, kind = 'linear')
            qphfun = interp1d(r, qnuPh_1, kind = 'linear')
            qplfun = interp1d(r, qnuPl_1, kind = 'linear')
            qnuA[k, :] = qafun(rnew)
            qnuPh[k, :] = qphfun(rnew)
            qnuPl[k, :] = qphfun(rnew)
        else:
            entryname, t, l, r, sth, rho, u, v, qloss, glo, ediff = hdf.read(hname, k*step+n1) # hdf5 read
        vfun = interp1d(r, v, kind = 'linear')
        var[k, :] = vfun(rnew)
        qfun = interp1d(r, qloss/perimeter, kind = 'linear')
        qar[k, :] = qfun(rnew)
        # print("size(ediff) = ",size(ediff))
        efun = interp1d(r, ediff, kind = 'linear')
        ear[k, :] = efun(rnew)
        ufun = interp1d(r, u, kind = 'linear')
        # print(u.min(), u.max())
        # ii = input('U')
        uar[k, :] = ufun(rnew)
        beta = betafun(Fbeta(rho, u, betacoeff))
        press = u/3./(1.-beta/2.)
        pfun = interp1d(r, press, kind = 'linear')
        par[k, :] = pfun(rnew)
        betar[k, :] = 2.*(1.-uar[k,:]/par[k,:]/3.)
        mfun = interp1d(r, -rho * v * across, kind = 'linear')
        wloss = (press>(0.9*umagtar))
        shock = ((v)[kleap:]-(v)[:-kleap]).argmin()
        rshock[k] = r[shock+kleap] #+r[minimum(shock+kleap+1, nt-1)])/2.
        if wloss.sum()> 1:
            # imfun = interp1d((-rho * v * across)[wloss], r[wloss], kind = 'linear', bounds_error=False, fill_value=NaN)
            # rvent[k] = imfun(mdot/2.)
            wvent = (press/umagtar)[(r>r.min())&(r<rshock[k])].argmax()
            if wvent >= (nr-1):
                print(nr)
                wvent -= 1
            rvent[k] = r[wvent]
            drvent[k] = r[wvent+1]-r[wvent]
            maxprat[k] = (press/umagtar)[(r>r.min())&(r<rshock[k])].max()
            betavent[k] = ((u+press)/rho)[wvent] * rstar
            if (rvent[k] >= 1.) & (maxprat[k] >= 1.):
                print("rvent = "+str(rvent[k])+" = "+str(r[wvent]))
                print("drvent = "+str(drvent[k]))
                print("t = "+str(t*tscale))
                # ii = input('R')
            #            print("maxprat = "+str(maxprat[k]))
        # effective BS's beta:
        betaeff[k] = ((u+press)/rho)[0:1].mean() * rstar
        betaeff_m[k] = (umagtar/rho)[0:1].mean() * rstar * 4.
        #  print(rstar)
        mdar[k, :] = mfun(rnew)
        # print("mdot[-1] = "+str(mdar[k,-1]))
        tar[k] = t
    nv=30
    vmin = round(var.min(),2)
    vmax = round(var.max(),2)
    vmin = maximum(vmin, -1.) ; vmax = minimum(vmax, 1.)
    vlev=linspace(vmin, vmax, nv, endpoint=True)
    print(var.min())
    print(var.max())
    varmean = var.mean(axis=0)
    varstd = var.std(axis=0)

    # velocity
    if ifplot:
        plots.somemap(rnew, tar*tscale, var, name=outdir+'/q2d_v', levels = vlev, inchsize = [4, 12], cbtitle = r'$v/c$', transpose = True, xrange = trange, ylog=iflog, xlog=True)
        plots.someplots(rnew, [-sqrt(1./rstar/rnew), rnew*0., varmean, varmean+varstd, varmean-varstd], formatsequence = [':k', '--k', '-k', '-g', '-g'], xlog = True, ylog = False, xtitle = r'$R/R_{\rm *}$', ytitle = r'$\langle v\rangle /c$', inchsize = [3.35, 2.], name=outdir+'/q2d_vmean')
        plots.someplots(tar*tscale, [rvent], name = outdir+'/rvent', xtitle = r'$t$, s', ytitle = r'$R/R_*$', xlog=iflog)

    # internal energy
    #    print(umag)
    for k in arange(nrnew):
        lurel[:,k] = log10(uar[:,k]/umagtarnew[k])
    if ifplot:
        umin = round(lurel[uar>0.].min(),2)
        umax = round(lurel[uar>0.].max(),2)
        lulev = linspace(umin, umax, nv, endpoint=True)
        print(lulev)
        plots.somemap(rnew, tar*tscale, lurel, name=outdir+'/q2d_u', levels = lulev, \
                inchsize = [4, 12], cbtitle = r'$\log_{10}u/u_{\rm mag}$', \
                addcontour = [par/umagtarnew/1., par/umagtarnew/0.9,
                par/umagtarnew/0.8], transpose = True, xrange = trange, ylog=iflog, xlog=True)
        plots.somemap(rnew, tar*tscale, log10(betar), name=outdir+'/q2d_b',
                inchsize = [4, 12], cbtitle = r'$\log_{10}\beta$', transpose = True, xrange = trange, xlog=iflog, ylog=True)
        # Q-:
        plots.somemap(rnew, tar*tscale, log10(qar), name=outdir+'/q2d_q', \
                inchsize = [4, 12], cbtitle = r'$\log_{10}Q$', transpose = True, xrange = trange, ylog=iflog, xlog=True)
        plots.somemap(rnew, tar*tscale, log10(abs(ear)), name=outdir+'/q2d_qe', \
                inchsize = [4, 12], cbtitle = r'$\log_{10}\left|F_{\rm diff}\right|$', transpose = True, xrange = trange, xlog = False, ylog=iflog)
        # mdot:
        mdlev = 3.*arange(nv)/double(nv-2)-1.
        plots.somemap(rnew, tar*tscale, mdar/mdot, name=outdir+'/q2d_m', \
                inchsize = [4, 12], cbtitle = r'$s / \dot{M}$', levels = mdlev, \
                transpose = True, xrange = trange, ylog=False, xlog=True) # , yrange=[1.,1.1], ylog = False)

        # mean mdar
        mdmean = mdar.mean(axis=0)
        mdstd = mdar.std(axis=0)

        plots.someplots(rnew, [rnew*0.+1., rnew*0., mdmean/mdot, (mdmean+mdstd)/mdot, (mdmean-mdstd)/mdot], formatsequence = [':k', '--k', '-k', '-g', '-g'], xlog = True, ylog = False, xtitle = r'$R/R_{\rm *}$', ytitle = r'$\langle s\rangle /\dot{M}$', inchsize = [3.35, 2.], name=outdir+'/q2d_mdmean')
 
        plots.someplots(tar*tscale, [betaeff, betaeff_m, betavent], xtitle=r'$t$, s', ytitle=r'$\frac{u+P}{\rho}\frac{R_*}{GM_*}$', formatsequence=['k.', 'r-', 'b:'], ylog = False, xlog = False,
            name=outdir+"/betaeff", yrange = [0., betaeff.max()*1.1])
        print("mean effective betaBS = "+str(betaeff.mean()))
        print("using magnetic energy, betaBS = "+str(betaeff_m.mean()))
        print("gas-to-total pressure ratio at the surface is "+str(betar[tar>0.9*tar.max(),0].mean()))
        if ifnu:
            plots.somemap(rnew, tar*tscale, log10(qnuA+qnuPh+qnuPh), name=outdir+'/q2d_nua', inchsize = [4, 12], cbtitle = r'$\log_{10}Q_{\nu}$, relative units', transpose = True, xrange = trange, ylog=False, xlog=True) # , yrange=[1.,1.1], ylog = False)
        
    # let us also make an ASCII table
    ftable = open(outdir+'/ftable.dat', 'w')
    ftable.write('# format: t[s] -- r/rstar -- rho -- v -- u/umag -- s/mdot -- Q- -- Qdiff \n')
    for kt in arange(nt):
        for kr in arange(nrnew):
            ftable.write(str(tar[kt]*tscale)+' '+str(rnew[kr])+' '+str(var[kt, kr])+' '+str(lurel[kt, kr])+' '+str(mdar[kt, kr]/mdot)+' '+str(qar[kt,kr])+' '+str(ear[kt,kr])+'\n')
    ftable.flush()
    ftable.close()

    # rvent output
    frvent = open(outdir+'/rvent.dat', 'w')
    frvent.write('# format: t[s] -- rvent/rstar\n')
    for kt in arange(nt):
        if rvent[kt]>1.:
            frvent.write(str(tar[kt]*tscale)+' '+str(rvent[kt])+'\n')
    frvent.flush()
    frvent.close()

#####################################
        
def diffuseplot(flist, nlist, rrange = [2.86, 2.9]):
    '''
    reads several outputs and makes the plots of radiation fluxes and diffusive photon fluxes along the field line
    '''

    nf = size(flist)
    
    rs = [] # radial coords
    qs = [] # radiation flux (outcoming)
    ds = [] # diffusion flux (along the line)
    vs = []
    csqs = []
    taulist = [] # radial optical depth
    
    for k in arange(nf):
        entryname, t, l, r, sth, rho, u, v, qloss, glo, ediff = hdf.read(flist[k], nlist[k])
        wfront = (v[1:]-v[:-1]).argmin()
        wfront += 1
        taucoord = cumulative_trapezoid(rho, x=l, initial = 0.)
        taucoord -= taucoord[wfront]
        geofile = os.path.dirname(flist[k])+"/geo.dat"
        r1, theta, alpha, across, l1, delta = geo.gread(geofile)
        perimeter = 2. * (across/delta + 2. * delta)
        rs.append(r)
        qs.append(qloss / perimeter)
        ds.append(ediff)
        taulist.append(taucoord)
        vs.append(v)
        csqs.append(u/rho)
        
    taufun = interp1d(r, taucoord)
    taurange = [taufun(rrange[0]), taufun(rrange[1])]
        
    dsrange = [(ds[-1])[(r>rrange[0]) & (r<rrange[1])].min(), (ds[-1])[(r>rrange[0]) & (r<rrange[1])].max()]
    # rs.append(r)
    # qs.append(1./r**2)
        
    plots.someplots(rs, qs, name = 'diffuse_qs', xtitle = r'$R/R_*$', ytitle = r'$Q^-$', multix = True, xrange = rrange, xlog = False, ylog = False, formatsequence = [':k', '-r', '-g', 'b--', 'r-'])
    plots.someplots(taulist, qs, name = 'diffuse_tqs', xtitle = r'$\tau_l$', ytitle = r'$Q^-$', multix = True, xrange = taurange, xlog = False, formatsequence = [':k', '-r', '-g', 'b--'], yrange = [0., 0.1])
    plots.someplots(rs, ds, name = 'diffuse_ds', xtitle = r'$R/R_*$', ytitle = r'$Q_{\rm D}$', multix = True, xrange = rrange, yrange = dsrange, xlog = False, formatsequence = [':k', '-r', '-g', 'b--'])
    plots.someplots(taulist, ds, name = 'diffuse_tds', xtitle = r'$\tau_l$', ytitle = r'$Q_{\rm D}$', multix = True, xrange = taurange, yrange = dsrange, xlog = False, formatsequence = [':k', '-r', '-g', 'b--'])
    plots.someplots(taulist, vs, name = 'diffuse_tv', xtitle = r'$\tau_l$', ytitle = r'$v/c$', multix = True, xrange = taurange, xlog = False, formatsequence = [':k', '-r', '-g', 'b--'])
    plots.someplots(taulist, csqs, name = 'diffuse_tcs', xtitle = r'$\tau_l$', ytitle = r'$P/\rho c^2$', multix = True, xrange = taurange, xlog = False, formatsequence = [':k', '-r', '-g', 'b--'])
    plots.someplots(taulist, rs, name = 'diffuse_tr', xtitle = r'$\tau_l$', ytitle = r'$R/R_*$', multix = True, xrange = taurange, yrange = rrange, xlog = False, formatsequence = ['.k', '-k', '-g', 'b--'])

