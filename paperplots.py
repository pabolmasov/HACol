import postpro
import plots
import os
from numpy import *

import configparser as cp
conffile = 'globals.conf'
config = cp.ConfigParser(inline_comment_prefixes="#")
config.read(conffile)

import bassun as bs

modellist = ['light', 'fidu', 'fidu2']
# ['mdot100w', 'mdot100w3', 'mdot100w5', 'mdot100w10', 'mdot100w20', 'mdot100w50']

def titanfetch():

    for a in modellist:
        print("mkdir /Users/pasha/HACol/titania_"+a)
        print("sshpass -p \"i138Sal\" scp pavabo@titan.utu.fi:/home/pavabo/tired/out_"+a+"/\*.\* /Users/pasha/HACol/titania_"+a+"/")
        os.system("mkdir /Users/pasha/HACol/titania_"+a)
        os.system("sshpass -p \"i138Sal\" scp pavabo@titan.utu.fi:/home/pavabo/tired/out_"+a+"/sfro*.dat /Users/pasha/HACol/titania_"+a+"/")
        
'''
# stitching together:
stitch('titania_dhuge/tireout.hdf5','titania_dhuge1/tireout.hdf5') 
system('mv titania_dhuge/tirecombine.hdf5 titania_dhuge/tire123.hdf5')
stitch('titania_dhuge/tire1234.hdf5','titania_dhuge/tireout.hdf5')
  
multishock_plot('titania_bs/sfront', trange=[0.,0.3])
  
quasi2d('titania_fidu/tireout.hdf5', 0,600)
quasi2d('titania_mdot30/tireout.hdf5', 0,1500, conf = 'M30')
quasi2d('titania_mdot1/tireout.hdf5', 0,1500, conf = 'M1')


postpro.multishock(0,1399, 1, prefix = 'out_mdot100w/tireout', dat=False, conf='M100W', xest=7.)
postpro.multishock(0,1129, 1, prefix = 'out_mdot100w3/tireout', dat=False, conf='M100Wdr3', xest=9.)
postpro.multishock(0,1911, 1, prefix = 'out_mdot100w5/tireout', dat=False, conf='M100Wdr5', xest=5.)
postpro.multishock(0,4000, 1, prefix = 'out_mdot100w10/tireout', dat=False, conf='M100Wdr10', xest=4.)
postpro.multishock(0,5000, 1, prefix = 'out_mdot100w20/tireout', dat=False, conf='M100Wdr20', xest=3.)

multishock(0,5000, 1, prefix = 'titania_light/tireout', dat=False, conf='LIGHT')
multishock(0,5000, 1, prefix = 'titania_fidu/tireout', dat=False)
multishock(0,5000, 1, prefix = 'titania_fidu2/tireout', dat=False)
multishock(0,5161, 1, prefix = 'titania_fidu_old/tirecombine', dat=False)
multishock(0,4754, 1, prefix = 'titania_RI/tireout', dat=False, conf='RI', kleap = 13)   
multishock(0,9656, 1, prefix = 'titania_nod/tireout', dat=False, conf='NOD') 
multishock(0,9990, 1, prefix = 'titania_narrow/tireout', dat=False, conf='NARROW', xest=8.)
multishock(0,9990, 1, prefix = 'titania_narrow2/tireout', dat=False, conf='NARROW', xest=8.)
multishock(0,6589, 1, prefix = 'titania_mdot30/tireout', dat=False, conf='M30', xest=7.)
multishock(0,3090, 1, prefix = 'titania_mdot1/tireout', dat=False, conf='M1', xest=1.5)
multishock(0,5000, 1, prefix = 'titania_mdot3/tireout', dat=False, conf='M3')
multishock(0,9492, 1, prefix = 'titania_dhuge_old/tirecombine', dat=False, conf='DHUGE', kleap=13)
multishock(0,9492, 1, prefix = 'titania_mdot100/tireout', dat=False, conf='M100', kleap=13)
multishock(0,1377, 1, prefix = 'titania_widenod/tireout', dat=False, conf='WIDENOD')
multimultishock_plot(["titania_fidu", "titania_fidu2", "titania_light"], parflux = True)
multimultishock_plot(["titania_fidu", "titania_nod"], parflux = True)
multimultishock_plot(["titania_fidu", "titania_rot", "titania_RI"], parflux = True)
multimultishock_plot(["titania_fidu", "titania_irr"], parflux = True, sfilter = 1.2)
dynspec(infile = 'titania_huge/sfront', nbins = 20, ntimes=50, iffront = True,deline=True, fosccol = -1)

quasi2d('titania_fidu2/tireout.hdf5', 0,1500)
mdotmap(0,1500,1, prefix='titania_fidu2/tireout', conf='FIDU') 
quasi2d('titania_mdot30/tireout.hdf5', 0,4000, conf='M30')
mdotmap(0,4000,1, prefix='titania_mdot30/tireout', conf='M30') 

mdotmap(0,500,1, prefix='titania_irr/tireout', conf='IRR')  
quasi2d('titania_irr/tireout.hdf5', 0,500, conf='IRR')

mdotmap(0,500,1, prefix='titania_wide/tireout', conf='WIDE') 
quasi2d('titania_narrow2/tireout.hdf5', 0,8000, conf='NARROW', step=20)
mdotmap(0,8000,20, prefix='titania_narrow2/tireout', conf='NARROW') 

mdotmap(0,500,1, prefix='titania_mdot1/tireout', conf='M1') 
mdotmap(0,500,1, prefix='titania_mdot3/tireout', conf='M3') 
mdotmap(0,500,1, prefix='titania_mdot100/tireout', conf='M100') 
mdotmap(0,500,1, prefix='titania_mdot100w/tireout', conf='M100W') 


multimultishock_plot(["titania_fidu2", "titania_fidu"])

postpro.multishock(0,5000, 1, prefix = 'titania_fidu/tirecombine', dat=False)
plots.quasi2d('titania_fidu/tirecombine.hdf5', 0,3000)
multishock(0,7460, 1, prefix = 'titania_nod/tireout', dat=False, conf='NOD')
postpro.multishock(0,670, 1, prefix = 'titania_irr/tireout', dat=False, conf='IRR')
postpro.multishock(0,1095, 1, prefix = 'titania_rot/tireout', dat=False, conf='ROT')
postpro.multishock(0,4400, 1, prefix = 'titania_wide/tireout', dat=False, conf='WIDE')
postpro.multishock(0,6816, 1, prefix = 'titania_widenod/tireout', dat=False, conf='WIDENOD')
quasi2d('titania_widenod/tireout.hdf5', 0,6816)
postpro.multishock(0,623, 1, prefix = 'titania_mdot1/tireout', dat=False, conf='M1')
postpro.multishock(0,427, 1, prefix = 'titania_mdot3/tireout', dat=False, conf='M3')
postpro.multishock(0,1850, 1, prefix = 'titania_mdot30/tireout', dat=False, conf='M30')
multishock(0,2048, 1, prefix = 'titania_mdot100/tireout', dat=False, conf='M100')
postpro.multishock(0,1119, 1, prefix = 'titania_mdot1n/tireout', dat=False, conf='M1N')

postpro.filteredflux("titania_fidu/tireout.hdf5", 0, 1000, rfraction = 0.5)
postpro.filteredflux("titania_fidu/tireout.hdf5", 0, 670, rfraction = 0.5) 
postpro.filteredflux("titania_rot/tireout.hdf5", 0, 1000, rfraction = 0.5)
postpro.filteredflux("titania_irr/tireout.hdf5", 0, 1000, rfraction = 0.5)
postpro.filteredflux("titania_wide/tireout.hdf5", 0, 1000, rfraction = 0.5)
postpro.filteredflux("titania_xireal/tireout.hdf5", 0, 1000, rfraction = 0.5)
multishock(0,8900, 1, prefix = 'titania_mdot100/tireout', dat=False, mdot=100.*4.*pi, kleap=3)
plots.quasi2d('titania_mdot3/tireout.hdf5', 0,427, conf='M3')
plots.quasi2d('titania_mdot30/tireout.hdf5', 0,1850, conf='M30')
plots.quasi2d('titania_mdot1/tireout.hdf5', 0,182, conf='M1') 
quasi2d('titania_mdot1n/tireout.hdf5', 0,1119, conf='M1N') 
plots.quasi2d('titania_wide/tireout.hdf5', 0,4000, conf='WIDE')
plots.quasi2d('titania_rot/tireout.hdf5', 0,1095, conf='ROT')
plots.quasi2d('titania_irr/tireout.hdf5', 0,670, conf='IRR')
plots.quasi2d('titania_v5/tireout.hdf5', 0,670, conf='V5') 
plots.quasi2d('titania_v30/tireout.hdf5', 0,670, conf='V30') 

plots.quasi2d('titania_rot/tireout.hdf5', 0,300)
plots.quasi2d('titania_mdot100/tireout.hdf5', 0,300)
plots.twomultishock_plot("titania_fidu/flux", "titania_fidu/sfront", "titania_rot/flux", "titania_rot/sfront")
plots.twomultishock_plot("titania_fidu/flux", "titania_fidu/sfront", "titania_irr/flux", "titania_irr/sfront")
postpro.comparer('galia_F/BS_solution_F', 'titania_fidu1/tireout04000')
comparer('galia_M30/BS_solution_M30', 'titania_mdot30/tireout06000', vone = -1.170382e+06/3e10)
comparer('galia_F/BS_solution_F', 'titania_BS/tireout', vone = -9.778880e+05/3e10, nentry = [4000,5000])
comparer('galia_M100/BS_solution_M100', 'titania_mdot100/tireout', vone = -1.957280e+06/3e10, nentry = [4000,5000])
comparer('galia_N/BS_solution_N', 'titania_narrow2/tireout', vone = -8.194837e+06/3e10, nentry = [4000,5000])

'''
def figkakkonen():
    drr = asarray([0.3, 0.25, 0.2, 0.1, 0.05, 0.02])
    xs = asarray([12.450, 8.869, 6.485, 3.733, 2.458, 1.656])
    dxs = asarray([0.010, 0.016, 0.009, 0.005, 0.003, 0.002])
    beta = asarray([0.63, 0.60, 0.56, 0.43, 0.308, 0.177])
    
    conf = 'M100W'
    rstar = config[conf].getfloat('rstar')
    m1 = config[conf].getfloat('m1')
    mu30 = config[conf].getfloat('mu30')
    mdot = config[conf].getfloat('mdot') * 4.*pi
    afac = config[conf].getfloat('afac')
    realxirad = config[conf].getfloat('xirad')
    b12 = 2.*mu30*(rstar*m1/6.8)**(-3) # dipolar magnetic field on the pole, 1e12Gs units
    umag = b12**2*2.29e6*m1
    outdir = 'titania_mdot100w'
    geometry = loadtxt(outdir+"/geo.dat", comments="#", delimiter=" ", unpack=False)
    across0 = geometry[0,3]  ;   delta0 = geometry[0,5]

    ndr = 100 ; drmin = 0.01 ; drmax = 0.7
    drar = (drmax/drmin)**(arange(ndr)/double(ndr-1))*drmin

    across0 = across0 * (drar/0.25)
    delta0 = delta0 * (drar/0.25)

    xs_BS = zeros(ndr) ; beta_BS = zeros(ndr)
    x0 = 1.5

    for k in arange(ndr):
        BSgamma = (across0[k]/delta0[k]**2)/mdot*rstar / (realxirad/1.5)
        BSeta = (8./21./sqrt(2.)*umag*3. * (realxirad/1.5))**0.25*sqrt(delta0[k])/(rstar)**0.125
        xs_tmp, BSbeta_tmp = bs.xis(BSgamma, BSeta, x0 = x0, ifbeta = True)
        xs_BS[k] = xs_tmp ; beta_BS[k] = BSbeta_tmp
        x0 = xs_tmp
        print(x0)

    rmax = 13.53
    plots.someplots([beta, beta_BS, beta_BS], [xs, xs_BS, beta_BS*0.+rmax], xtitle=r'$\beta_{\rm BS}$', ytitle=r'$R_{\rm shock}/R_*$', multix = True, formatsequence = ['k.', 'r-', 'b:'], name = 'forpaper/figkakkonen', xlog = False, ylog = False, dys = dxs, vertical=2./3.)


def massrace():
    dirlist = ['titania_mdot100w3', 'titania_mdot100w', 'titania_mdot100w5',
               'titania_mdot100w10', 'titania_mdot100w20', 'titania_mdot100w50']
    nlist = size(dirlist)
    tlist = [] ;  mlist = []

    confs = ['M100Wdr3','M100W', 'M100Wdr5', 'M100Wdr10', 'M100Wdr20', 'M100Wdr50']
    
    for k in arange(nlist):
        rstar = config[confs[k]].getfloat('rstar')
        m1 = config[confs[k]].getfloat('m1')
        mu30 = config[confs[k]].getfloat('mu30')
        mdot = config[confs[k]].getfloat('mdot') * 4.*pi
        afac = config[confs[k]].getfloat('afac')
        tscale = config[confs[k]].getfloat('tscale') * m1
        mscale = config[confs[k]].getfloat('massscale') * m1**2
        b12 = 2.*mu30*(rstar*m1/6.8)**(-3) # dipolar magnetic field on the pole, 1e12Gs units
        umag = b12**2*2.29e6*m1
        totalfile = dirlist[k]+'/totals.dat'
        lines = loadtxt(totalfile, comments="#", delimiter=" ", unpack=False)
        t = lines[:,0] ; mass = lines[:,1]
        geometry = loadtxt(dirlist[k]+"/geo.dat", comments="#", delimiter=" ", unpack=False)
        across0 = geometry[0,3] ; cth0 = cos(geometry[0,1])
        mcol = across0 * rstar**2 * umag / m1 * (1.+3.*cth0**2)/4.
        print("mcol = "+str(mcol*mscale)+"g")
        tr = mcol / mdot * tscale
        tlist.append(t/tr) ; mlist.append(mass/mcol)
    
    plots.someplots(tlist, mlist, xtitle=r'$t/t_{\rm r}$', ytitle=r'$M/M_{\rm col}$',
                    multix = True, formatsequence = ["k-" for x in range(nlist)],
                    linewidthsequence = nlist - arange(nlist),
                    xlog = False, ylog = False, name = 'forpaper/massrace',
                    xrange=[0,10])

def geotest():
    gfile = loadtxt("out/geo.dat", comments="#", delimiter=" ", unpack=False)
    r = gfile[:,0] ;  across = gfile[:,3]
    plots.someplots(r/r[0], [across/across[0]*(r[0]/r)**3-1.], xlog = True, ylog = True, \
                    name = 'geotest', formatsequence = ['k-'])

