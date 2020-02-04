from numpy import *
from scipy.integrate import *
from scipy.interpolate import *
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import os

from globals import *

import hdfoutput as hdf
import geometry as geo
import bassun as bs
if ifplot:
    import plots
    from matplotlib.pyplot import ioff
    ioff()

def rcoolfun(geometry, mdot):
    '''
    calculates the cooling radius from the known geometry
    '''
    r = geometry[:,0] ; across = geometry[:,3] ; delta = geometry[:,5]

    f = delta**2/across * mdot- r
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

def dynspec(infile='out/flux', ntimes=10, nbins=100, binlogscale=False):
    '''
    makes a dynamic spectrum by making Fourier in each of the "ntimes" time bins. Fourier PDS is binned to "nbins" bins
    '''
    lines = loadtxt(infile+".dat", comments="#", delimiter=" ", unpack=False)
    t=lines[:,0] ; l=lines[:,1]
    nsize=size(t)
    tbin=linspace(t.min(), t.max(), ntimes+1)
    tcenter=(tbin[1:]+tbin[:-1])/2.
    freq1=1./(t.max())*double(ntimes)/2. ; freq2=freq1*double(nsize)/double(ntimes)/2.
    if(binlogscale):
        binfreq=logspace(log10(freq1), log10(freq2), num=nbins+1)
    else:
        binfreq=linspace(freq1, freq2, nbins+1)
    binfreqc=(binfreq[1:]+binfreq[:-1])/2.
    pds2=zeros([ntimes, nbins]) ;   dpds2=zeros([ntimes, nbins])
    t2=zeros([ntimes+1, nbins+1], dtype=double)
    nbin=zeros([ntimes, nbins], dtype=double)
    binfreq2=zeros([ntimes+1, nbins+1], dtype=double)
    fdyns=open(infile+'_dyns.dat', 'w')
    for kt in arange(ntimes):
        wt=(t<tbin[kt+1]) & (t>=tbin[kt])
        lt=l[wt]
        fsp=fft.rfft((lt-lt.mean())/l.std(), norm="ortho")
        nt=size(lt)
        freq = fft.rfftfreq(nt, (t[wt].max()-t[wt].min())/double(nt))
        pds=abs(fsp*freq)**2
        t2[kt,:]=tbin[kt] ; t2[kt+1,:]=tbin[kt+1] 
        binfreq2[kt,:]=binfreq[:] ; binfreq2[kt+1,:]=binfreq[:] 
        for kb in arange(nbins):
            wb=((freq>binfreq[kb]) & (freq<=binfreq[kb+1]))
            nbin[kt,kb] = size(pds[wb])
            #            print("size(f) = "+str(size(freq)))
            #            print("size(pds) = "+str(size(pds)))
            pds2[kt, kb]=pds[wb].mean() ; dpds2[kt, kb]=pds[wb].std()
            # ascii output:
            fdyns.write(str(tcenter[kt])+' '+str(binfreq[kb])+' '+str(binfreq[kb+1])+' '+str(pds2[kt,kb])+' '+str(dpds2[kt,kb])+" "+str(nbin[kt,kb])+"\n")
    fdyns.close()
    print(t2.max())
    plots.plot_dynspec(t2,binfreq2, log10(pds2), outfile=infile+'_dyns', nbin=nbin)

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
                
    plots.binplot(binedges, fn, dfn, fname=infile+"_hist", fit = exp(p[0]*log(binedges)+p[1]))

#####################################################################
def shock_hdf(n, infile = "out/tireout.hdf5", kleap = 3):
    '''
    finds the position of the shock in a given entry of the infile
    kleap allows to measure the velocity difference using several cells
    '''
    entryname, t, l, r, sth, rho, u, v, qloss = hdf.read(infile, n)
    n=size(r)
    #    v=medfilt(v, kernel_size=3)
    v1=savgol_filter(copy(v), 2*kleap+1, 1) # Savitzky-Golay filter
    #find maximal compression:
    dvdl = (v[kleap:]-v[:-kleap])/(l[kleap:]-l[:-kleap])
    wcomp = (dvdl).argmin()
    wcomp1 = maximum((wcomp-kleap),0)
    wcomp2 = minimum((wcomp+kleap),n-1)
    print("maximal compression found at r="+str((r[wcomp1]+r[wcomp2])/2.)+" +/- "+str((-r[wcomp1]+r[wcomp2])/2.))
    if isnan(r[wcomp]):
        print("t = "+str(t))
        print(dvdl.min(), dvdl.max())
        print(dvdl[wcomp])
        plots.someplots(r[1:], [v[1:], v1[1:], dvdl], name = "shocknan", xtitle=r'$r$', ytitle=r'$v$', xlog=False, formatsequence = ['k.', 'r-', 'b'])
        ii=input('r')
    return (r[wcomp1]+r[wcomp2])/2.,(-r[wcomp1]+r[wcomp2])/2., v[wcomp1], v[wcomp2]
    #    return (r[wcomp:wcomp1]).mean(), r[wcomp:wcomp1],v[wcomp], v[wcomp1]
# v[wcomp], v[wcomp+1]

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
    return (r[wcomp]+r[wcomp+1])/2., (r[wcomp+1]-r[wcomp])/2.,v[maximum(wcomp-kleap,00)], v[minimum(wcomp+1+kleap, size(r)-1)]
    
def multishock(n1,n2, dn, prefix = "out/tireout", dat = True, mdot=mdot, afac = afac, kleap = 1, realxirad = xirad):
    '''
    draws the motion of the shock front with time, for a given set of HDF5 entries or ascii outputs
    '''
    
    n=arange(n1, n2, dn, dtype=int)
    s=arange(size(n), dtype=double)
    ds=arange(size(n), dtype=double)
    dv=arange(size(n), dtype=double)
    v2=arange(size(n), dtype=double)
    v1=arange(size(n), dtype=double)
    compression=arange(size(n), dtype=double)
    print(size(n))
    outdir = os.path.dirname(prefix)
    fluxlines = loadtxt(outdir+"/flux.dat", comments="#", delimiter=" ", unpack=False)
    geometry = loadtxt(outdir+"/geo.dat", comments="#", delimiter=" ", unpack=False)
    t=fluxlines[:,0] ; f=fluxlines[:,1]
    across0 = geometry[0,3]  ;   delta0 = geometry[0,5]
    BSgamma = (2.*across0/delta0**2)/mdot*rstar / (realxirad/1.5)
    # umag is magnetic pressure
    BSeta = (8./21./sqrt(2.)*30.*umag*m1 * (realxirad/1.5))**0.25*sqrt(delta0)/(rstar)**0.125
    xs = bs.xis(BSgamma, BSeta, x0=3.0)
    #     print("xs = "+str(xs))
    #    ii=input("xs")
    # spherization radius
    rsph =1.5*mdot/4./pi
    eqlum = mdot/rstar
    print("m1 = "+str(m1))
    print("mdot = "+str(mdot))
    print("rstar = "+str(rstar))
    # iterating to find the cooling radius
    rcool = rcoolfun(geometry, mdot)
    for k in arange(size(n)):
        if(dat):
            stmp, dstmp, v1tmp, v2tmp = shock_dat(n[k], prefix=prefix, kleap = kleap)
        else:
            stmp, dstmp, v1tmp, v2tmp = shock_hdf(n[k], infile = prefix+".hdf5", kleap = kleap)
        s[k] = stmp ; ds[k] = dstmp
        v1[k] = v1tmp   ; v2[k] =  v2tmp
        dv[k] = v1tmp - v2tmp
        compression[k] = v2tmp/v1tmp

    print("predicted shock position: xs = "+str(xs)+" (rstar)")
    print("cooling limit: rcool/rstar = "+str(rcool/rstar))
    f /= 4.*pi  ; eqlum /= 4.*pi
        
    if(ifplot):
        ws=where(s>1.)
        n=n[ws]
        plots.someplots(t[n], [s[ws], s*0.+xs], name = outdir+"/shockfront", xtitle=r'$t$, s', ytitle=r'$R_{\rm shock}/R_*$', xlog=False, formatsequence = ['k-', 'r-', 'b-'])
        plots.someplots(f[n], [s[ws], s*0.+xs], name=outdir+"/fluxshock", xtitle=r'$L/L_{\rm Edd}$', ytitle=r'$R_{\rm shock}/R_*$', xlog=True, ylog=False, formatsequence = ['k-', 'r-', 'b-'], vertical = eqlum)
        plots.someplots(t[n], [f[n], f[n]*0.+eqlum], name = outdir+"/flux", xtitle=r'$t$, s', ytitle=r'$L/L_{\rm Edd}$', xlog=False, ylog=False)
        plots.someplots(t[n], [-v1[ws], -v2[ws], sqrt(2./s[ws]/rstar), sqrt(2./s[ws]/rstar)/7.], name = outdir+"/vleap",xtitle=r'$t$, s', ytitle=r'$ v /c$', xlog=False, formatsequence = ['k-', 'b:', 'r-', 'r-'])

    print("effective compression factor "+str(compression[isfinite(compression)].mean()))
    # ascii output
    fout = open(outdir+'/sfront.dat', 'w')
    fout.write("# time -- shock position -- downstream velocity -- upstream velocity \n")
    for k in arange(size(n)):
        fout.write(str(t[n[k]])+" "+str(s[k])+" "+str(v1[k])+" "+str(v2[k])+"\n")
    fout.close()
    fglo = open(outdir + '/sfrontglo.dat', 'w') # BS shock position and equilibrium flux
    fglo.write('# equilibrium luminosity -- BS shock front position / rstar -- Rcool position / rstar\n')

    if isscalar(xs):
        fglo.write(str(eqlum)+' '+str(xs)+' '+str(rcool/rstar)+'\n')
    else:
        fglo.write(str(eqlum)+' '+str(xs[0])+' '+str(rcool/rstar)+'\n')
    fglo.close()
    # last 0.1s average shock position
    if t[n].max() > (t[n].min() + 0.1):
        wlaten = (t[n] > (t[n].max()-0.1))
        wlate = (t > (t.max()-0.1))
        xmean = s[wlaten].mean() ; xrms = s[wlaten].std()+ds[wlaten].mean()
        print("s/RNS = "+str(xmean)+"+/-"+str(xrms)+"\n")
        fmean = f[wlate].mean() ; frms = f[wlate].std()
        print("flux = "+str(fmean)+"+/-"+str(frms)+"\n")
        
###############################
def tailfitfun(x, p, n, x0, y0):
    return ((x-x0)**2)**(p/2.)*n+y0

def tailfit(prefix = 'out/flux', trange = None):
    fluxlines = loadtxt(prefix+".dat", comments="#", delimiter=" ", unpack=False)
    
    t=fluxlines[:,0] ; f=fluxlines[:,1]
    if(trange is None):
        t1 = t; f1 = f
    else:
        t1=t[(t>trange[0])&(t<trange[1])]
        f1=f[(t>trange[0])&(t<trange[1])]
    par, pcov = curve_fit(tailfitfun, t1, f1, p0=[-2.,5., 0., f1.min()])
    plots.someplots(t, [f, tailfitfun(t, par[0], par[1], par[2], par[3]), t*0.+par[3]], name = prefix+"_fit", xtitle=r'$t$, s', ytitle=r'$L$', xlog=False, ylog=False, formatsequence=['k.', 'r-', 'g:'])
    print("slope ="+str(par[0])+"+/-"+str(sqrt(pcov[0,0])))
    print("y0 ="+str(par[3])+"+/-"+str(sqrt(pcov[3,3])))

##################################################################
def mdotmap(n1, n2, step,  prefix = "out/tireout", ifdat = False):
    # reconstructs the mass flow
    # reding geometry:
    geofile = os.path.dirname(prefix)+"/geo.dat"
    fluxfile = os.path.dirname(prefix)+"/flux.dat"
    fluxlines = loadtxt(fluxfile, comments="#")
    tar=fluxlines[:,0]
    print(geofile)
    r, theta, alpha, across, l, delta = geo.gread(geofile) 
    print(mdot)
    nr = size(r) 
    indices = arange(n1, n2, step)
    nind = size(indices)
    md2 = zeros([nind, nr], dtype=double)
    t2 = zeros([nind, nr], dtype=double)
    r2 = zeros([nind, nr], dtype=double)   

    
    for k in arange(nind): # arange(n1, n2, step):
        hname = prefix + ".hdf5"
        if(ifdat):
            fname = prefix + hdf.entryname(indices[k], ndig=5) + ".dat"
            print(fname)
            lines = loadtxt(fname, comments="#")
            r = lines[:,0] ; rho = lines[:,1] ; v = lines[:,2]
            t=tar[indices[k]]
        else:
            entryname, t, l, r, sth, rho, u, v, qloss = hdf.read(hname, indices[k])
        md2[k, :] = (rho * v * across)[:]
        t2[k, :] = t  ;     r2[k, :] = r[:]

    # ascii output:
    fmap = open(prefix+"_mdot.dat", "w")
    for k in arange(k):
        for kr in arange(nr):
            fmap.write(str(t2[k, kr])+" "+str(r2[k, kr])+" "+str(md2[k, kr])+"\n")
    fmap.close()
    if(ifplot):
        # graphic output
        nlev=30
        plots.somemap(r2, t2, -md2/mdot, name=prefix+"_mdot", levels = 2.*arange(nlev)/double(nlev-2))
        mdmean = -md2.mean(axis=0)
        plots.someplots(r, [mdmean/(4.*pi), mdmean*0.+mdot/(4.*pi)], name=prefix+"_mdmean",
                        xtitle='$R/R_*$', ytitle=r"$\dot{M}c^2/L_{\rm Edd}$", formatsequence=['k.', 'r-'])
        
        
def taus(n, prefix = 'out/tireout', ifhdf = True):
    '''
    calculates the optical depths along and across the flow
    '''
    geofile = os.path.dirname(prefix)+"/geo.dat"
    print(geofile)
    r, theta, alpha, across, l, delta = geo.gread(geofile) 
    if(ifhdf):
        hname = prefix + ".hdf5"
        entryname, t, l, r, sth, rho, u, v, qloss = hdf.read(hname, n)
    else:
        entryname = hdf.entryname(n, ndig=5)
        fname = prefix + entryname + ".dat"
        lines = loadtxt(fname, comments="#")
        rho = lines[:,1]
    taucross = delta * rho  # across the flow
    dr = (r[1:]-r[:-1])  ;   rc = (r[1:]+r[:-1])/2.
    taualong = (rho[1:]+rho[:-1])/2. * dr
    taucrossfun = interp1d(r, taucross, kind = 'linear')
    if(ifplot):
        plots.someplots(rc/rstar, [taucrossfun(rc), taualong], name = prefix+"_tau", xtitle=r'$r/R_{\rm NS}$', ytitle=r'$\tau$', xlog=True, ylog=True, formatsequence=['k-', 'r-'])

def virialratio(n, prefix = 'out/tireout', ifhdf = True):
    '''
    checks the virial relations in the flow
    '''
    geofile = os.path.dirname(prefix)+"/geo.dat"
    print(geofile)
    r, theta, alpha, across, l, delta = geo.gread(geofile) 
    if(ifhdf):
        hname = prefix + ".hdf5"
        entryname, t, l, r, sth, rho, u, v, qloss = hdf.read(hname, n)
    else:
        entryname = hdf.entryname(n, ndig=5)
        fname = prefix + entryname + ".dat"
        lines = loadtxt(fname, comments="#")
        rho = lines[:,1]
    egrav = trapz(rho * across / r, x = l)
    ethermal = trapz(u * across, x = l)
    ebulk = trapz(rho * across * v**2/2., x = l)

    return t, egrav, ethermal, ebulk, (ethermal+ebulk)/egrav

def virialtest(n1, n2, prefix = 'out/tireout'):
    '''
    tests virial relations as a function of time
    '''
    nar = arange(n2-n1)+n1

    virar = zeros(n2-n1) ; tar = zeros(n2-n1)
    
    for k in arange(n2-n1):
        t, egrav, ethermal, ebulk, viratio = virialratio(nar[k], prefix = prefix)
        virar[k] = viratio ; tar[k] = t

    if(ifplot):
        plots.someplots(tar, [virar], name = prefix+"_vire", xtitle=r'$t$', ytitle=r'$E_{\rm k} / E_{\rm g}$', xlog=False, ylog=False, formatsequence=['k-'])

def filteredflux(hfile, n1, n2, rfraction = 0.9):
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

    lint = zeros(n2-n1)
    ltot = zeros(n2-n1)
    tar = zeros(n2-n1)
    
    for k in arange(n2-n1)+n1:
        entryname, t, l, r, sth, rho, u, v, qloss = hdf.read(hfile, k)
        lint[k] = simps(qloss[wr], x=l[wr])
        ltot[k] = simps(qloss, x=l)
        tar[k] = t
        #    ltot /= 4.*pi ; lint /= 4.*pi # convert to Eddington units
    if(ifplot):
        # overplotting with the total flux
        plots.someplots(tar, [lint, ltot, ltot-lint, lint*0.+mdot*0.2], xlog=False, formatsequence = ['k-', 'g--', 'b--', 'r-'], xtitle='t, s', ytitle=r'$L/L_{\rm Edd}$', name= os.path.dirname(hfile)+'/cutflux')

    # ascii output:
    fout = open(os.path.dirname(hfile)+"/cutflux.dat", "w")
    for k in arange(n2-n1)+n1:
        fout.write(str(tar[k])+" "+str(lint[k])+" "+str(ltot[k])+"\n")
        fout.flush()
    fout.close()
    
