from numpy import *
from scipy.integrate import *
from scipy.interpolate import *
from scipy.optimize import curve_fit
import os

from globals import *

import hdfoutput as hdf
import bassun as bs
if ifplot:
    import plots

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
    plots.dynspec(t2,binfreq2, log10(pds2), outfile=infile+'_dyns', nbin=nbin)

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
def shock_hdf(n, infile = "out/tireout.hdf5"):
    '''
    finds the position of the shock in a given entry of the infile
    '''
    entryname, t, l, r, sth, rho, u, v = hdf.read(infile, n)

    #find maximal compression:
    dvdl = (v[1:]-v[:-1])/(l[1:]-l[:-1])
    wcomp = (dvdl).argmin()
    print("maximal compression found at r="+str(r[wcomp]))
    return (r[wcomp]+r[wcomp+1])/2., (r[wcomp+1]-r[wcomp])/2.

def shock_dat(n, prefix = "out/tireout"):
    '''
    finds the position of the shock from a given dat-file ID
    '''
    fname = prefix + hdf.entryname(n, ndig=5) + ".dat"
    lines = loadtxt(fname, comments="#")
    r = lines[:,0] ; v = lines[:,2]
    #find maximal compression:
    dvdl = (v[1:]-v[:-1])/(r[1:]-r[:-1])
    wcomp = (dvdl).argmin()
    #    print("maximal compression found at r="+str(r[wcomp])+".. "+str(r[wcomp+1])+"rstar")
    return (r[wcomp]+r[wcomp+1])/2., (r[wcomp+1]-r[wcomp])/2.
    
def multishock(n1,n2, dn, prefix = "out/tireout", dat = True):
    '''
    draws the motion of the shock front with time, for a given set of HDF5 entries or ascii outputs
    '''
    
    n=arange(n1, n2, dn, dtype=int)
    s=arange(size(n), dtype=double)
    ds=arange(size(n), dtype=double)
    print(size(n))
    outdir = os.path.dirname(prefix)
    fluxlines = loadtxt(outdir+"/flux.dat", comments="#", delimiter=" ", unpack=False)
    geometry = loadtxt(outdir+"/geo.dat", comments="#", delimiter=" ", unpack=False)
    t=fluxlines[:,0] ; f=fluxlines[:,1]
    across0 = geometry[0,3]  ;   delta0 = geometry[0,5]
    BSgamma = (2.*across0/delta0**2)/mdot*rstar
    # umag is magnetic pressure
    BSeta = (8./21./sqrt(2.)*30.*umag*m1)**0.25*sqrt(delta0)/(rstar)**0.125
    xs = bs.xis(BSgamma, BSeta, x0=20.)

    # spherization radius
    rsph =1.5*mdot/4./pi
    eqlum = mdot/rstar
    print("mdot = "+str(mdot))
    print("rstar = "+str(rstar))
    # iterating to find the cooling radius
    rcool = rcoolfun(geometry, mdot)
    for k in arange(size(n)):
        if(dat):
            stmp, dstmp = shock_dat(n[k], prefix=prefix)
        else:
            stmp, dstmp = shock_hdf(n[k], infile = prefix+".hdf5")
        s[k] = stmp ; ds[k] = dstmp

    print("predicted shock position: xs = "+str(xs)+" (rstar)")
    print("cooling limit: rcool/rstar = "+str(rcool/rstar))
        
    if(ifplot):
        plots.someplots(t[n], [s, s*0.+xs, s*0.+rcool/rstar], name = outdir+"/shockfront", xtitle=r'$t$, s', ytitle=r'$R_{\rm shock}/R_*$', xlog=False, formatsequence = ['k-', 'r-', 'b-'])
        plots.someplots(f[n], [s, s*0.+xs, s*0.+rcool/rstar], name=outdir+"/fluxshock", xtitle=r'Flux', ytitle=r'$R_{\rm shock}/R_*$', xlog=False, ylog=False, formatsequence = ['k-', 'r-', 'b-'], vertical = eqlum)
    # ascii output
    fout = open(outdir+'/sfront.dat', 'w')
    for k in arange(size(n)):
        fout.write(str(t[n[k]])+" "+str(s[k])+" "+str(ds[k])+"\n")
    fout.close()
    fglo = open(outdir + '/sfrontglo.dat', 'w') # BS shock position and equilibrium flux
    fglo.write('# equilibrium luminosity -- BS shock front position / rstar -- Rcool position / rstar\n')
    fglo.write(str(eqlum)+' '+str(xs[0])+' '+str(rcool/rstar)+'\n')
    fglo.close()
    
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
    plots.someplots(t, [f, tailfitfun(t, par[0], par[1], par[2], par[3])], name = prefix+"_fit", xtitle=r'$t$, s', ytitle=r'$L$', xlog=True, ylog=True)
    print("slope ="+str(par[0]))
    print("y0 ="+str(par[3]))
