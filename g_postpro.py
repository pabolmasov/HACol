import numpy
from numpy import *

from scipy.integrate import *
from scipy.interpolate import *
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import os

import g_hdfoutput as hdf
import geometry as geo
import bassun as bs
from beta import *
import g_plots as plots
import builtins
import sys
import configparser as cp
import g_units as units

from scipy import __version__ as scipy_version

# Compatible with old and new SciPy versions
scipy_version_tuple = tuple(
    int(x) for x in scipy_version.split('.')[:2]
)

if scipy_version_tuple >= (1, 14):
    from scipy.integrate import cumulative_trapezoid as cumtrapz
else:
    from scipy.integrate import cumtrapz

# Compatible with old and new NumPy versions
if hasattr(numpy, 'trapezoid'):
    numerical_trapezoid = numpy.trapezoid
else:
    numerical_trapezoid = numpy.trapz

conffile = 'globals.conf'
config = cp.ConfigParser(inline_comment_prefixes="#")
config.read(conffile)
ifplot =  config['DEFAULT'].getboolean('ifplot') 
ifplot=1  # to plot everything

# @galja added: for separate ifplot option:
conffile1 = 'plot.conf'
configplot = cp.ConfigParser(inline_comment_prefixes="#")
configplot.read(conffile1)
ifplot = configplot['DEFAULT'].getboolean('ifplot')
ifplot=True

# Constants
Consts = {
    'C': 2.997925e10,      # Speed of light in cm/s (example value)
    'Led': 2e38,    # Some constant (example value)
    'Rns': 1e6,      # Radius (example value)
    'kappaT': 0.35,  # Some constant (example value)
    'G': 6.67428e-8,  # Gravitational constant in cm^3/g/s^2
    'Msun': 1.989e33, # Solar mass in grams (example value)
    'mx': 1.4,       # Placeholder for mass (example value)
    'SigmaSB' : 5.6704e-5 # post. Stephana-Bolzmana
}



verbatim_vent=0 
DEBUG=0

if ifplot:
    import g_plots as plots
    from matplotlib.pyplot import ioff
    ioff()



    
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


def plot_log_comparison(dirname, filename, bs_x, bs_y, sim_x, sim_y, sim_err=None, extra_curves=None, formatsequence=None, legendsequence=None, xtitle='', ytitle='', title=None):
    curves_x = []
    curves_y = []
    valid_values = []

    bs_x = asarray(bs_x).ravel()
    bs_y = asarray(bs_y).ravel()

    valid_bs = isfinite(bs_x) & isfinite(bs_y) & (bs_x > 0.0) & (bs_y > 0.0)

    if valid_bs.sum() >= 2:
        curves_x.append(bs_x[valid_bs])
        curves_y.append(bs_y[valid_bs])
        valid_values.append(bs_y[valid_bs])
    else:
        print("Warning: insufficient valid analytical values for", filename)

    sim_x = asarray(sim_x).ravel()
    sim_y = asarray(sim_y).ravel()

    if sim_err is not None:
        sim_err = asarray(sim_err).ravel()
        valid_sim = isfinite(sim_x) & isfinite(sim_y) & isfinite(sim_err) & (sim_x > 0.0) & (sim_y > 0.0) & (sim_y + sim_err > 0.0) & (sim_y - sim_err > 0.0)
    else:
        valid_sim = isfinite(sim_x) & isfinite(sim_y) & (sim_x > 0.0) & (sim_y > 0.0)

    if valid_sim.sum() >= 2:
        curves_x.append(sim_x[valid_sim])
        curves_y.append(sim_y[valid_sim])
        valid_values.append(sim_y[valid_sim])

        if sim_err is not None:
            curves_x.append(sim_x[valid_sim])
            curves_y.append((sim_y + sim_err)[valid_sim])
            valid_values.append((sim_y + sim_err)[valid_sim])

            curves_x.append(sim_x[valid_sim])
            curves_y.append((sim_y - sim_err)[valid_sim])
            valid_values.append((sim_y - sim_err)[valid_sim])
    else:
        print("Warning: insufficient valid simulation values for", filename)

    if extra_curves is not None:
        for extra_x, extra_y in extra_curves:
            extra_x = asarray(extra_x).ravel()
            extra_y = asarray(extra_y).ravel()
            valid_extra = isfinite(extra_x) & isfinite(extra_y) & (extra_x > 0.0) & (extra_y > 0.0)

            if valid_extra.sum() >= 2:
                curves_x.append(extra_x[valid_extra])
                curves_y.append(extra_y[valid_extra])
                valid_values.append(extra_y[valid_extra])

    if len(curves_x) == 0:
        print("Warning: no valid curves available for", filename)
        return

    valid_values = concatenate(valid_values)
    valid_values = valid_values[isfinite(valid_values) & (valid_values > 0.0)]

    if valid_values.size == 0:
        print("Warning: no positive values available for", filename)
        return

    yrange = [valid_values.min()/2.0, valid_values.max()*2.0]

    if formatsequence is None:
        formatsequence = ['k-'] * len(curves_x)

    if legendsequence is None:
        legendsequence = [''] * len(curves_x)

    if len(formatsequence) != len(curves_x):
        print("Warning: formatsequence length does not match number of valid curves for", filename)
        return

    if len(legendsequence) != len(curves_x):
        print("Warning: legendsequence length does not match number of valid curves for", filename)
        return

    plots.someplots(curves_x, curves_y, name=os.path.join(dirname, filename), ylog=True, formatsequence=formatsequence, legendsequence=legendsequence, xtitle=xtitle, ytitle=ytitle, multix=True, yrange=yrange, title=title)

def acomparer(infile, nentry=1000, ifhdf=True, conf='DEFAULT', nocalc=False, trange=None, xrange=None):
    '''
    compares the structure of the flow to the analytic solution by B&S76
    '''

    print ("\n In acomparer():")
    # nentry can be either one entry number or an array/list containing
    # the first and last entry numbers
    sintry = size(nentry)

    rstar = config[conf].getfloat('rstar')
    rstarg = rstar
    m1 = config[conf].getfloat('m1')
    mu30 = config[conf].getfloat('mu30')
    mdot = config[conf].getfloat('mdot') * 4. * pi
    afac = config[conf].getfloat('afac')

    mass1 = config[conf].getfloat('m1')
    #tscale = config[conf].getfloat('tscale') * mass1
    tscale_s = units.tscale_s(config, conf)
    rhoscale = config[conf].getfloat('rhoscale') / mass1
    realxirad = config[conf].getfloat('xirad')
    mow = config[conf].getfloat('mow')

    b12 = 2. * mu30 * (rstar * m1 / 6.8)**(-3)
    umag = units.umag_calc_polar(b12, m1)

    betacoeff = config[conf].getfloat('betacoeff') * m1**(-0.25) / mow

    if not nocalc:

        if ifhdf:

            inhdf = infile + '.hdf5'

            if sintry <= 1:

                # If the requested entry does not exist, use the next available entry
                while hdf.check_entry_exists(inhdf, nentry) == 0:
                    nentry = nentry + 1

                entry, t, l, xp, sth, rhop, up, vp, qloss, glo, ediff = hdf.read(inhdf, nentry)

                t = t * tscale_s
                betap = Fbeta(rhop, up, betacoeff)

                dv = zeros_like(vp, dtype=float)
                du = zeros_like(up, dtype=float)
                drho = zeros_like(rhop, dtype=float)
                dbeta = zeros_like(betap, dtype=float)
                dqloss = zeros_like(qloss, dtype=float)

                nentries = 1
                tstart = t
                tend = t

            else:

                nentry = asarray(nentry, dtype=int)

                num_entries = hdf.num_bunches_total(inhdf)

                if nentry[0] < 0:
                    nentry[0] = 0

                if nentry[1] >= num_entries:
                    print("in acomparer: setting required range to the maximum available entries")
                    nentry[1] = num_entries - 1

                if nentry[0] > nentry[1]:
                    raise ValueError("Invalid entry range in acomparer: first entry is larger than last entry")

                entry_ar = hdf.stored_entries_nums(inhdf)

                rhop_sum = None
                up_sum = None
                vp_sum = None
                qloss_sum = None
                beta_sum = None

                rho2_sum = None
                up2_sum = None
                vp2_sum = None
                qloss2_sum = None
                beta2_sum = None

                nentries = 0
                selected_times = []

                for entry_number in entry_ar:

                    if entry_number < nentry[0] or entry_number > nentry[1]:
                        continue

                    entry, t1, l, xp, sth, rho1, up1, vp1, qloss1, glo, ediff = hdf.read(inhdf, entry_number)

                    t1 = t1 * tscale_s

                    if trange is not None and not (trange[0] <= t1 <= trange[1]):
                        continue

                    beta1 = Fbeta(rho1, up1, betacoeff)

                    if nentries == 0:
                        rhop_sum = zeros_like(rho1, dtype=float)
                        up_sum = zeros_like(up1, dtype=float)
                        vp_sum = zeros_like(vp1, dtype=float)
                        qloss_sum = zeros_like(qloss1, dtype=float)
                        beta_sum = zeros_like(beta1, dtype=float)

                        rho2_sum = zeros_like(rho1, dtype=float)
                        up2_sum = zeros_like(up1, dtype=float)
                        vp2_sum = zeros_like(vp1, dtype=float)
                        qloss2_sum = zeros_like(qloss1, dtype=float)
                        beta2_sum = zeros_like(beta1, dtype=float)

                    rhop_sum += rho1
                    up_sum += up1
                    vp_sum += vp1
                    qloss_sum += qloss1
                    beta_sum += beta1

                    rho2_sum += rho1**2
                    up2_sum += up1**2
                    vp2_sum += vp1**2
                    qloss2_sum += qloss1**2
                    beta2_sum += beta1**2

                    selected_times.append(t1)
                    nentries += 1
                    t = t1

                if nentries == 0:
                    raise ValueError("No entries selected in acomparer; check nentry and trange")

                rhop = rhop_sum / float(nentries)
                up = up_sum / float(nentries)
                vp = vp_sum / float(nentries)
                qloss = qloss_sum / float(nentries)
                betap = beta_sum / float(nentries)

                rho_variance = rho2_sum / float(nentries) - rhop**2
                up_variance = up2_sum / float(nentries) - up**2
                vp_variance = vp2_sum / float(nentries) - vp**2
                qloss_variance = qloss2_sum / float(nentries) - qloss**2
                beta_variance = beta2_sum / float(nentries) - betap**2

                rho_variance = maximum(rho_variance, 0.0)
                up_variance = maximum(up_variance, 0.0)
                vp_variance = maximum(vp_variance, 0.0)
                qloss_variance = maximum(qloss_variance, 0.0)
                beta_variance = maximum(beta_variance, 0.0)

                drho = sqrt(rho_variance)
                du = sqrt(up_variance)
                dv = sqrt(vp_variance)
                dqloss = sqrt(qloss_variance)
                dbeta = sqrt(beta_variance)

                tstart = selected_times[0]
                tend = selected_times[-1]

            # Continue with the rest of your original acomparer() code here.
        else:
            sintry=0
            xp, qp = readtireout(infile, ncol = [3, 2, 1])

    else:
        lines = loadtxt(os.path.dirname(infile) + '/avprofile.dat', comments='#')
        xp = lines[:,0] ; vp = lines[:,1] ; up = lines[:,2] ; betap = lines[:,3] ; tempp = lines[:,4] ; rhop = lines[:,5] ; qloss = lines[:,6]
        dv = lines[:,7] ; du = lines[:,8] ; dbeta = lines[:,9] ; dqloss = lines[:,10]
        # since /avprofile.dat does not have drho data ->
        drho = zeros_like(rhop, dtype=float)
        sintry = 0

        #  betap = pratp / (1.+pratp)

    geofile = os.path.dirname(infile)+"/geo.dat"
    r, theta, alpha, across, l, delta = geo.gread(geofile)
    perimeter = 2. * (across/delta + 2.*delta)
    
    qloss /= perimeter ; dqloss /= perimeter # flux from unit surface area
    
    umagtar = umag * (1.+3.*cos(theta)**2)/4. * (r/rstar)**(-6.)

    BSgamma = (across/delta**2)[0]/mdot*rstar / (realxirad/1.5)
    # umag is magnetic pressure
    # b12 = 2.*mu30*(rstar*m1/6.8)**(-3) # dipolar magnetic field on the pole, 1e12Gs units
    # umag = b12**2*2.29e6*m1
    
    BSeta = (8./21./sqrt(2.)*umag*3. * (realxirad/1.5))**0.25*sqrt(delta[0])/(rstar)**0.125

    # print("check BSeta 1 = "+str(BSeta))
    # BSeta = (8./21./sqrt(2.)*umagtar[0]*3. * (realxirad/1.5))**0.25*sqrt(delta[0])/(rstar)**0.125
    # print("check BSeta2 = "+str(BSeta))
    # xs, BSbeta = bs.xis(BSgamma, BSeta, x0=4., ifbeta = True)

    BSr, BSv, BSu = bs.BSsolution(BSgamma, BSeta)
    print("finite BSr =", isfinite(BSr).sum(), "of", size(BSr))
    print("finite BSv =", isfinite(BSv).sum(), "of", size(BSv))
    print("finite BSu =", isfinite(BSu).sum(), "of", size(BSu))
    BSv *= -1./sqrt(rstar)
    BSumagtar = umag * BSr**(-6.)

    BSu = 3. * BSu/BSu[0] * BSr**6
    tempg = (BSu * BSumagtar / mass1)**(0.25) * 3.35523 # keV
    tempp = (up * umagtar / mass1)**(0.25) * 3.35523 # keV
    
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
        fout.write("# R  -- v -- u -- Prat -- T -- rho -- qloss -- dv -- du -- dbeta -- dqloss \n")
        nx = size(xp)
        for k in arange(nx):
            s = str(xp[k]) + " " + str(vp[k]) + " " + str((up/3/umagtar)[k]) + " " + str(betap[k]) + " " + str(tempp[k]) + " " + str(rhop[k]) + " " + str(qloss[k]) + " " + str(dv[k])+ " "+str((du/3/umagtar)[k])+" "+str(dbeta[k])+" "+str(dqloss[k])+"\n"
            if (DEBUG) : print(s)
            fout.write(s)
            fout.flush()
        fout.close()

    if ifplot:

        if (sintry > 1) or nocalc:
            print("len(r/rstar) =", len(r/rstar))
            print("len(BSr)     =", len(BSr))
            print("len(vp)      =", len(vp))
            print("len(BSv)     =", len(BSv))
            print("len(dv)      =", len(dv))
            print(str(sintry)+' = sintry')

            finite_BS = isfinite(BSr) & isfinite(BSv) & (BSr > 0.0) & (-BSv > 0.0)

            if finite_BS.sum() < 2:
                print("Warning: BSsolution contains no valid positive velocity values; skipping acomparer velocity plot")
            else:
                BSr_plot = BSr[finite_BS]
                BSv_plot = BSv[finite_BS]

                finite_sim = isfinite(r) & isfinite(vp) & isfinite(dv) & (r > 0.0) & (-vp > 0.0) & (-vp + dv > 0.0) & (-vp - dv > 0.0)

                if finite_sim.sum() < 2:
                    print("Warning: simulation velocity arrays contain no valid positive values; skipping acomparer velocity plot")
                else:
                    r_plot = r[finite_sim]
                    vp_plot = vp[finite_sim]
                    dv_plot = dv[finite_sim]

                    y_values = concatenate([-vp_plot, -vp_plot + dv_plot, -vp_plot - dv_plot, -BSv_plot, 1.0/sqrt(r_plot), 1.0/sqrt(r_plot)/7.0])
                    y_values = y_values[isfinite(y_values) & (y_values > 0.0)]

                    if y_values.size < 2:
                        print("Warning: no valid positive y values for logarithmic velocity plot")
                    else:
                        # plot_log_comparison(dirname, 'acompare_v', r_plot/rstar, -BSv_plot, r_plot/rstar, -vp_plot, dv_plot, extra_curves=[(r_plot/rstar, 1.0/sqrt(r_plot)), (r_plot/rstar, 1.0/sqrt(r_plot)/7.0)], formatsequence=['r--', 'k-', 'r:', 'r:', 'b:', 'b:'], legendsequence=['BS', 'simulation', 'simulation + error', 'simulation - error', r'$r^{-1/2}$', r'$r^{-1/2}/7$'], xtitle=r'$R/R_{\rm NS}$', ytitle=r'$-v/c$', title=conf)

                        print("len(BSr_plot) =", len(BSr_plot), "len(BSv_plot) =", len(BSv_plot))
                        print("len(r_plot) =", len(r_plot), "len(vp_plot) =", len(vp_plot), "len(dv_plot) =", len(dv_plot))

                        plot_log_comparison(dirname, 'acompare_v', BSr_plot, -BSv_plot, r_plot/rstar, -vp_plot, dv_plot, extra_curves=[(r_plot/rstar, 1.0/sqrt(r_plot)), (r_plot/rstar, 1.0/sqrt(r_plot)/7.0)], formatsequence=['r--', 'k-', 'r:', 'r:', 'b:', 'b:'], legendsequence=['BS', 'simulation', 'simulation + error', 'simulation - error', r'$r^{-1/2}$', r'$r^{-1/2}/7$'], xtitle=r'$R/R_{\rm NS}$', ytitle=r'$-v/c$', title=conf)


                        valid_u_bs = isfinite(BSr) & isfinite(BSu) & (BSr > 0.0) & (BSu > 0.0)
                        valid_u_sim = isfinite(r) & isfinite(up) & isfinite(du) & (r > 0.0) & (up > 0.0) & (up+du > 0.0) & (up-du > 0.0)
                        BSr_u_plot = BSr[valid_u_bs]
                        BSu_plot = BSu[valid_u_bs]
                        r_u_plot = r[valid_u_sim]
                        up_plot = up[valid_u_sim]
                        du_plot = du[valid_u_sim]

                        plot_log_comparison(dirname, 'acompare_u', BSr_u_plot, BSu_plot/3.0, r_u_plot/rstar, up_plot, du_plot, extra_curves=[(r_u_plot/rstar, up_plot*0.0+1.0)], formatsequence=['k-', 'r--', 'r:', 'r:', 'b:'], legendsequence=['BS', 'simulation', 'simulation + error', 'simulation - error', '1'], xtitle=r'$R/R_{\rm NS}$', ytitle=r'$P/P_{\rm mag}$', title=conf)

                        valid_beta_bs = isfinite(BSr) & isfinite(betag) & (BSr > 0.0) & (betag > 0.0)
                        valid_beta_sim = isfinite(r) & isfinite(betap) & isfinite(dbeta) & (r > 0.0) & (betap > 0.0) & (betap+dbeta > 0.0) & (betap-dbeta > 0.0)
                        BSr_beta_plot = BSr[valid_beta_bs]
                        betag_plot = betag[valid_beta_bs]
                        r_beta_plot = r[valid_beta_sim]
                        betap_plot = betap[valid_beta_sim]
                        dbeta_plot = dbeta[valid_beta_sim]

                        plot_log_comparison(dirname, 'acompare_p', BSr_beta_plot, betag_plot, r_beta_plot/rstar, betap_plot, dbeta_plot, formatsequence=['k-', 'r--', 'r:', 'r:'], legendsequence=['BS', 'simulation', 'simulation + error', 'simulation - error'], xtitle=r'$R/R_{\rm NS}$', ytitle=r'$\beta$', title=conf)

                        valid_rho_bs = isfinite(BSr) & isfinite(BSrho) & (BSr > 0.0) & (BSrho > 0.0)
                        valid_rho_sim = isfinite(r) & isfinite(rhop) & isfinite(drho) & (r > 0.0) & (rhop > 0.0) & (rhop+drho > 0.0) & (rhop-drho > 0.0)
                        BSr_rho_plot = BSr[valid_rho_bs]
                        BSrho_plot = BSrho[valid_rho_bs]
                        r_rho_plot = r[valid_rho_sim]
                        rhop_plot = rhop[valid_rho_sim]
                        drho_plot = drho[valid_rho_sim]
                        plot_log_comparison(dirname, 'acompare_rho', BSr_rho_plot, BSrho_plot, r_rho_plot/rstar, rhop_plot, drho_plot, formatsequence=['k-', 'r--', 'r:', 'r:'], legendsequence=['BS', 'simulation', 'simulation + error', 'simulation - error'], xtitle=r'$R/R_{\rm NS}$', ytitle=r'$\rho/\rho^*$', title=conf)
        else:
            plots.someplots([r/rstar, BSr, r/rstar, r/rstar], [-vp, -BSv, 1./sqrt(r), 1./sqrt(r)/7.], name=dirname+'/acompare_v', ylog=True, formatsequence=['r--', 'k-', 'b:', 'b:', 'r:', 'r:'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$-v/c$', multix = True, yrange= [-BSv.max()/2., -BSv.min()*7.*2.], title=conf)
            plots.someplots([BSr, r/rstar, r/rstar], [BSu/3, up, up*0.+1.], name=dirname+'/acompare_u', ylog=True, formatsequence=['k-', 'r--', 'b:'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$P/P_{\rm mag}$', multix = True, yrange = [BSu.min()/10., maximum(BSu.max()*2.,5.)], title=conf)
            plots.someplots([BSr, r/rstar], [BSrho, rhop], name=dirname+'/acompare_rho', ylog=True, formatsequence=['k-', 'r--'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$\rho/\rho^*$', multix = True, title=conf)
        plots.someplots([BSr, r/rstar], [tempg, tempp], name=dirname+'/acompare_T', ylog=True, formatsequence=['k-', 'r--'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$T$, {\rm keV}', multix = True, title=conf)
        teff = 4.74501 * mass1**(-0.25) * qloss**0.25 # keV
        dteff = dqloss / qloss * teff * 0.25
        plots.someplots([r/rstar], [teff], name=dirname+'/acompare_Teff', ylog=False, formatsequence=['k-', 'r--'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$T_{\rm eff}$, keV', multix = True, title=conf)
 #        plots.someplots([BSr, r/rstar], [betag, betap], name=dirname+'/acompare_p', ylog=True, formatsequence=['k-', 'r--'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$\beta$', multix = True)

def avcompare_list(dirlist, rrange = None, conf = 'DEFAULT', tauscale = None):
    '''
    tauscale is the number of the profile used to calculate the optical depth
    '''
    mass1 = config[conf].getfloat('m1')

    nf = size(dirlist)
    
    xlist = [] ; vlist = [] ; ulist = [] ; plist = [] ; qlist = [] ; tefflist = []
    
    if tauscale is not None:
        taulist = []
    
    for k in arange(nf):
        lines = loadtxt(dirlist[k] + '/avprofile.dat', comments = '#')
        xp = lines[:,0] ; vp = lines[:,1] ; up = lines[:,2] ; pratp = lines[:,3] ; tempp = lines[:,4] ; rhop = lines[:,5] ; qloss = lines[:,6]
        teff = 4.74501 * mass1**(-0.25) * qloss**0.25 # keV
        xlist.append(xp) ; vlist.append(vp) ; ulist.append(up)  ; plist.append(pratp/(1.+pratp))
        qlist.append(qloss) ; tefflist.append(teff)
        
        if tauscale is not None:
            geofile = dirlist[k]+'/geo.dat'
            r, theta, alpha, across, l, delta = geo.gread(geofile)
            # r/r[0] shd coincide with xp
            tau = cumnumerical_trapezoid(rhop, x=r)
            # position of the shock front:
            dvdl = abs((vp[1:]-vp[:-1])/(l[1:]-l[:-1]))
            wfront = dvdl.argmax()
            tau -= tau[wfront]
            taulist.append(tau)
            if k == tauscale:
                tau2x = interp1d(tau, (xp[1:]+xp[:-1])/2., bounds_error = False, fill_value = (tau[0], tau[-1]))
                x2tau = interp1d((xp[1:]+xp[:-1])/2., tau, bounds_error = False, fill_value = (xp[0], xp[-1]))

    if rrange is None:
        plots.someplots(xlist, vlist, multix=True, formatsequence=['r--', 'k-', 'g:', 'b-.'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$-v/c$', name='avtwo_v', inchsize=[5,3], title=conf)
        plots.someplots(xlist, ulist, multix=True, formatsequence=['r--', 'k-', 'g:', 'b-.'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$P/P_{\rm mag}$', name='avtwo_u', ylog=True, inchsize=[5,3], title=conf)
        plots.someplots(xlist, plist, multix=True, formatsequence=['r--', 'k-', 'g:', 'b-.'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$\beta$', name='avtwo_p', ylog=True, inchsize=[5,3], title=conf)
        plots.someplots(xlist, qlist, multix=True, formatsequence=['r--', 'k-', 'g:', 'b-.'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$Q^-$', name = 'avtwo_q', xlog = True, ylog = True, inchsize=[5,3], title=conf)
        plots.someplots(xlist, tefflist, multix=True, formatsequence=['r--', 'k-', 'g:', 'b-.'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$T_{\rm eff}$, {\rm keV}', name = 'avtwo_teff', xlog = True, ylog = False, inchsize=[5,3], title=conf)
    else:
        plots.someplots(xlist, vlist, multix=True, formatsequence=['r--', 'k-', 'g:', 'b-.'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$-v/c$', name='avtwo_v', xrange = rrange, xlog = False, inchsize=[5,3], title=conf)
        urange = [ulist[-1][(xlist[-1]>rrange[0]) & (xlist[-1]<rrange[1])].min()*0., ulist[-1][(xlist[-1]>rrange[0]) & (xlist[-1]<rrange[1])].max()*1.1]
        plots.someplots(xlist, ulist, multix=True, formatsequence=['r--', 'k-', 'g:', 'b-.'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$P/P_{\rm mag}$', name='avtwo_u', xrange = rrange, xlog = False, yrange = urange, inchsize=[5,3], title=conf)
        qrange = [qlist[-1][(xlist[-1]>rrange[0]) & (xlist[-1]<rrange[1])].min()*0., qlist[-1][(xlist[-1]>rrange[0]) & (xlist[-1]<rrange[1])].max()*1.1]
        teffrange = [tefflist[-1][(xlist[-1]>rrange[0]) & (xlist[-1]<rrange[1])].min()*0., tefflist[-1][(xlist[-1]>rrange[0]) & (xlist[-1]<rrange[1])].max()*1.1]
        plots.someplots(xlist, qlist, multix=True, formatsequence=['r--', 'k-', 'g:', 'b-.'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$Q^-$', name = 'avtwo_q', xrange = rrange, yrange = qrange, xlog = False, inchsize=[5,3], title=conf)
        if tauscale is not None:
            plots.someplots(xlist, tefflist, multix=True, formatsequence=['r--', 'k-', 'g:', 'b-.'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$T_{\rm eff}$, {\rm keV}', name = 'avtwo_teff', xrange = rrange, yrange = teffrange, xlog = False, inchsize=[5,3], secaxfunpair = (x2tau, tau2x), title=conf)
            print(tau2x(-1.), tau2x(0.), tau2x(1.))
        else:
            plots.someplots(xlist, tefflist, multix=True, formatsequence=['r--', 'k-', 'g:', 'b-.'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$T_{\rm eff}$, {\rm keV}', name = 'avtwo_teff', xrange = rrange, xlog = False, inchsize=[5,3], title=conf)

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
    mdot = config[conf].getfloat('mdot') * 4.*pi
    afac = config[conf].getfloat('afac')
    mass1 = config[conf].getfloat('m1')
    #tscale = config[conf].getfloat('tscale') * mass1
    tscale_s = units.tscale_s(config, conf)
    rhoscale = config[conf].getfloat('rhoscale') / mass1
    realxirad = config[conf].getfloat('xirad')
    mow = config[conf].getfloat('mow')
    b12 = 2.*mu30*(rstar*m1/6.8)**(-3) # dipolar magnetic field on the pole, 1e12Gs units
    umag = units.umag_calc_polar(b12,m1)
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
                print("time range = "+str(tstart*tscale_s)+".."+str(tend*tscale_s)+"s")
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
        plots.someplots([xg, xp, xp], [ug, up, up*0.+3.], name=outdir+'BScompare_u', ylog=True, formatsequence=['k-', 'r--', 'b:'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$U/U_{\rm mag}$', multix = True, title=conf)
        if sintry >= 1:
            plots.someplots([xp, xg, xp, xp, xp, xp], [-vp, -vg, 1./sqrt(rstar*xp), 1./sqrt(rstar*xp)/7., -vp+dv, -vp-dv], name=outdir+'BScompare_v', ylog=True, formatsequence=['r--', 'k-', 'b:', 'b:', 'r:', 'r:'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$-v/c$', multix = True, title=conf)
        else:
            plots.someplots([xp, xg, xp, xp], [-vp, -vg, 1./sqrt(rstar*xp), 1./sqrt(rstar*xp)/7.], name=outdir+'BScompare_v', ylog=True, formatsequence=['r--', 'k-', 'b:', 'b:', 'r:', 'r:'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$-v/c$', multix = True, title=conf)
        plots.someplots([xg, xp], [betag, betap], name=outdir+'BScompare_p', ylog=True, formatsequence=['k-', 'r--'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$\beta$', multix = True, title=conf)
        plots.someplots([xg, xp], [tempg, tempp], name=outdir+'BScompare_T', ylog=True, formatsequence=['k-', 'r--'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$T$, {\rm keV}', multix = True, title=conf)
        plots.someplots([xg, xp], [rhog/rhoscale, rhop], name=outdir+'BScompare_rho', ylog=True, formatsequence=['k-', 'r--'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$\rho/\rho^*$', multix = True, title=conf)
        plots.someplots([xg, xp], [virialbetaG, virialbetaP], name=outdir+'BScompare_virbeta', ylog=False, formatsequence=['k-', 'r--'], xtitle = r'$R/R_{\rm NS}$', ytitle =  r'$\beta_{\rm BS}$', multix = True, title=conf)
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



def dynspec(infile='out/flux', ntimes=10, nbins=100, binlogscale=False, deline = False, ncol = 5, iffront = False, stnorm = False, fosccol = None, simfreq = None, conf = 'DEFAULT', trange = None):

    '''
    makes a dynamic spectrum by making Fourier in each of the "ntimes" time bins. Fourier PDS is binned to "nbins" bins
    "ncol" is the number of data column in the input file (the last one is taken by default)
    
    '''

    print("\n In dynspec():")
    rstar = config[conf].getfloat('rstar')
    m1 = config[conf].getfloat('m1')
    #tscale = config[conf].getfloat('tscale') * m1
    tscale_s = units.tscale_s(config, conf)
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
    if trange is not None:
        w = (t> trange[0]) * (t<trange[1])
        t=t[w] ;  l=l[w]
        if fosccol is not None:
            fosc = fosc[w]
        # print(t.min(), t.max())
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
    freq1=1./(t.max()-t.min())*double(ntimes)/2. ; freq2 = minimum(1./median(t[1:]-t[:-1])/2., 1500.)
    # print(freq1, freq2)
    # ii =input('F')
    if(binlogscale):
        binfreq=logspace(log10(freq1), log10(freq2), num=nbins+1)
    else:
        binfreq=linspace(freq1, freq2, nbins+1)
    freq1 = 0.
    binfreqc=(binfreq[1:]+binfreq[:-1])/2.
    binfreqs=(binfreq[1:]-binfreq[:-1])/2.

    pds2 = full((ntimes, nbins), nan, dtype=double)
    dpds2 = full((ntimes, nbins), nan, dtype=double)

    t2 = full((ntimes + 1, nbins + 1), nan, dtype=double)
    nbin = zeros((ntimes, nbins), dtype=double)
    binfreq2 = full((ntimes + 1, nbins + 1), nan, dtype=double)

    fmax = full(ntimes, nan, dtype=double)
    dfmax = full(ntimes, nan, dtype=double)
    xmean = full(ntimes, nan, dtype=double)
    xstd = full(ntimes, nan, dtype=double)

    if fosccol is not None:
        foscmean = full(ntimes, nan, dtype=double)
        foscstd = full(ntimes, nan, dtype=double)
    # average PDS
    taver = [0.25, 0.4] # averaging interval for meansp
    
    fdyns=open(infile+'_dyns.dat', 'w')
    ffreqmax = open(infile+'_fmax.dat', 'w')
    for kt in arange(ntimes):

        wt = ( (t >= tbin[kt]) & (t < tbin[kt + 1]) & isfinite(t) & isfinite(xs) )

        tt = t[wt]
        lt = xs[wt]

        # Remove invalid values explicitly
        good = isfinite(tt) & isfinite(lt)
        tt = tt[good]
        lt = lt[good]

        nt = size(lt)

        # A Fourier transform and linear fit require at least two points
        if nt < 2:
            print( "dynspec: skipping time bin", kt, "because it contains only", nt, "valid point(s)" )
            continue

        # Time values must not all be identical
        if ptp(tt) <= 0.0:
            print( "dynspec: skipping time bin", kt, "because all times are identical" )
            continue

        # Fit using time relative to the bin mean.
        # This is much better conditioned than fitting directly against tt.
        tt0 = tt - tt.mean()

        if deline:
            if nt >= 2:
                pfit = polyfit(tt0, lt, 1)
                trend = pfit[0] * tt0 + pfit[1]
                lt_detrended = lt - trend
            else:
                continue
        else:
            lt_detrended = lt - lt.mean()

        # Avoid invalid normalization
        if stnorm:
            lt_std = lt_detrended.std()

            if not isfinite(lt_std) or lt_std == 0.0:
                print( "dynspec: skipping time bin", kt, "because its standard deviation is zero" )
                continue

            fsp = fft.rfft(lt_detrended)
            fsp /= lt_std

        else:
            lt_sum = lt.sum()

            if not isfinite(lt_sum) or lt_sum == 0.0:
                print("dynspec: skipping time bin", kt, "because the original signal sum is zero")
                continue

            fsp = fft.rfft(lt_detrended)
            fsp *= 2.0 / lt_sum

        nt = size(lt)
        print("time bin", kt, "contains", nt, "points", "t range =", tt.min(), tt.max(), "signal range =", lt.min(), lt.max())
        dt_values = diff(tt)
        dt_values = dt_values[ isfinite(dt_values) & (dt_values > 0.0) ]

        if size(dt_values) == 0:
            print( "dynspec: skipping time bin", kt, "because no positive time interval exists" )
            continue

        dt = median(dt_values)

        freq = fft.rfftfreq(nt, dt)
        pds = real(fsp * freq)**2 + imag(fsp * freq)**2

        t2[kt,:]=tbin[kt] ; t2[kt+1,:]=tbin[kt+1] 
        binfreq2[kt,:]=binfreq[:] ; binfreq2[kt+1,:]=binfreq[:] 
        for kb in arange(nbins):
            wb=((freq>=binfreq[kb]) & (freq<=binfreq[kb+1]))
            nbin[kt,kb] = wb.sum()
            #            print("size(f) = "+str(size(freq)))
            #print("size(pds) = "+str(size(pds)))
            if wb.sum() > 1:
                pds2[kt, kb]=pds[wb].mean() ; dpds2[kt, kb]=pds[wb].std()
            # ascii output:
            fdyns.write(str(tcenter[kt])+' '+str(binfreq[kb])+' '+str(binfreq[kb+1])+' '+str(pds2[kt,kb])+' '+str(dpds2[kt,kb])+" "+str(nbin[kt,kb])+"\n")
            # print(str(tcenter[kt])+' '+str(binfreq[kb])+' '+str(binfreq[kb+1])+' '+str(pds2[kt,kb])+' '+str(dpds2[kt,kb])+" "+str(nbin[kt,kb])+"\n")
            # ii = input('P')
        # finding maximum:
        valid_frequency_bins = ((nbin[kt, :] > 1) & isfinite(pds2[kt, :]) &             (arange(nbins) > 0))

        if not any(valid_frequency_bins):
            print("dynspec: no valid frequency bins in time bin", kt)
            continue

        candidate_indices = where(valid_frequency_bins)[0]
        nfmax = candidate_indices[argmax(pds2[kt, candidate_indices])]

        fmax[kt] = (binfreq[nfmax] + binfreq[nfmax + 1] ) / 2.0

        dfmax[kt] = (binfreq[nfmax + 1] - binfreq[nfmax]) / 2.0

        ffreqmax.write(str(tcenter[kt])+" "+str(tsize[kt])+" "+str(fmax[kt])+" "+str(dfmax[kt])+"\n")

        xmean[kt] = lt.mean()
        xstd[kt] = lt.std()

        if fosccol is not None:
            fosc_bin = fosc[wt][good]

            if size(fosc_bin) > 0:
                foscmean[kt] = fosc_bin.mean()
                foscstd[kt] = fosc_bin.std()
    fdyns.close()
    ffreqmax.close()
    valid_plot = isfinite(xmean) & isfinite(xstd) & isfinite(fmax) & isfinite(dfmax) & (xmean > 0.0) & (fmax > 0.0) & (xstd >= 0.0) & (dfmax >= 0.0)
    if ifplot:
        # frange = plots.plot_dynspec(t2, binfreq2, pds2, outfile=infile+'_dyns', nbin=nbin, logscale=True)
        print("finite xmean =", isfinite(xmean).sum())
        print("finite fmax  =", isfinite(fmax).sum())
        print("valid plot points =", valid_plot.sum())
        if valid_plot.sum() < 2:
            print("dynspec: not enough valid points for plotting")
            return None
        print("xmean =", xmean)
        print("fmax  =", fmax)

        valid_pds = (nbin > 1) & isfinite(pds2) & (pds2 > 0.0)

        if any(valid_pds):
            frange = plots.plot_dynspec(t2, binfreq2, pds2, outfile=infile+'_dyns', nbin=nbin, logscale=True)
        else:
            print("dynspec: no valid PDS values; skipping dynamic-spectrum plot")
            frange = None
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
            umag = units.umag_calc_polar(b12,m1)
            BSeta = (8./21./sqrt(2.)*umag*3. * (realxirad/1.5))**0.25*sqrt(delta0)/(rstar)**0.125
            print("BSgamma = "+str(BSgamma))
            print("BSeta = "+str(BSeta))
            xs, BSbeta = bs.xis(BSgamma, BSeta, x0=4., ifbeta = True)
            print("xs =", xs)
            if not isfinite(BSbeta) or not all(isfinite(asarray(xs))):
                print("Warning: bs.xis returned invalid values")
            valid_x = isfinite(xmean) & isfinite(xstd) & (xmean > 0.0) & (xmean >= geo_r.min()/rstar) & (xmean <= geo_r.max()/rstar)

            if valid_x.sum() < 1:
                print("dynspec: no valid shock positions for analytical BS curve")
                xtmp = array([])
                tth = array([])
                fth = array([])
            else:
                xtmp = xmean[valid_x]
                tth_valid = tscale_s * rstar**1.5 * m1 * bs.dtint(BSgamma, xtmp, cthfun)
                tth = full(size(xmean), nan, dtype=double)
                tth[valid_x] = tth_valid
                fth = full(size(xmean), nan, dtype=double)
                fth[valid_x] = 1.0 / tth_valid
                print("valid xmean values =", xtmp)
                print("tth =", tth_valid)
            # 0.0227364 / xirad / sqrt(rstar*xmean)/rstar / mdot * (1./delta_s+2.*delta_s / across_s)**2 * 2.03e5/m1 # Hz
            if (size(unique(xmean))>5):
                goodx = isfinite(xmean) & isfinite(fmax) & isfinite(xstd) & (xmean > xstd) & (xmean > 0.0) & (fmax > 0.0) & (tcenter > tcenter[2])

                if goodx.sum() >= 2 and ptp(log(xmean[goodx])) > 0.0:
                    logx = log(xmean[goodx])
                    logf = log(fmax[goodx])
                    logx0 = logx - logx.mean()
                    pfit, pcov = polyfit(logx0, logf, 1, cov=True)
                    print("dln(f)/dln(R_s) =", pfit[0], "+/-", sqrt(maximum(pcov[0, 0], 0.0)))
                else:
                    print("Not enough valid points for the logarithmic polyfit")

            if fosccol is not None:
                print("foscmean =", foscmean)
                fth = foscmean.copy()
                fth[~isfinite(fth) | (fth <= 0.0)] = nan
                valid_line = isfinite(xmean) & isfinite(fth) & (xmean > 0.0) & (fth > 0.0)
                xtmp = xmean[valid_line]
                fth_plot = fth[valid_line]
            else:
                valid_line = valid_x & isfinite(fth) & (fth > 0.0)
                xtmp = xmean[valid_line]
                fth_plot = fth[valid_line]

            addline_data = [xtmp, fth_plot] if xtmp.size >= 2 else None



            #If the number of valid points is zero or one, reduce the number of time bins:
            #pp.dynspec(infile=outdir+'/sfront', nbins=20, ntimes=5, iffront=True, deline=False, fosccol=-1, trange=[0., 0.5], stnorm=False)

            valid_plot = isfinite(xmean) & isfinite(xstd) & isfinite(fmax) & isfinite(dfmax) & (xmean > 0.0) & (fmax > 0.0) & (xstd >= 0.0) & (dfmax >= 0.0)

            if valid_plot.sum() < 2:
                print("dynspec: not enough valid points for the shock-frequency plot")
                print("  valid xmean points =", valid_plot.sum())
            else:
                x_plot = xmean[valid_plot]
                xstd_plot = xstd[valid_plot]
                fmax_plot = fmax[valid_plot]
                dfmax_plot = dfmax[valid_plot]

                positive_lower = x_plot - xstd_plot
                positive_lower = positive_lower[positive_lower > 0.0]

                if positive_lower.size > 0:
                    xmin_plot = builtins.max(float(quantile(positive_lower, 0.2)), 1.0)
                else:
                    xmin_plot = builtins.max(float(x_plot.min()), 1.0)

                xmax_plot = builtins.min(float(x_plot.max()), float(geo_r.max() / rstar))

                if xmax_plot <= xmin_plot:
                    xmin_plot = float(x_plot.min())
                    xmax_plot = float(x_plot.max())

                valid_line = valid_x & isfinite(fth) & (fth > 0.0)

                if valid_line.sum() >= 2:
                    addline_data = [xmean[valid_line], fth[valid_line]]
                else:
                    print("dynspec: no valid analytical or oscillation-frequency line")
                    addline_data = None

                plots.errorplot(x_plot, xstd_plot, fmax_plot, dfmax_plot, outfile=infile+'_xfmax', xtitle=r'$R_{\rm shock}/R_{*}$', ytitle='$f$, Hz', yrange=frange, xrange=[xmin_plot, xmax_plot], addline=addline_data, xlog=True, ylog=False, lticks=[4, 5, 6, 8, 10])
            #original: plots.errorplot(xmean, xstd, fmax, dfmax, outfile=infile+'_xfmax', xtitle=r'$R_{\rm shock}/R_{*}$', ytitle='$f$, Hz', yrange=frange, xrange=[maximum(quantile(xmean[isfinite(xmean)]-xstd[isfinite(xmean)], 0.2), 1.), minimum(xmean[isfinite(xmean)].max(), geo_r.max()/rstar)], addline=[xtmp, fth[valid_x]], xlog=True, ylog=False, lticks=[4, 5, 6, 8, 10])


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
def shock_hdf(ndump, infile = "out/tireout.hdf5", kleap = 5, uvcheck = False, uvcheckfile = 'uvcheck', conf=""):
    '''
    finds the position of the shock in a given entry of the infile
    kleap allows to measure the velocity difference using several cells
    '''
    
    if DEBUG : print("ndump=",ndump)
    
    entryname, t, l, r, sth, rho, u, v, qloss, glo, ediff = hdf.read(infile, ndump)
    if (entryname == -1) : 
        return -1 , 0, 0, 0, 0, 0, 0, 0, 0 
    
    n=size(r)
    if DEBUG : print("n=",n) 
    #    v=medfilt(v, kernel_size=3)
    # v1=savgol_filter(copy(v), 2*kleap+1, 1) # Savitzky-Golay filter
    #find maximal compression:
    
    
    dvdl = (v[kleap:]-v[:-kleap])/(l[kleap:]-l[:-kleap])
    
    wcomp = (dvdl).argmin()
    
    wcomp1 = maximum((wcomp-kleap),0)
    wcomp2 = minimum((wcomp+kleap),n-1)
    if DEBUG : print("maximal compression found at r="+str((r[wcomp1]+r[wcomp2])/2.)+" +/- "+str((-r[wcomp1]+r[wcomp2])/2.)) 
    # print("element No "+str(wcomp1)+" out of "+str(n))
    if isnan(r[wcomp]):
        print("t = "+str(t))
        print(dvdl.min(), dvdl.max())
        print(dvdl[wcomp])
        if ifplot:
            plots.someplots(r[1:], [v[1:], v1[1:], dvdl], name = "shocknan", xtitle=r'$r$', ytitle=r'$v$', xlog=False, formatsequence = ['k.', 'r-', 'b'], title=conf)
        ii=input('r')

    ltot = numerical_trapezoid(qloss[1:], x=l[1:])
    if wcomp1 > 2:
        lbelowshock = numerical_trapezoid(qloss[1:wcomp1], x = l[1:wcomp1])
        #    wnearshock = maximum(wcomp1-10,1):minimum(wcomp1+10, n-1)
        lonshock = numerical_trapezoid(qloss[maximum(wcomp1-10,1):minimum(wcomp1+10, n-1)], x = l[maximum(wcomp1-10,1):minimum(wcomp1+10, n-1)])
        #added by galja MAY 2023 to calculate max flux along the tube at shock
        ediff_shock = ediff[maximum(wcomp1-10,1):minimum(wcomp1+10, n-1)]
        ediff_max_shock  = max(ediff_shock) 
        #sum(ediff_shock)/size(ediff_shock);
        #END added by galja
        
    else:
        lbelowshock = 0.
        lonshock = 0.
        ediff_max_shock = 0.

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
                            vertical = (r[wcomp1]+r[wcomp2])/2., title = conf)
            
    return t, (r[wcomp1]+r[wcomp2])/2.,(-r[wcomp1]+r[wcomp2])/2., v[wcomp1], v[wcomp2], ltot, lbelowshock, lonshock,  -((u*4./3.+v**2/2.)*v)[-1], ediff_max_shock, -((u*4./3.+v**2/2.)*v)[1]  

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
    
import re
def read_p_ratio_from_dat (fname):   
    start = 0 # some starting index
    end = 3 # some ending index
    #print(fname)
    with open(fname) as fin:
        data = fin.readlines()[start:end]
        
    time = re.findall(r"t = (\S+)s", data[0]) #seconds
    values = re.split(r'\s', data[2])
    return (time[0],values[3])
    
def multishock(n1, n2, dn, prefix = "out/tireout", dat = False, conf = None, kleap = 5, xest = 4.):
    '''
    draws the motion of the shock front with time, for a given set of HDF5 entries or ascii outputs
    n2 is the size
    '''
    print ("In multishock():")
    if conf is None:
        conf = 'DEFAULT'
    rstar = config[conf].getfloat('rstar')
    m1 = config[conf].getfloat('m1')
    #tscale = config[conf].getfloat('tscale') * m1
    tscale_s = units.tscale_s(config, conf)
    mu30 = config[conf].getfloat('mu30')
    mdot = config[conf].getfloat('mdot') * 4.*pi
    afac = config[conf].getfloat('afac')
    drrat = config[conf].getfloat('drrat')
    realxirad = config[conf].getfloat('xirad')
    xifac = config[conf].getfloat('xifac')
    b12 = 2.*mu30*(rstar*m1/6.8)**(-3) # dipolar magnetic field on the pole, 1e12Gs units
    
    r_e = config[conf].getfloat('r_e_coeff') * (mu30**2/mdot)**(2./7.)*m1**(-10./7.) * xifac # magnetosphere radius
    

    #compare desired numbers of entries with existing:
    num_entries = hdf.num_bunches_total(prefix+".hdf5")
    if (n1<0) : 
        print ("in multishock set n1",n1," to zero")
        n1 = 0
    if (dn*n2>num_entries) :
        print ("in multishock set n2 to maximum possible ->")
        n2 = int((num_entries - n1) // step)
        
        
    n=arange(n1, n1+dn*n2, dn, dtype=int)
    #entry_ar = hdf.stored_entries_nums (prefix+".hdf5")
    num_available = num_entries - n1

    if num_available <= 0:
        raise ValueError(
            f"No entries available: n1={n1}, num_entries={num_entries}"
        )

    n2 = builtins.min(n2, num_available // dn)

    n = arange(
        n1,
        n1 + dn * n2,
        dn,
        dtype=int
    )

    entry_ar = hdf.stored_entries_nums(prefix + ".hdf5")


    N = size(n)

    t  = full(N, nan, dtype=double)
    s  = full(N, nan, dtype=double)
    ds = full(N, nan, dtype=double)
    dv = full(N, nan, dtype=double)
    v2 = full(N, nan, dtype=double)
    v1 = full(N, nan, dtype=double)

    lc_out       = full(N, nan, dtype=double)
    lc_in        = full(N, nan, dtype=double)
    lc_tot       = full(N, nan, dtype=double)
    lc_part      = full(N, nan, dtype=double)
    lc_nearshock = full(N, nan, dtype=double)
    l_shock_along = full(N, nan, dtype=double)
    compression  = full(N, nan, dtype=double)

    print("size n=",size(n))

    outdir = os.path.dirname(prefix) 
    fluxlines = loadtxt(outdir+"/flux.dat", comments="#", delimiter=" ", unpack=False)
    geometry = loadtxt(outdir+"/geo.dat", comments="#", delimiter=" ", unpack=False)
    
    tf=fluxlines[:,0] ;  ff=fluxlines[:,1]
    th = geometry[:,1] ; r = geometry[:,0]
    cross_section = geometry[:,3]; #galja
    cthfun = interp1d(r/r[0], cos(th)) # we need a function allowing to calculate cos\theta (x)
    across0 = geometry[0,3]  ;   delta0 = geometry[0,5]
    acrosslast =  geometry[-1,3]
    #   acrosslast = pi*4.*afac*drrat * geometry[-1,0]**2 
    
    
    BSgamma = (across0/delta0**2)/mdot*rstar / (realxirad/1.5)
    umag = units.umag_calc_local (b12, m1, th[0])

    
    acrossfun = interp1d(r/rstar, cross_section, bounds_error=False)
        
    # umag is magnetic pressure
    BSeta = (8./21./sqrt(2.) * umag * 3. * (realxirad/1.5))**0.25*sqrt(delta0)/(rstar)**0.125
    print("BSgamma =", BSgamma)
    print("BSeta   =", BSeta)
    print("xest    =", xest)

    if not isfinite(BSgamma) or not isfinite(BSeta):
        raise ValueError(
            f"Invalid input to bs.xis: BSgamma={BSgamma}, BSeta={BSeta}"
        )

    xs, BSbeta = bs.xis(
        BSgamma,
        BSeta,
        x0=xest,
        ifbeta=True
    )

    print("xs      =", xs)

    if not all(isfinite(asarray(xs))) or not isfinite(BSbeta):
        print (
            "**** bs.xis() returned NaN or infinity. "
            f"Inputs were BSgamma={BSgamma}, BSeta={BSeta}, xest={xest}"
        )
    
    
    # @galja For printing into file, store stdout and organize f:
    original_stdout = sys.stdout  # save original  @galja
    save_summary = open(outdir+'/summary.txt', 'a') 
    sys.stdout = save_summary # @galja redirect print to summary 
   
    print("eta = "+str(BSeta))
    print("gamma = "+str(BSgamma))
    print("eta gamma^{1/4} = "+str(BSeta*BSgamma**0.25))
    print("beta = "+str(BSbeta))
    print("predicted shock position = "+str(xs))
    #dt_BS = tscale * rstar**1.5 * bs.dtint(BSgamma,min(xs,max(r/r[0])), cthfun)
    #ii = input("xs") commented by @galja
    # spherization radius
    rsph =1.5*mdot/4./pi
    eqlum = [mdot/rstar*(1.-BSbeta)]
    if size(eqlum) > 0:
        eqlum = eqlum[0]
    print("m1 = "+str(m1))
    print("mdot = "+str(mdot))
    print("rstar = "+str(rstar))
    
    # @galja this print was below:
    print("predicted shock position: xs = "+str(xs)+" (rstar)")
    
    save_summary.close()
    sys.stdout = original_stdout # @galja back to stdout
    
    # iterating to find the cooling radius
    rcool = rcoolfun(geometry, mdot)
    for k in arange(size(n)):
        # some confusion here about entry....
        entry = entry_ar[n[k]]
        #entry = entry_ar[k]
        #originally was entry = n[k]
        #if DEBUG :
        # print("k=",k," n[k]=",n[k],"entry_ar[n[k]]=",entry)
        
        if (dat):
            stmp, dstmp, v1tmp, v2tmp = shock_dat(entry, prefix=prefix, kleap = kleap)
        else:
            ttmp, stmp, dstmp, v1tmp, v2tmp, ltot, lpart, lshock, uvtmp, earshock, uv1 = shock_hdf(entry, infile = prefix+".hdf5", kleap = kleap, uvcheck = (k == (size(n)-1)), uvcheckfile = outdir+"/uvcheck", conf = conf)
            if (ttmp == -1) :
                continue
        s[k] = stmp ; ds[k] = dstmp
        
        
        v1[k] = v1tmp   ; v2[k] =  v2tmp
        dv[k] = v1tmp - v2tmp
        if isfinite(v1tmp) and v1tmp != 0.0:
            compression[k] = v2tmp / v1tmp
        else:
            compression[k] = nan
        lc_tot[k] = ltot ; lc_part[k] = lpart ; 
        lc_nearshock[k] = lshock ; l_shock_along[k] = earshock
        t[k] = ttmp
        lc_out[k] = uvtmp * acrosslast
        lc_in[k] = uv1 * across0 # =0
        
    valid = (
        isfinite(t) &
        isfinite(s) &
        isfinite(ds) &
        isfinite(v1) &
        isfinite(v2) &
        isfinite(lc_tot) &
        isfinite(lc_part)
    )

    if not any(valid):
        raise ValueError(
            "No valid shock entries were read. "
            "Check shock_hdf(), the HDF5 entries, and bs.xis()."
        )

    t = t[valid]
    s = s[valid]
    ds = ds[valid]
    dv = dv[valid]
    v1 = v1[valid]
    v2 = v2[valid]
    lc_out = lc_out[valid]
    lc_in = lc_in[valid]
    lc_tot = lc_tot[valid]
    lc_part = lc_part[valid]
    lc_nearshock = lc_nearshock[valid]
    l_shock_along = l_shock_along[valid]
    compression = compression[valid]
    #print("predicted shock position: xs = "+str(xs)+" (rstar)")
    
    #save_summary = open(outdir+'/summary.txt', 'a') 
    #sys.stdout = save_summary # @galja again print to summary
    #    print("cooling limit: rcool/rstar = "+str(rcool/rstar))
    print("multishock(): flux array lc_tot size is "+str(size(lc_tot)))
    ff /= 4.*pi  ; eqlum /= 4.*pi ; lc_tot /= 4.*pi ; lc_part /= 4.*pi  ; lc_in/=4.*pi; lc_out /= 4.*pi ; lc_nearshock /= 4.*pi
    l_shock_along /= 4.*pi
    t *= tscale_s

    dt_current = tscale_s * rstar**1.5 * m1 * bs.dtint(BSgamma, s, cthfun)
    
    
    if(ifplot):
        finite_lc_part = isfinite(lc_part)

        if not any(finite_lc_part):
            print("No finite lc_part values; skipping plots.")
            ws = array([], dtype=int)
        else:
            lc_part_min = nanmin(lc_part)

            ws = where(
                isfinite(s) &
                isfinite(lc_part) &
                (s > 1.0) &
                (lc_part > lc_part_min)
            )[0]
        n=ws
        #print (" ws=",ws)
        #shock_across = acrossfun(s[ws])
         
        lvent = lc_out - lc_tot +  mdot/rstar/4./pi - mdot/2./r_e/4./pi;
        

        if ifplot and len(ws) > 1:
            magn =  mdot/rstar/4./pi
            plots.someplots(t[ws], [ lc_part[ws],  t[ws]*0.+mdot/rstar/4./pi*(1.-BSbeta),  l_shock_along[ws], lvent[ws]], name = outdir+"/lumshocks_some", xtitle=r'$t$, s', ytitle=r'$L/L_{\rm Edd}$', formatsequence = [ 'r-',   'k:', 'm:','b-'], legendsequence = [ r'$L_{X}$',   r'$(1-\beta_{\rm BS})L_{acc}$',  r'$L_{\rm shock,along}$',r'$L_{\rm vent}$',],inchsize = [5, 4], ylog = True,yrange=[1e-3*magn, magn*3], legend_charsize=10, title=conf)


            plots.someplots(t[ws], [lc_tot[ws], lc_part[ws], lc_out[ws], lc_tot[ws]-lc_part[ws], t[ws]*0.+mdot/rstar/4./pi*(1.-BSbeta), t[ws]*0.+mdot/rstar/4./pi, lc_tot[ws]-mdot/rstar/4./pi*(1.-BSbeta), l_shock_along[ws], lc_nearshock[ws]], name=outdir+"/lumshocks", xtitle=r'$t$, s', ytitle=r'$L/L_{\rm Edd}$', formatsequence=['k-', 'r-', 'b:', 'g-.', 'k:', 'k--', 'm-', 'm:', 'm--'], legendsequence=[r'$L_{tot}$', r'$L_{X}$', r'$L_{out}$', r'$L_{tot}-L_{X}$', r'$(1-\beta_{\rm BS})L_{acc}$', r'$L_{acc}$', r'$L_{tot}-(1-\beta_{\rm BS})L_{acc}$', r'$L_{\rm shock,along}$', r'$L_{\rm shock,side}$'], inchsize=[5, 4], ylog=True, yrange=[1e-5*magn, magn*10], y_min_ticks_step=10, legend_charsize=10, legend_cols=3, title=conf, shift_title_left=1./20.)

            plots.someplots(t[ws], [s[ws], s[ws]*0.+xs], name=outdir+"/shockfront", xtitle=r'$t$, s', ytitle=r'$R_{\rm shock}/R_*$', xlog=False, formatsequence=['k-', 'r-', 'b:'], verticalformatsequence='b:', inchsize=[5, 4], title=conf)

            plots.someplots(lc_part[ws], [s[ws], s[ws]*0.+xs, s[ws]*0.+r_e/rstar, s[ws]*0.+r[-1]/rstar], name=outdir+"/fluxshock", xtitle=r'$L_{X}/L_{\rm Edd}$', ytitle=r'$R_{\rm shock}/R_*$', xlog=(lc_tot[ws].max()/median(lc_tot[ws])) > 10., ylog=True, legendsequence=['shock', 'BS', 'r_e', 'max r'], formatsequence=['k-', 'r-', 'b-', 'm:'], vertical=eqlum, verticalformatsequence='r-', vertical_full_length=True, inchsize=[5, 4], title=conf)

            plots.someplots(t[ws], [-v1[ws], -v2[ws], sqrt(2./s[ws]/rstar), sqrt(2./s[ws]/rstar)/7.], name=outdir+"/vleap", xtitle=r'$t$, s', ytitle=r'$v/c$', xlog=False, formatsequence=['k-', 'b:', 'r-', 'r-'], title=conf)

            plots.someplots(s[ws], [1./dt_current[ws], 1./(tscale_s*rstar**1.5*s[ws]**3.5)], xtitle=r'$R_{\rm shock}/R_*$', xlog=True, ylog=True, formatsequence=['ro', 'k-'], name=outdir+"/ux", ytitle=r'$f$, Hz', title=conf)
        elif ifplot:
            print("Not enough valid points for plotting.")



    if (DEBUG) :
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
    
    save_summary = open(outdir+'/summary.txt', 'a') 
    sys.stdout = save_summary # @galja redirect print to summary 
    
    print("s/RNS = "+str(xmean)+"+/-"+str(xrms)+"\n")
    fmean = ff[wlate].mean() ; frms = ff[wlate].std()
    print("flux = "+str(fmean)+"+/-"+str(frms)+"\n")
    print("total flux = "+str(lc_tot[wlaten].mean())+"+/-"+str(lc_tot[wlaten].std())+"\n below the shock: "+str(lc_part[wlaten].mean())+"+/-"+str(lc_part[wlaten].std())+"\n")
    print("lc_out = "+str(lc_out[wlaten].mean())+"\n")
        
    sys.stdout = original_stdout # @galja back to stdout    
    save_summary.close()
###############################



def last_entry_dat_file (prefix = "out/tireout") :
    import glob
    FilenamesList = sorted(glob.glob(prefix+'*.dat'))
    NumFiles= len(FilenamesList)
    n=NumFiles
    return FilenamesList[-1];

def boundary_evolution ( prefix = "out/tireout", dat = False, conf = None,  xest = 4.):
    '''
    draws evolution of P/Pmag at the NS surface
    '''
    
    if conf is None:
        conf = 'DEFAULT'
    rstar = config[conf].getfloat('rstar')
    m1 = config[conf].getfloat('m1')
    #tscale = config[conf].getfloat('tscale') * m1
    #new: tscale_s = units.tscale_s(config, conf)
    mu30 = config[conf].getfloat('mu30')
    mdot = config[conf].getfloat('mdot') * 4.*pi
    afac = config[conf].getfloat('afac')
    drrat = config[conf].getfloat('drrat')
    realxirad = config[conf].getfloat('xirad')
    b12 = 2.*mu30*(rstar*m1/6.8)**(-3) # dipolar magnetic field on the pole, 1e12Gs units

    #get information about existing dat files:
    import glob
    

    FilenamesList = sorted(glob.glob(prefix+'*.dat'))
    NumFiles= len(FilenamesList)
    n=NumFiles
    t = zeros(n, dtype=double)
    p_ratio=arange(n, dtype=double)
    
    outdir = os.path.dirname(prefix) 
    geometry = loadtxt(outdir+"/geo.dat", comments="#", delimiter=" ", unpack=False)
    
    
    th = geometry[:,1] ; r = geometry[:,0]
    cross_section = geometry[:,3]; #galja
    cthfun = interp1d(r/r[0], cos(th)) # we need a function allowing to calculate cos\theta (x)
    across0 = geometry[0,3]  ;   delta0 = geometry[0,5]
    acrosslast =  geometry[-1,3]
    #   acrosslast = pi*4.*afac*drrat * geometry[-1,0]**2 
    
    
    BSgamma = (across0/delta0**2)/mdot*rstar / (realxirad/1.5)
    umag = units.umag_calc_polar (b12, m1) * (1.+3.*(cos(th[0]))**2)/4.
    
    
    acrossfun = interp1d(r/rstar, cross_section, bounds_error=False)
        
    # umag is magnetic pressure
    BSeta = (8./21./sqrt(2.) * umag * 3. * (realxirad/1.5))**0.25*sqrt(delta0)/(rstar)**0.125
    xs, BSbeta = bs.xis(BSgamma, BSeta, x0=xest, ifbeta = True)
    
    print("Reading Pratio")
    for k in arange(n):
        if (dat):
            t[k], p_ratio[k] = read_p_ratio_from_dat (FilenamesList[k])   
        
    print("End reading Pratio")
    
    if(ifplot):
        plots.someplots(t, [p_ratio/3], name = outdir+"/Pcond_race", xtitle=r'$t$, s', ytitle=r'$P/P_{\rm mag} (R_*)$', title=conf)
        
        
        


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
            plots.someplots(t, [f, tailexpfun(t, par[0], par[1], par[2], par[3]), t*0.+par[3]], name = prefix+"_fit", xtitle=r'$t$, s', ytitle=r'$L$', xlog=False, ylog=False, formatsequence=['k.', 'r-', 'g:'], yrange=[f[f>0.].min(), f.max()], title=conf)
            if par[1]>0.:
                plots.someplots(t, [f-par[3], tailexpfun(t, par[0], par[1], par[2], 0.)], name = prefix+"_dfit", xtitle=r'$t$, s', ytitle=r'$\Delta L$', formatsequence=['k.', 'r-'], ylog = True, xlog = False, title=conf)
            else:
                plots.someplots(t, [par[3]-f, -tailexpfun(t, par[0], par[1], par[2], 0.)], name = prefix+"_dfit", xtitle=r'$t$, s', ytitle=r'$\Delta L$', formatsequence=['k.', 'r-'], ylog = True, xlog = False, title=conf)
        else:
            plots.someplots(t, [f, tailfitfun(t, par[0], par[1], par[2], par[3]), t*0.+par[3]], name = prefix+"_fit", xtitle=r'$t$, s', ytitle=r'$L$', xlog=False, ylog=False, formatsequence=['k.', 'r-', 'g:'], yrange=[f[f>0.].min(), f.max()])
            plots.someplots(t, [f-par[3], tailfitfun(t, par[0], par[1], par[2], 0.)], name = prefix+"_dfit", xtitle=r'$t$, s', ytitle=r'$\Delta L$', formatsequence=['k.', 'r-'], ylog = True, xlog = False, title=conf)
    
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
    print("\nCalculate the optical depths along and across the flow...")
    rstar = config[conf].getfloat('rstar')
    m1 = config[conf].getfloat('m1')
    mu30 = config[conf].getfloat('mu30')
    mow = config[conf].getfloat('mow')
    vscale = 2.99792458e10
    rscale = config[conf].getfloat('rscale')*m1
    uscale = config[conf].getfloat('uscale')/m1 # uscale  1.73886e16 # c**4/GMsun kappa dimension of energy density
    b12 = 2.*mu30*(rstar*m1/6.8)**(-3) # dipolar magnetic field on the pole, 1e12Gs units
    mdot = config[conf].getfloat('mdot') * 4.*pi
    afac = config[conf].getfloat('afac')
    realxirad = config[conf].getfloat('xirad')

    geofile = os.path.dirname(prefix)+"/geo.dat"
    
    r, theta, alpha, across, l, delta = geo.gread(geofile) 
    level1= across*0+1
    # r is normalized to rg
    
    
    if(ifhdf):
        hname = prefix + ".hdf5"
        print(hname,  " n=", n)
        # if entry does not exists, tak the next:
        while (hdf.check_entry_exists(hname, n) == 0) :
                n=n+1
        # print("! This entry exists:", hname,  " n=",n)
        entryname, t, l, r, sth, rho, u, v, qloss, glo, ediff = hdf.read(hname, n)
        #entryname, t, l, r, sth, rho, u, v, qloss, glo 
        
        # in hdf5, u - is the radiative energy!
        # r is normalized to Rstar
        
        urad_phys = u *  uscale
        #we need to make  Urad/Umag  from u :
        umagtar =  units.umag_calc_polar_phys(b12) * (1.+3.*cos(theta)**2)/4. * r**(-6.)
        urad_umag = urad_phys/umagtar
        
       
    else:
        #lines = loadtxt( os.path.dirname(prefix) + '/avprofile.dat', comments = '#')
        #r = lines[:,0] ; v = lines[:,1] ; u = lines[:,2] ; prat = lines[:,3] ; temp = lines[:,4] ; rho = lines[:,5]
        print (prefix + ".dat" )
        lines = loadtxt( prefix + ".dat" , comments = '#')
        r = lines[:,0] ; rho = lines[:,1] ; v = lines[:,2] ; u = lines[:,3] ; qloss = lines[:,4] ; 
        
        urad_umag = u
        
        # r is normalized to Rstar
        # magnetic field energy density erg/cm3]:
        umagtar =  units.umag_calc_polar_phys(b12) * (1.+3.*cos(theta)**2)/4. * r**(-6.)
        # energy density [erg/cm3]:
        urad_phys = u  * umagtar
          
        #ediff = lines[:,5]
        # entryname = hdf.entryname(n, ndig=5)
        # fname = prefix + entryname + ".dat"
        # lines = loadtxt(fname, comments="#")
        # rho = lines[:,1]
        
        
   


    radial_dir = os.path.join(os.path.dirname(prefix), "radial_times")
    os.makedirs(radial_dir, exist_ok=True)



    u_norm = urad_phys / uscale
    
    taucross = delta * rho  # size across the flow
    
    dr = (r[1:]-r[:-1])  ;   rc = (r[1:]+r[:-1])/2.
    taualong = (rho[1:]+rho[:-1])/2. * dr
    taucrossfun = interp1d(r, taucross, kind = 'linear')
    
    pratio = urad_umag  # urad/Pmag (max 3), Pmag = B^2/8/Pi; Prad/Pmag = Urad/Umag/3  = "u/3"
    #urad  = u *Pmag = u*Umag
    
    # magnetic field energy density:
    #-?TO REMOVE umagtar = umag * (1.+3.*g.cth**2)/4. * (rstar/g.r)**6
    #-?TO REMOVE umag = umag_calc_polar (b12, m1) * (1.+3.*(cos(th[0]))**2)/4.
    #-?TO REMOVE umagtar = umag * (1.+3.*(1.-sth**2))/4. / (r)**6 # r is already in rstar units
    
   
    # added galja MAY 2023
    mass1 = config[conf].getfloat('m1')
    rhoscale = config[conf].getfloat('rhoscale') / mass1 
    rho_phys = rho*rhoscale
    
    # profile of effective BS's beta:
    #betacoeff = config[conf].getfloat('betacoeff') * (m1)**(-0.25)/mow
    #betafun = betafun_define() # defines the interpolated function for beta
    #beta = betafun(Fbeta(rho, u, betacoeff))
    #press = u_norm/3./(1.-beta/2.) # total gas +rad pressure; u=ugas+urad
    
    betaeff = (4/3*u_norm/rho) * r*rstar
    #print ("R",r[0:1] * rstar)
    #print ("u",u_norm[0:1], umagtar, b12)
    #print ("rho",rho[0:1])
    # print ("beta=", betaeff)
    # print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Step n=", n );
    

    if (ifplot):
        plots.someplots(r, [rho_phys,r**(-9),r**(-2)], name = os.path.join(radial_dir, "rho" + str(n)),  xtitle=r'$r/R_{\rm NS}$', ytitle=r'$\rho$', xlog=True, ylog=True, formatsequence=['k-', 'r-','r:'], legendsequence = [ r'$\rho$',r'$r^{-9}$',r'$r^{-2}$'], title=conf)
        
        plots.someplots(r, [abs(v), rho, rho*across*abs(v)*1e-3, across, urad_umag/3], name = os.path.join(radial_dir, "pars" + str(n)), xtitle=r'$r/R_{\rm NS}$', ytitle='', xlog=True, ylog=True, formatsequence=['r-','g-','b:','r:','m--'], legendsequence = [ r'$v$', r'$\rho$',r'$\rho v A_\perp/10^3$', r'$A_\perp$',r'$P/P_{\rm mag}$'], yrange = [1e-6,1e6], legend_charsize = 10, legend_pos = [0.5,1.2], legend_cols=3, title=conf)
        
        #Re_GM = r.max()*1e6/
        plots.someplots(r, [ u_norm/rho, 4./3. * u_norm/rho * r*rstar], name = os.path.join(radial_dir, "fbeta" + str(n)),  xtitle=r'$r/R_{\rm NS}$', ytitle=r'$u/\rho$', xlog=True, ylog=True, formatsequence=['g-','m-'], legendsequence = [ r'$u/\rho$', r'$\beta(R)$'],  legend_charsize = 10, legend_pos = [0.5,1.2], legend_cols=3, title=conf)
        
        
        plots.someplots(r, [rho*across*abs(v)*1e-3, across], name = os.path.join(radial_dir, "mass_flux" + str(n)),  xtitle=r'$r/R_{\rm NS}$', ytitle='', xlog=True, ylog=True, formatsequence=['b-','r:'], legendsequence = [r'$\rho v A_\perp/10^3$', r'$A_\perp$'], yrange = [1e-1,1e4], title=conf)
        
        plots.someplots(r, [betaeff], name = os.path.join(radial_dir, "beta" + str(n)), xtitle=r'$r/R_{\rm NS}$', ytitle='', xlog=True, ylog=True, formatsequence=['b-'], legendsequence = [r'$\beta_{\rm eff}$'], yrange = [0.1,1], title=conf)
        
        
        plots.someplots(r, [abs(v),1./sqrt(r), abs(v)/sqrt(r)/7., 2./3.*sqrt(urad_phys/rho_phys)/vscale], name = os.path.join(radial_dir, "v" + str(n)), xtitle=r'$r/R_{\rm NS}$', ytitle=r'$|v|/c$', xlog=False,  x_min_ticks_step=10,ylog=True, formatsequence=['r-','k:','r:', 'b-'], legendsequence = [ r'$v$',r'$v_{\rm free fall}$',r'$v/7$',r'$v_{\rm s}$'],yrange = [1e-7,1.2], legend_pos = (0.5, 1.2), title=conf )
    #END of  added galja MAY 2023
    
    
    if (ifplot):    
        plots.someplots(rc-1., [taucrossfun(rc), taualong], name = os.path.join(radial_dir, "tau" + str(n)),  xtitle=r'$r/R_{\rm NS}-1$', ytitle=r'$\tau$', xlog=True, ylog=True, formatsequence=['k-', 'r-'],legendsequence = [r'$\tau$',r'${\rm d} \tau_l$'], title=conf)
        
        plots.someplots(rc, [taucrossfun(rc)], name = os.path.join(radial_dir, "tau_p" + str(n)),  xtitle=r'$r/R_{\rm NS}$', ytitle=r'$\tau_\perp$', xlog=False, ylog=True, formatsequence=['k-', 'r-'], title=conf)
        
        plots.someplots(rc-1., [2./(rho[1:]+rho[:-1])/rc/rstar, (delta[1:]+delta[:-1])/2./rc/rstar, dr/rc], name = os.path.join(radial_dir, "dR" + str(n)), xtitle=r'$r/R_{\rm NS}-1$', ytitle=r'$\delta/R$, $1/\kappa \rho R$', xlog=True, ylog=True, formatsequence=['k-', 'r--', 'b:'], legendsequence = [r'$1/\kappa \rho R$',r'$\delta/R$',r'$dR/R$'], title=conf)
        
        #plots.someplots(rc-1., [(pratio[1:]+pratio[:-1])/2 ], name = prefix+"_p_ratio"+str(n), xtitle=r'$r/R_{\rm NS}-1$', ytitle=r'$P/P_{\rm mag}$', xlog=True, ylog=True, formatsequence=['k-'])
    else:
        # ASCII output
        fout = open(os.path.dirname(prefix)+'/tauprofile.dat', 'w')
        fout.write("# R  -- tperp -- tparallel \n")
        nx = size(rc)
        for k in arange(nx):
            s = str(rc[k]) + " " + str(taucrossfun(rc[k])) + " " + str(taualong[k]) + "\n"
            #print(s)
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

    #tscale = config[conf].getfloat('tscale') * config[conf].getfloat('m1')
    tscale_s = units.tscale_s(config, conf)

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
    tar *= tscale_s
    if(ifplot):
        # overplotting with the total flux
        plots.someplots(tar, [lint, ltot, ltot-lint, lint*0.+mdot*0.2], xlog=False, formatsequence = ['k-', 'g--', 'b--', 'r-'], xtitle='t, s', ytitle=r'$L/L_{\rm Edd}$', name= os.path.dirname(hfile)+'/cutflux', title=conf)

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
        umag = units.umag_calc_polar (b12, m1)
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
        #print(ltot[k])

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
        ekin[k] = numerical_trapezoid(rho * v**2/2. * across, x = l)
        epot[k] = -numerical_trapezoid(rho / r * across, x = l)
        eheat[k] = numerical_trapezoid(u * across, x = l)
        etot[k] = ekin[k] + epot[k] + eheat[k]
        tar[k] = t
        ltot[k] = numerical_trapezoid(qloss, x=l)
        mass[k] = numerical_trapezoid(rho * across, x = l)

    llost = cumnumerical_trapezoid(ltot, x=tar, initial = 0.)
        
    m1 = config[conf].getfloat('m1')
    #tscale = config[conf].getfloat('tscale') * m1
    tscale_s = units.tscale_s(config, conf)
    tar *= tscale_s
    plots.someplots(tar, [etot, ekin, epot, eheat, etot+llost], 
                    name = os.path.dirname(infile)+'/energytest',
                    formatsequence = ['k-', 'b--', 'r:', 'g-.', 'm--'],
                    xtitle = r'$t$, s', ytitle = r'$E$', title=conf)
    plots.someplots(mass, [etot, ekin, epot, eheat, llost], 
                    name = os.path.dirname(infile)+'/energytest_m',
                    formatsequence = ['k-', 'b--', 'r:', 'g-.', 'm--'],
                    xtitle = r'$M$', ytitle = r'$E$', title=conf)

    dedt = (etot[1:]-etot[:-1])/(tar[1:]-tar[:-1]) * tscale_s
    dedt_p = (epot[1:]-epot[:-1])/(tar[1:]-tar[:-1]) * tscale_s

    plots.someplots(tar[1:], [-dedt, ltot[1:], dedt_p, dedt_p+ltot[1:]-dedt], formatsequence = ['k-', 'b--', 'k:', 'r:'], 
                    name = os.path.dirname(infile)+'/energytest_d', xlog = False, 
                    xtitle = r'$t$, s', ytitle = r'$dE/dt$', yrange = [-ltot.max(), ltot.max()*2.], title=conf)

def masstest(indir, conf='DEFAULT'):

    print ("\n\n In masstest():")
    mu30 = config[conf].getfloat('mu30')
    mass1 = config[conf].getfloat('m1')
    xifac = config[conf].getfloat('xifac')
    mdot = config[conf].getfloat('mdot') * 4.*pi  # in units of Ledd/c

    massscale = config[conf].getfloat('massscale') * mass1**2

    massfile = indir + '/totals.dat'
    masslines = loadtxt(massfile, comments="#", delimiter=" ", unpack=False)
    
    t=masslines[:,0] ; m=masslines[:,1] ; mlost=masslines[:,3] ; macc=masslines[:,4] ; mdotcurrent = masslines[:,5]

    plots.someplots(t, [m-m[0], macc-mlost-(macc[0]-mlost[0]), (macc-macc[0]), (mlost-mlost[0])], name = indir+'/mbalance', formatsequence = ['k-', 'r:', 'b--', 'm-', 'c-'],
    xlog = False, ylog = False, xtitle = r'$t$, s', ytitle = r'$M$ code units', legendsequence=[r'$M_{\rm col}$', r'$M_{\rm acc}-M_{\rm lost}$', r'$M_{\rm acc}$', r'$M_{\rm lost}$'], title=conf,charsize=14)
    print("-->mass change / mass injected normalized = "+str((m[-1]-m[0])/(macc[-1]-macc[0])))

    m *= massscale
    mlost *= massscale
    macc *= massscale


    Rstar = mass1 * config[conf].getfloat('rstar') * config[conf].getfloat('rscale') #cm

    r_e = config[conf].getfloat('r_e_coeff') * (mu30**2/mdot)**(2./7.)*mass1**(-10./7.) * xifac # magnetosphere radius
    Rext = r_e * config[conf].getfloat('rscale') * mass1 # magn radius in cm


    max_column_mass = pi * 32/3 * config[conf].getfloat('afac') * config[conf].getfloat('drrat') * (mu30*1e30)**2 / Consts['G']/Consts['Msun']/mass1/ Rstar/Rext


    plots.someplots(t, [m-m[0], macc-mlost-(macc[0]-mlost[0]), (macc-macc[0]), (mlost-mlost[0])], name = indir+'/mbalance_g', formatsequence = ['k-', 'r:', 'b--', 'm-', 'c-'],
    xlog = False, ylog = False, xtitle = r'$t$, s', ytitle = r'$M$ (g)', legendsequence=[r'$M_{\rm col}$', r'$M_{\rm acc}-M_{\rm lost}$', r'$M_{\rm acc}$', r'$M_{\rm lost}$'], title=conf,charsize=14)
    print("-->mass change / mass injected (g) = "+str((m[-1]-m[0])/(macc[-1]-macc[0])))
    
    plots.someplots_addrightY(t, [m-m[0], macc-mlost-(macc[0]-mlost[0]), (macc-macc[0])*1e-1, (mlost-mlost[0])*1e-1], name = indir+'/mbalance_norm', koef = 1/max_column_mass, formatsequence = ['k-', 'r:', 'b--', 'm-', 'c-'],  xlog = False, ylog = False, xtitle = r'$t$, s', ytitle = r'$M$ (g)', ytitle_add=r'$M_{\rm col,max}$', legendsequence=[r'$M_{\rm col}$',  r'$M_{\rm acc}-M_{\rm lost}$', r'$M_{\rm acc}/10$', r'$M_{\rm lost}/10$'], title=conf)
    print("mass change / mass injected = "+str((m[-1]-m[0])/(macc[-1]-macc[0])))
    
    plots.someplots((t[1:]+t[:-1])/2., [(mlost[1:]-mlost[:-1])/(t[1:]-t[:-1])], name=indir+'/mdot', xtitle = r'$t$, s', ytitle = r'$\dot{M}$, g s$^{-1}$', xlog = False, 
    #xrange=[1.5,1.6], 
    ylog = True, title=conf)

def plot_bunch  (r,t,q,v,m,u,lurel,p,rho,beta,across,mdot=None,outdir='',trange=None,nv=30,inchsize = [4, 12],tag=''):
    
    print ("plotting...")
    
    # Q *4*pi*r^2/Mdot
    plots.somemap(r, t, 4.*pi *q*r**2/mdot, name=outdir+'/q2d_q'+tag, inchsize = inchsize, cbtitle = r'$R^2 Q^- / L_{\rm Edd}$', xrange = trange, transpose=True, title = conf) #, levels = 3.*arange(nv)/double(nv-2)-1.)
    
    # lg Q
    plots.somemap(r, t, log10(q), name=outdir+'/q2d_LGq'+tag, inchsize = inchsize, cbtitle = r'$\log_{10}Q$', transpose = True, xrange = trange, title = conf)
    
  
    plots.somemap(r, t, v, name=outdir+'/q2d_v'+tag, inchsize = inchsize, cbtitle = r'$v/c$',  xrange = trange,transpose=True, title = conf)
    
    # Pmag/P:
    plots.somemap(r, t, lurel, name=outdir+'/q2d_u'+tag, inchsize = inchsize, cbtitle = r'$\log_{10}\left(u/u_{\rm mag}\right)$', xrange = trange, addcontour = [u/3./1., u/3./0.9, u/3./0.8],transpose=True, title = conf)
    
    
    # mdot (r,t):
    plots.somemap(r, t, m, name=outdir+'/q2d_m'+tag, inchsize = inchsize, cbtitle = r'$s / \dot{M}$', xrange = trange, transpose=True, levels = 3.*arange(nv)/double(nv-2)-1., title = conf)
    
    
    
    #plot beta:
    #plots.somemap(r, t, log10(beta), name=outdir+'/q2d_b',inchsize = inchsize, cbtitle = r'$\log_{10}\beta$', transpose = True, xrange = trange)

    # plot mdot(r)
    #mdmean = mean(m)
    #mdstd = std(m)
    #plots.someplots(r, [r*0.+1., r*0., r*0.+mdmean, (mdmean+mdstd), (mdmean-mdstd)], formatsequence = [':k', '--k', '-k', '-g', '-g'], xlog = True, ylog = False, xtitle = r'$R/R_{\rm *}$', ytitle = r'$\langle s\rangle /\dot{M}$', inchsize = [3.35, 2.], name=outdir+'/q2d_mdmean')
    

def gquasiplot (infile, conf = 'DEFAULT', trange = None):
    with open(infile) as f:
        lines = array([line.strip().split() for line in f],float)
    #print(lines)    
    t = lines[1:,0] ;  r = lines[1:,1] ;  v = lines[1:,2] ;  lurel = lines[1:,3];  m = lines[1:,4]   
    #print(t)
     
def gquasiplot2 (infile, conf = 'DEFAULT', trange = None):
    import csv
        
    data = []
    
    # Auto-detect the CSV dialect that is being used
    #
    with open(infile, 'rb') as fp: 
        #test_bytes = fp.read(1024) # Grab a sample of the CSV for format detection.
        #fp.seek(0)  # Rewind
        has_header = 1 
        #csv.Sniffer().has_header(test_bytes.decode('utf-8'))  # Check to see if there's a header in the file.
        #dialect = csv.Sniffer().sniff(test_bytes)
        inputreader = csv.reader(fp, delimiter=" ")
        if has_header:
            next(inputreader) # Skip the header if we have one.
        for row in inputreader:
        # Filter out empty fields
            row = [x for x in row if x != ""]
            data.append(row)
    lines = asarray(data, dtype = np.float)
    t = lines[:,0] ;  r = lines[:,1] ;  v = lines[:,2] ;  lurel = lines[:,3];  m = lines[:,4]   
    #print(t)
    
def gquasiplot3 (infile, conf = 'DEFAULT', trange = None):
        import pandas 
       
        #lines = pandas.read_csv (infile, header=None, comment="#", delimiter=" ",nrows=10) 
        
        #lines = pandas.read_csv (infile, header=None, comment="#", delimiter=" ",skiprows= lambda x: not x%100)
        lines = pandas.read_csv (infile, header=None, comment="#", delimiter=" ", skiprows=[i for i in range(1,45000)])
        
        t = lines.iloc[:,0] ;  r = lines.iloc[:,1] ;  v = lines.iloc[:,2] ;  lurel = lines.iloc[:,3];  m = lines.iloc[:,4]   
        #print(r)
    
def test_hdf5 (conf='DEFAULT'):
    import h5py as h5
    nentry=100
    inhdf = 'out_fidu/test.hdf5'
    alldata  = h5.File(inhdf, "r")
    #print(alldata)
    
    #dset = alldata.create_dataset("test", (2, 2))
    print ( print("Keys: %s" % alldata.keys()))
    keys1 = list(alldata.keys())[0]
    print(alldata[keys1])
    print(alldata[keys1]['rho'][::40])
    
    print ("test_hdf5 ends")
    exit()
    print(list(alldata[keys1]))    
    
    sizenow=shape(alldata[keys1]['rho'])[0]
    print(sizenow)
    needsize=20
    
    group = alldata[keys1]
    #print(group)
    for key_ in list(alldata[keys1]):
        print(key_)
        data = group[key_][()]
        data = alldata[keys1][key_][::round(sizenow/needsize)]
        print(data)
        #data = alldata[keys1][key_][::round(sizenow/needsize)]
        #print(data)
        #exit()
        #print(alldata[keys1][key_])
        #newcol = alldata[keys1][key_]
        #newcol = alldata[keys1][key_][::round(sizenow/needsize)]
        print(alldata[keys1][key_][::round(sizenow/needsize)])
       
        #print(alldata[keys1][key_][::round(sizenow/needsize)]) 
    #After you are done
    alldata.close()
    exit()
    
  #  print(type(alldata[key])) 
    
 #   allkeys = list(alldata[key])
 #   print(allkeys) 
  #  exit()
    #for key in list(alldata[key])
   # print(alldata[key]['rho'])
    
   
    
    
    #newdata = alldata[key]['rho'][::round(sizenow/needsize)]
   # print(newdata)
    #group = f[key]

#Checkout what keys are inside that group.
    
   # exit()
# This assumes group[some_key_inside_the_group] is a dataset, 
# and returns a np.array:
  #  data = group[some_key_inside_the_group][()]
#Do whatever you want with data
    #exit()


    #dset[0,2:10,1:9:3]
#dset[:,::2,5]
#dset[0]
#dset[1,5]
#dset[0,...]
#dset[...,6]
    exit()
    entry, t, l, xp, sth, rhop, up, vp, qloss, glo, ediff  = hdf.read(inhdf, nentry)
     
def quasi2d_nocalc(infile, conf = 'DEFAULT', trange = None, tag = ''):
    outdir = os.path.dirname(infile)
    
    # get cross-section:
    geometry = loadtxt(outdir+"/geo.dat", comments="#", delimiter=" ", unpack=False)
    across = geometry[:,3] 
    
    mdot = config[conf].getfloat('mdot')  #  no 4 pi  here! probably that's right'
    
    lines = loadtxt(infile, comments="#", delimiter=" ", unpack=False)
    
    t = lines[:,0] ;  r = lines[:,1] ;  v = lines[:,2] ;  lurel = lines[:,3];  m = lines[:,4]; 
    q = lines[:,5]  ; qd = lines[:,6] ; p = lines [:,7]; rho = lines [:,8]; beta = lines [:,9];
    
    nt = size(unique(t)) ; nr = size(unique(r))
    
    t = unique(t) ; r = unique(r)
    v = reshape(v, [nt, nr]) ;  lurel = reshape(lurel, [nt, nr]) ;  m = reshape(m, [nt, nr])
    q = reshape(q, [nt, nr])
    u = 10.**lurel
    
    
    if ifplot:
        plot_bunch  (r,t,q,v,m,u,lurel,p,rho,beta,across,mdot,outdir,trange,nv=30,tag=tag)
        
       



#############################################
def quasi2d(hname, n1, n2, conf = 'DEFAULT', step = 1, kleap = 5, trange = None, tag='', xi_range=None):
    '''
    makes quasi-2D Rt plots or an RT table
    n1 = starting entry number
    n2 = number of entries/dumps
    n2 - size =  requested number of entries starting from n1
    '''
    
    
    print ("\nStart quasi2d():\n make quasi-2D plots for "+hname+" using ",n2,"dumps starting from ",n1)
    outdir = os.path.dirname(hname)
    if tag == '_early':
        figuredir = os.path.join(outdir, 'early')
    elif tag == '_late':
        figuredir = os.path.join(outdir, 'late')
    else:
        figuredir = outdir

    os.makedirs(figuredir, exist_ok=True)
    betafun = betafun_define() # defines the interpolated function for beta
    
    #compare desired numbers of entries with existing:
    num_entries = hdf.num_bunches_total(hname)
    if (n1<0) : 
        print ("in quasi2d set n1",n1," to zero")
        n1 = 0
    if n1 >= num_entries:
        raise ValueError("Starting entry n1 is outside the available entries")

    max_n2 = int((num_entries - n1 + step - 1) // step)

    if n2 > max_n2:
        print("in quasi2d: reducing n2 to", max_n2)
        n2 = max_n2
    
    
    
    # desired number of time steps
    nt=n2
    
    # length of resulting vectors over time:
    time_ar_size = nt
    
    print ("length over time in plot=", time_ar_size)
    # geometry:
    geofile = outdir+"/geo.dat"
    print ("Reading file "+geofile)
    r, theta, alpha, across, l, delta = geo.gread(geofile)
    perimeter = 2. * (across/delta + 2.*delta)

    #print(theta)

    # loop until find first frame and get globals:
    for entry_num in range(n1,n1+n2) :
        entryname, t, l, r, sth, rho, u, v, qloss, glo, ediff = hdf.read(hname, entry_num)
        if (entryname != -1):
            break
    
    cth = sqrt(1.-sth**2)

    rstar = glo['rstar']
    
    umag = glo['umag']
    
    m1 = config[conf].getfloat('m1')
    #tscale = config[conf].getfloat('tscale') * m1
    tscale_s = units.tscale_s(config, conf)
    rscale = config[conf].getfloat('rscale') * m1
    massscale = config[conf].getfloat('massscale') * m1**2
    rstar = config[conf].getfloat('rstar')
    mu30 = config[conf].getfloat('mu30')
    mdot = config[conf].getfloat('mdot') * 4.*pi
    
    afac = config[conf].getfloat('afac')
    realxirad = config[conf].getfloat('xirad')
    mow = config[conf].getfloat('mow')
    b12 = 2.*mu30*(rstar*m1/6.8)**(-3) # dipolar magnetic field on the pole, 1e12Gs units
    
    
    umag1 = units.umag_calc_polar (b12, m1)
    print("mdot = "+str(mdot))
    # ii = input("M")
    print("Check Umag polar = "+str(umag)+" = "+str(umag1)+"\n")
    betacoeff = config[conf].getfloat('betacoeff') * (m1)**(-0.25)/mow
    
    mcol = across[0] * rstar**2 * umag / m1 * (1.+3.*cth[0]**2)/4.
    #tr = mcol / mdot * tscale_s
    #print("Approximate replenishment time tr (s) = "+str(tr))
    print ("tscale_s = ",tscale_s)
    
    # ii =input('tr')
    
    nr=size(r)
    nrnew = 300 # radial mesh interpolated to nrnew
    rnew = (r.max()/r.min())**(arange(nrnew)/double(nrnew-1))*r.min()
    sthfun = interp1d(r, sth)
    sthnew = sthfun(rnew)

    lfun = interp1d(r, l)
    lnew = lfun(rnew)

    perimeterfun = interp1d(r, perimeter)
    perimeternew = perimeterfun(rnew)

    umagtar = umag * (1.+3.*(1.-sth**2))/4. / (r)**6 # r is already in rstar units
    umagtarnew = umag * (1.+3.*(1.-sthnew**2))/4. / (rnew)**6 # r is already in rstar units
    var = zeros([time_ar_size, nrnew], dtype=double)
    uar = zeros([time_ar_size, nrnew], dtype=double)
    par = zeros([time_ar_size, nrnew], dtype=double)
    rhoar = zeros([time_ar_size, nrnew], dtype=double)
    betar = zeros([time_ar_size, nrnew], dtype = double) # this is pgas/ptot
    qar = zeros([time_ar_size, nrnew], dtype=double)
    ear = zeros([time_ar_size, nrnew], dtype=double)
    Teff_diff = zeros([time_ar_size, nrnew], dtype=double)
    lurel = zeros([time_ar_size, nrnew], dtype=double)
    mdar = zeros([time_ar_size, nrnew], dtype=double)
    tar = zeros(time_ar_size, dtype=double)
    numentryar = zeros(time_ar_size, dtype=int)
    rvent = zeros(time_ar_size, dtype=double)
    mdot_out = zeros(time_ar_size, dtype=double)
    maxprat = zeros(time_ar_size, dtype=double)
    drvent = zeros(time_ar_size, dtype=double)
    rshock = zeros(time_ar_size, dtype=double)
    betavent = zeros(time_ar_size, dtype = double) # this one is BS's beta
    betaeff = zeros(time_ar_size, dtype = double) # and this one, too
    betaeff_m = zeros(time_ar_size, dtype = double) # this, too
    #    var[0,:] = v[:] ; uar[0,:] = u[:] ; tar[0] = t
    
    k = 0 # no of element in vectors  +step : just in case
    entry_ar = hdf.stored_entries_nums (hname)
    

    # Select entries by their actual entry number
    entry_indices = where(entry_ar >= n1)[0]

    if size(entry_indices) == 0:
        raise ValueError("No HDF5 entries found at or after n1={}".format(n1))

    entry_indices = entry_indices[::step]
    candidate_entries = entry_ar[entry_indices]

    # Select by physical time, if requested
    if trange is not None:
        selected_entries = []

        for entry in candidate_entries:
            entryname_test, t_test, l_test, r_test, sth_test, rho_test, u_test, v_test, qloss_test, glo_test, ediff_test = hdf.read(hname, entry)

            if entryname_test == -1:
                continue

            t_test_physical = t_test * tscale_s

            if trange[0] <= t_test_physical <= trange[1]:
                selected_entries.append(entry)

        candidate_entries = asarray(selected_entries, dtype=int)

    # Limit the number of selected entries to n2
    candidate_entries = candidate_entries[:int(n2)]

    time_ar_size = len(candidate_entries)
    nt = time_ar_size

    if time_ar_size == 0:
        raise ValueError("No entries selected for trange={}".format(trange))

    print("Selected", time_ar_size, "entries")
    print("First selected entry =", candidate_entries[0])
    print("Last selected entry =", candidate_entries[-1])
#
#     for kk1 in range(n1,n1+n2) :
#     #for kcount in arange(nt):
#         entry = entry_ar[kk1-n1] # to begin with 0th element
        ## was entry = kcount*step+n1
        #entry = entry_ar[kk1]
    for entry in candidate_entries:
        entryname, t, l, r, sth, rho, u, v, qloss, glo, ediff = hdf.read(hname, entry) # hdf5 read
        #print ('k= ',k, "kcount=", kcount, "nt=",nt)   
        if (entryname==-1):
            #print ('kcount=',kcount)
            continue
        
         
        
        vfun = interp1d(r, v, kind = 'linear')
        var[k, :] = vfun(rnew)
        qfun = interp1d(r, qloss/perimeter, kind = 'linear')
        qar[k, :] = qfun(rnew)
        # print("size(ediff) = ",size(ediff))
        efun = interp1d(r, ediff, kind = 'linear')
        ear[k, :] = efun(rnew)
        
        # temperature equivalent to  radiative flux along the tube
        Teff_diff_fun =  interp1d(r, 4.74501 * m1**(-0.25) * (abs(ediff)/across)**0.25, kind = 'linear' ) # keV
        Teff_diff[k,:] = Teff_diff_fun (rnew)
        
        ufun = interp1d(r, u, kind = 'linear')
        uar[k, :] = ufun(rnew)
        rhofun = interp1d(r,rho, kind = 'linear')
        rhoar[k, :] = rhofun(rnew)
        beta = betafun(Fbeta(rho, u, betacoeff))
        press = u/3./(1.-beta/2.) # total gas +rad pressure; u=ugas+urad
        pfun = interp1d(r, press, kind = 'linear')
        par[k, :] = pfun(rnew) # total gas +rad pressure;
        betar[k, :] = 2.*(1.-uar[k,:]/par[k,:]/3.)
        mfun = interp1d(r, -rho * v * across, kind = 'linear')
        wloss = (press>(0.8*umagtar))
        shock = ((v)[kleap:]-(v)[:-kleap]).argmin()
        rshock[k] = r[shock+kleap] #+r[minimum(shock+kleap+1, nt-1)])/2.
        
        if wloss.sum()> 3:
            # imfun = interp1d((-rho * v * across)[wloss], r[wloss], kind = 'linear', bounds_error=False, fill_value=NaN)
            # rvent[k] = imfun(mdot/2.)
            wvent = (press/umagtar)[(r>r.min())&(r<rshock[k])].argmax()
            if wvent >= (nr-1):
                #print(nr)
                wvent -= 1
            rvent[k] = r[wvent]
            drvent[k] = r[wvent+1]-r[wvent]
            maxprat[k] = (press/umagtar)[(r>r.min())&(r<rshock[k])].max()
            betavent[k] = ((u+press)/rho)[wvent] * rstar
            if (rvent[k] >= 1.) & (maxprat[k] >= 1.) & (verbatim_vent):
                print("rvent = "+str(rvent[k])+" = "+str(r[wvent]))
                print("drvent = "+str(drvent[k]))
                print("t (s) = "+str(t*tscale_s))
                # ii = input('R')
            #            print("maxprat = "+str(maxprat[k]))
        # effective BS's beta:
        betaeff[k] = ((u+press)/rho)[0:1].mean() * rstar
        
        # rstar is in rg units 
        
        betaeff_m[k] = (umagtar/rho)[0:1].mean() * rstar * 4.
        #  print(rstar)
        mdar[k, :] = mfun(rnew)
        mdot_out[k] = mdar[k, -1]
        # print("mdot[-1] = "+str(mdar[k,-1]))
        tar[k] = t

        # added galja MAY 2023
        numentryar[k] = entry
        # END of added galja MAY 2023
        k = k+1
    nv=30
    vmin = round(var.min(),2)
    vmax = round(var.max(),2)
    vmin = maximum(vmin, -1.) ; vmax = minimum(vmax, 1.)
    vlev=linspace(vmin, vmax, nv, endpoint=True)
    print("vmin=",var.min())
    print("vmax=",var.max())
    varmean = var.mean(axis=0)
    varstd = var.std(axis=0)

    # velocity
    if ifplot:
        plots.somemap(rnew, tar*tscale_s, var, name=os.path.join(figuredir, 'q2d_v'+tag), levels=vlev, inchsize=[4, 12], cbtitle=r'$v/c$', transpose=True, xrange=trange, xi_range=xi_range, title=conf+": Radial velocity normalized by speed of light")

        plots.someplots(rnew, [-sqrt(1./rstar/rnew), rnew*0., varmean, varmean+varstd, varmean-varstd], formatsequence=[':k', '--k', '-k', '-g', '-g'], xlog=True, ylog=False, xtitle=r'$R/R_{\rm *}$', ytitle=r'$\langle v\rangle /c$', inchsize=[3.35, 2.], name=os.path.join(figuredir, 'q2d_vmean'+tag), xi_range=xi_range, title=conf)
      
        
        umag = units.umag_calc_polar (b12, m1)
        BSgamma = (across/delta**2)[0]/mdot*rstar / (realxirad/1.5)
        # umag is magnetic pressure
        BSeta = (8./21./sqrt(2.)*umag*3. * (realxirad/1.5))**0.25*sqrt(delta[0])/(rstar)**0.125
        xs, BSbeta = bs.xis(BSgamma, BSeta,  ifbeta = True)
        
        plots.someplots(tar*tscale_s, [rvent, rshock, rvent*0+r[-1], rvent*0+xs], name=os.path.join(figuredir, 'rvent'+tag), xtitle=r'$t$, s', ytitle=r'$R_{\rm vent}/R_*$', xi_range=xi_range, formatsequence=['-k', '--k', '-b', '-g'], legendsequence=['vent', 'shock', 'max r', 'BS shock'], title=conf)
        
        plots.someplots(tar*tscale_s, [numentryar], name=os.path.join(figuredir, 'entries'+tag), xtitle=r'$t$, s', ytitle=r'Entry No', xlog=False, title=conf)

    # internal energy
    #    print(umag)
    for k in arange(nrnew):
        lurel[:,k] = log10(uar[:,k]/umagtarnew[k])
    if ifplot:
        umin = round(lurel[uar>0.].min(),2)
        umax = round(lurel[uar>0.].max(),2)
        lulev = linspace(umin, umax, nv, endpoint=True)
        if DEBUG : print(lulev)
        
        
        
        # plots.somemap(rnew, tar*tscale_s, lurel, name=os.path.join(figuredir, 'q2d_p'+tag), levels=lulev, inchsize=[4, 12], cbtitle=r'$\log_{10}P/P_{\rm mag}$', addcontour=[par/umagtarnew/1., par/umagtarnew/0.9, par/umagtarnew/0.8], transpose=True, xrange=trange, xi_range=xi_range, title=conf)
            
        plots.somemap(rnew, tar*tscale_s, lurel, name=os.path.join(figuredir, 'q2d_p'+tag), levels=lulev, inchsize=[4, 12], cbtitle=r'$\log_{10}P/P_{\rm mag}$', addcontour=[par/umagtarnew/1.0,], transpose=True, xrange=trange, xi_range=xi_range, title=conf+": Pressure balance. Contour with P/Pmag=1")
        #in plot_bunch:
        
        # eliminate impact of zero beta:
        betar=betar+1e-12
        plots.somemap(rnew, tar*tscale_s, log10(betar), name=os.path.join(figuredir, 'q2d_b'+tag), inchsize=[4, 12], cbtitle=r'$\log_{10}\beta$', transpose=True, xrange=trange, xi_range=xi_range, title=conf+": Pgas/Ptotal")

        # Q-: in plot_bunch:
        plots.somemap(rnew, tar*tscale_s, log10(qar), name=os.path.join(figuredir, 'flux'+tag), inchsize=[4, 12], cbtitle=r'$\log_{10}Q$', transpose=True, xrange=trange, xi_range=xi_range, title=conf+": Radiation from column wall, internal units")
        
        ## temperature equivalent to  radiative flux along the tube
        #Teff_diff =   4.74501 * m1**(-0.25) * (abs(ear))**0.25 # keV    
        #new: see Teff_diff above
        # Teff from surface of sides:  
        Teff =   4.74501 * m1**(-0.25) * (abs(qar))**0.25 # keV    
        
        temperature_max = builtins.max(float(Teff_diff.max()), float(Teff.max()))
        templev = temperature_max * arange(nv) / float(nv - 2)
        
        plots.somemap(rnew, tar*tscale_s, Teff, name=os.path.join(figuredir, 'q2d_Teff_surf'+tag), inchsize=[4, 12], cbtitle=r'$T_{\rm eff,surf} ({\rm keV}) $', transpose=True, xrange=trange, xi_range=xi_range, levels=templev, title=conf+": Effective temperature at wall")

        # not in plot_bunch:
        ear=ear+1e-12 # eliminate impact of zero ear

        
        plots.somemap(rnew, tar*tscale_s, Teff_diff, name=os.path.join(figuredir, 'q2d_Teff_along'+tag), inchsize=[4, 12], cbtitle=r'$T_{\rm eff,along} ({\rm keV}) $', transpose=True, xrange=trange, ylog=False, xi_range=xi_range, levels=templev, title=conf+": Radiative flux along the tube in terms of effective temperature")

        # mdot:not in plot_bunch: there,  alternative relative value is plotted
        mdlev = 3.*arange(nv)/double(nv-2)-1.
        plots.somemap(rnew, tar*tscale_s, mdar/mdot, name=os.path.join(figuredir, 'q2d_m'+tag), inchsize=[4, 12], cbtitle=r'$s / \dot{M}$', levels=mdlev, transpose=True, xrange=trange, xi_range=xi_range, title=conf+": Radial mass rate normalized to outer value")

        # mean mdar over time
        mdmean = mdar.mean(axis=0)  
        mdstd = mdar.std(axis=0)
        
        
        if DEBUG : 
            print ("Md_mean=",mdmean) 
            # print (mdot,config[conf].getfloat('mdot'))
            # print(mdar[0])
            #print(mdar[1])
            #print(tar*tscale) 
            #print(mdot_out)
     
     
     #accretion rate at the outer boundary rises with time?????

        ##??? accretion rate at the outer boundary
        plots.someplots(tar*tscale_s, [mdot_out, mdot_out*0+mdot], formatsequence=['-k', '--k'], xlog=False, ylog=False, xtitle=r'$t$', ytitle=r'$\dot{m} (R_{\rm out})$', inchsize=[3.35, 2.], name=os.path.join(figuredir, 'mdot_req'+tag), xi_range=xi_range, title=conf)
        
        # derivative_m = mdar[:, 1:] - mdar[:, :-1]

        derivative_mdot = diff(mdar, axis=1)
        d_l = diff(lnew)


        # Reshape the last column to be 2D (with one column)
        last_element = derivative_mdot[:, -1].reshape(-1, 1)
        last_d_l =  d_l[-1]

        # Now append
        derivative_mdot = append(derivative_mdot, last_element, axis=1)
        d_l = append(d_l, last_d_l)
        #derivative_mdot = append(derivative_mdot, derivative_mdot[:,-1],axis=0)
        der_mean = derivative_mdot.mean(axis=0)


        # print(derivative_mdot, derivative_mdot.shape)
        plots.someplots(rnew, [rnew*0.+1., rnew*0., mdmean/mdot, (mdmean+mdstd)/mdot, (mdmean-mdstd)/mdot, der_mean/mdot], formatsequence=[':k', '--k', '-k', '-g', '-g', '-b'], xlog=True, ylog=True, xtitle=r'$R/R_{\rm *}$', ytitle=r'$\langle s\rangle /\dot{M}$', inchsize=[3.35, 2.], name=os.path.join(figuredir, 'q2d_mdmean'+tag), xi_range=xi_range, title=conf, charsize=9)

        # print("----> perim:",perimeternew.shape)
        # print("----> der_mean:",der_mean.shape)
        # print("----> dl",d_l.shape)
        der_mean_g = der_mean * massscale
        plots.someplots(rnew, [der_mean_g/4./pi/rnew**2/rscale**2, der_mean_g/perimeternew/d_l/rscale**2], formatsequence=['-g', '--r'], xlog=True, ylog=True, xtitle=r'$R/R_{\rm *}$', ytitle=r'${\rm g\, cm}^2 {\rm s}^{-1} $', legendsequence=[r'$\Delta \dot{M}/4\pi R^2 $', r'$\Delta \dot{M}/ \Pi {\rm d}l$'], inchsize=[3.35, 2.], name=os.path.join(figuredir, 'q2d_Sigma_loss_mean'+tag), xi_range=xi_range, title=conf, charsize=9)

        plots.someplots(arcsin(sthnew)*180/pi, [der_mean_g/4./pi/rnew**2/rscale**2, der_mean_g/perimeternew/d_l/rscale**2], formatsequence=['-g', '--r'], xlog=False, ylog=True, xtitle=r'$\theta (^{\rm o})$', ytitle=r'${\rm g\, cm}^2 {\rm s}^{-1} $', legendsequence=[r'$\Delta \dot{M}/4\pi R^2 $', r'$\Delta \dot{M}/ \Pi {\rm d}l$'], inchsize=[3.35, 2.], name=os.path.join(figuredir, 'q2d_Sigma_loss_mean_theta'+tag), xi_range=xi_range, title=conf, charsize=9)

        plots.someplots(rnew, [der_mean_g], formatsequence=['-b'], xlog=True, ylog=True, xtitle=r'$R/R_{\rm *}$', ytitle=r'$\Delta \dot M ({\rm g s}^{-1})$', inchsize=[3.35, 2.], name=os.path.join(figuredir, 'q2d_diff_mdot_mean'+tag), xi_range=xi_range, title=conf, charsize=9)
    # plots.someplots(rnew, [rnew*0.+1., rnew*0., mdmean/mdot, (mdmean+mdstd)/mdot, (mdmean-mdstd)/mdot], formatsequence = [':k', '--k', '-k', '-g', '-g'], xlog = True, ylog = True, xtitle = r'$R/R_{\rm *}$', ytitle = r'$\langle s\rangle /\dot{M}$', inchsize = [3.35, 2.], name=outdir+'/q2d_mdmean'+tag, xi_range = xi_range, title=conf,charsize=9)
 
        
        plots.someplots(tar*tscale_s, [betaeff, betaeff_m, betavent], xtitle=r'$t$, s', ytitle=r'$\beta_{\rm eff} = \frac{u+P}{\rho}\frac{R_*}{GM_*}$', formatsequence=['k.', 'r-', 'b:'], ylog=False, xlog=False, legendsequence=[r'$\beta_{\rm eff} (R_\star)$', r'?$u_{\rm mag}/u_{\rm grav}$', r'$\beta(vent)$'], name=os.path.join(figuredir, 'betaeff'+tag), yrange=[0., betaeff.max()*1.1], xi_range=xi_range, title=conf)
        
        original_stdout = sys.stdout
        save_summary = open(outdir+'/summary.txt', 'a') 
        sys.stdout = save_summary # @galja redirect print to summary  # @galja
        print("mean effective betaBS = "+str(betaeff.mean()))
        print("using magnetic energy, betaBS = "+str(betaeff_m.mean()))
        print("gas-to-total pressure ratio at the surface is "+str(betar[tar>0.9*tar.max(),0].mean()))
        
        sys.stdout = original_stdout # @galja
        save_summary.close()
   
        
    # let us also make an ASCII table with not more than max(1000, time_ar_size) time steps.
    nmax=builtins.min(1000,time_ar_size)
    ftable = open(outdir+'/ftable.dat', 'w')
    # u/umag = 3 P/Pmag?
    ftable.write('# format: t[s] -- r/rstar -- v -- u/umag -- s/mdot -- Q- -- Qdiff -- Prad -- rho -- beta_gas_rad --- \n')
    nmax = builtins.min(1000, time_ar_size)
    step = builtins.max(1, int(round(time_ar_size / nmax)))

    for kt in range(0, int(time_ar_size), step):
    # process kt
    #for kt in arange(0,time_ar_size,max(1,round(time_ar_size/nmax))):  #galja, was "for kt in arange(nt):"
        for kr in arange(nrnew):
            ftable.write(str(tar[kt]*tscale_s)+' '+str(rnew[kr])+' '+str(var[kt, kr])+' '+str(lurel[kt, kr])+' '+str(mdar[kt, kr]/mdot)+' '+str(qar[kt,kr])+' '+str(ear[kt,kr])+' '+str(par[kt, kr]/mdot)+' '+str(rhoar[kt, kr]/mdot)+' '+str(betar[kt, kr]/mdot)+'\n')
    ftable.flush()
    ftable.close()

    # rvent output
    frvent = open(outdir + '/rvent.dat', 'w')
    frvent.write('# format: t[s] -- rvent/rstar\n')

    ntime = int(time_ar_size)
    step = builtins.max(1, int(round(ntime / nmax)))

    for kt in range(0, ntime, step):
        if rvent[kt] > 1.0:
            frvent.write(f'{tar[kt] * tscale_s} {rvent[kt]}\n')

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

