import geometry as geo

from g_consts import Consts

import numpy as np

def model_m1(config, conf='DEFAULT'):
    return config[conf].getfloat('m1')

def tscale_raw(config, conf='DEFAULT'):
    return config[conf].getfloat('tscale')


def tscale_s(config, conf='DEFAULT'):
    return tscale_raw(config, conf) * model_m1(config, conf)

def code_time_to_s(time_code, config, conf='DEFAULT'):
    return time_code * tscale_s(config, conf)

def seconds_to_code_time(time_s, config, conf='DEFAULT'):
    return time_s / tscale_s(config, conf)

def bs_travel_time_s(dt_dimensionless, rstar, config, conf='DEFAULT'):
    return tscale_s(config, conf) * rstar**1.5 * model_m1(config, conf) * dt_dimensionless



def massscale_raw(config, conf='DEFAULT'):
    return config[conf].getfloat('massscale')

def massscale_g(config, conf='DEFAULT'):
    return massscale_raw(config, conf) * model_m1(config, conf) * model_m1(config, conf)



def mdotscale_raw(config, conf='DEFAULT'):
    return config[conf].getfloat('mdotscale')

def mdotscale_g_s(config, conf='DEFAULT'):
    return mdotscale_raw(config, conf) * model_m1(config, conf)


def umag_calc_polar (b12, m1) :
    # polar magnetic pressure B2/8/pi at the NS surface   (units of pressure?)
    return (b12**2*2.29e6*m1)


def umag_calc_local (b12, m1, th0) :
    # magnetic pressure B2/8/pi at the NS surface at specific magnetic latitude
    return umag_calc_polar (b12, m1) * (1.+3.*(np.cos(th0))**2)/4.

def umag_calc_polar_phys (b12) :
    # polar magnetic pressure B2/8/pi at the NS surface is SGC units, b12 = B(Gauss)/1e12

    return (b12**2 * 1e24 / 8. / Consts['Pi'])

def umag_calc_local_phys (b12, costh0) :
    # magnetic pressure B2/8/pi at the NS surface is SGC units, b12 = B(Gauss)/1e12 at the NS surface at specific magnetic latitude
    # costh0 = cos(theta0), at NS surface
    return umag_calc_polar_phys (b12) * (1.+3.*(costh0)**2)/4.



def r_e (config, conf='DEFAULT') :

    mu30 = config[conf].getfloat('mu30') # magnetic moment, 10^{30} Gs cm^3 units
    m1 = config[conf].getfloat('m1') # NS mass (solar units)
    mdot = config[conf].getfloat('mdot') * 4. * Consts['Pi'] # internal units, GM/varkappa c
    rstar = config[conf].getfloat('rstar') # GM/c^2 units
    xifac = config[conf].getfloat('xifac')
    return config[conf].getfloat('r_e_coeff') * (mu30**2/mdot)**(2./7.)*m1**(-10./7.) * xifac # magnetosphere radius


def dr_e (config, conf='DEFAULT') :
    return config[conf].getfloat('drrat') * r_e (config, conf=conf)

def Led (config, conf='DEFAULT'):
    # 2.0008667042014996e+38
    kappa = Consts['C']**2/( Consts['G'] *Consts['Msun'] *config[conf].getfloat('rhoscale')  )
    return 4.0 * Consts['Pi'] * Consts['G'] *Consts['Msun'] *model_m1(config, conf) *Consts['C'] /kappa


def tr_phys(config, conf='DEFAULT'):
    phys_factor_to_cm = Consts['G'] * Consts['Msun'] * model_m1(config, conf) / Consts['C']**2
    rstar = config[conf].getfloat('rstar')
    mu30 = config[conf].getfloat('mu30')
    m1 = model_m1(config, conf)
    afac = config[conf].getfloat('afac')
    mdot = config[conf].getfloat('mdot') * 4. * Consts['Pi'] # internal units, GM/varkappa c
    rstar_phys = rstar * phys_factor_to_cm
    re_phys = r_e(config, conf=conf) * phys_factor_to_cm
    dr_e_phys = dr_e(config, conf=conf) * phys_factor_to_cm
    if re_phys <= rstar_phys: raise ValueError("tr_phys: magnetospheric radius must exceed stellar radius; Re={} cm, Rstar={} cm".format(re_phys, rstar_phys))
    sth0 = (rstar_phys / re_phys) ** 0.5
    cth0 = (1.0 - rstar_phys / re_phys) ** 0.5
    delta0 = rstar_phys * sth0 / (1.0 + 3.0 * cth0**2)**0.5 * dr_e_phys / re_phys
    Across = 4.0 * Consts['Pi'] * afac * rstar_phys * sth0 * delta0
    b12 = 2.0 * mu30 * (rstar * m1 / 6.8)**(-3)
    umag0 = umag_calc_local_phys(b12, cth0)
    # print ("Rstar=",rstar_phys)
    # print ("mdotscale_g_s=", mdot*mdotscale_g_s(config, conf=conf) )
    # print ("The latter should be equal to:")
    # print ("check=",2000.*Led(config, conf=conf)/Consts['C'] /Consts['C'] )
    return Across * umag0 * rstar_phys**2 / mdot/ mdotscale_g_s(config, conf=conf) / Consts['G'] / Consts['Msun']
