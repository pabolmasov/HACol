import multiprocessing
from multiprocessing import Pool
from numpy import *

from tire_RK import alltire

print("now you can arrange for a proper config file list in multitire and launch it as multitire()")

def multitire(glolist = ['WIDE', 'WIDENOD'], nproc = None):

    nglo = size(glolist)
    if nproc is None:
        nproc = nglo
    pool = multiprocessing.Pool(nproc)
    pool.map(alltire, glolist)
                                                              
    pool.close()

#multitire(glolist = ['R_FIDU', 'R_NOD', 'R_WIDE', 'R_WIDENOD',
                   #  'R_M1', 'R_M3', 'R_M30', 'R_M100',
                   #  'R_ROT', 'R_NARROW', 'R_M1N', 'DRSMALL'
                   #  'IRR', 'RI', 'M100W', 'M100WI',
                   #  'M300W', 'HALFSIDES', 'LIGHT', 'R_HUGE',
                   #  'R_N4', 'R_V5', 'R_X', 'R_NU'])



