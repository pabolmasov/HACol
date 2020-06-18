import multiprocessing
from multiprocessing import Pool
from numpy import *

from tire_RK import alltire

print("now you can arrange for a proper config file list in multitire and launch it as multitire()")

def multitire(glolist = ['FIDU', 'ROT'], nproc = None):

    nglo = size(glolist)
    if nproc is None:
        nproc = nglo
    pool = multiprocessing.Pool(nproc)
    pool.map(alltire, glolist)
                                                              
    pool.close()

#multitire(glolist = ['FIDU', 'NOD', 'WIDE', 'WIDENOD', 'M1', 'M3', 'M30', 'M100', 'ROT', 'IRR', 'RI', 'NU', 'V5', 'V30', 'N4', 'X'])



