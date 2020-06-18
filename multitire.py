import multiprocessing
from multiprocessing import Pool
from numpy import *

from tire_RK import alltire

print("now you can arrange for a proper config file list in multitire and launch it as multitire()")

def multitire():

    nproc = 2

    glolist = ['FIDU', 'ROT']
    nglo = size(glolist)
    pool = multiprocessing.Pool(nproc)
    pool.map(alltire, glolist)
    #    for k in arange(nglo):
    #        Process(target=alltire, args=(glolist[k],))    
    pool.close()
