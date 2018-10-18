import h5py
from globals import *

def entryname(n):
    entry = str(n).rjust(6, '0') # allows for 6 positions (hundreds of thousand of entries)
    return entry

def init(hname, l, r, sth, cth): # , m1, mdot, eta, afac, re, dre, omega):
    '''
    writing globals and geometry to the output HDF5 file
    '''
    hfile = h5py.File(hname, "w")
    glo = hfile.create_group("globals")
    glo.attrs['m1']      = m1
    glo.attrs['mdot']      = mdot
    glo.attrs['eta']      = eta
    glo.attrs['afac']      = afac
    glo.attrs['re']      = re
    glo.attrs['dre']      = dre
    glo.attrs['omega']      = omega
    glo.attrs['rstar']      = rstar

    geom = hfile.create_group("geometry")
    geom.create_dataset("l", data=l)
    geom.create_dataset("r", data=r)
    geom.create_dataset("sth", data=sth)
    geom.create_dataset("cth", data=cth)
    
    hfile.flush()
    return hfile # returns file stream reference
    
def dump(hfile, nout, t, rho, v, u):
    '''
    writing one snapshot
    '''
    entry = entryname(nout)
    grp = hfile.create_group("entry"+entry)
    grp.attrs["t"] = t
    grp.create_dataset("rho", data=rho)
    grp.create_dataset("v", data=v)
    grp.create_dataset("u", data=u)
    hfile.flush()

def close(hfile):
    hfile.close()

#########################

def read(hname, nentry):
    '''
    read a single entry from an HDF5
    '''
    hfile = h5py.File(hname, "r")
    geom=hfile["geometry"]
    glo=hfile["globals"]
    rstar=glo.attrs["rstar"]
    entry = entryname(nentry)
    l=geom["l"][:]  ;  r=geom["r"][:] ;  sth=geom["sth"][:] # reading geometry
    data=hfile["entry"+entry]
    rho=data["rho"][:] ; u=data["u"][:] ; v=data["v"][:] # reading the snapshot
    t=data.attrs["t"]
    print("t="+str(t))
    hfile.close()
    return entry, t, l, r/rstar, sth, rho, u, v 

def toasc(hname='tireout.hdf5', nentry=0):
    '''
    convert a single HDF5 entry to an ascii table
    '''
    entry, t, l, r, sth, rho, u, v = read(hname, nentry)

    nr=size(r)
    # write an ascii file
    fout = open(hname+'_'+entry, 'w')
    for k in arange(nr):
        fout.write(str(r[k])+" "+str(rho[k])+" "+str(v[k])+" "+str(u[k])+"\n")
    fout.close()
    
