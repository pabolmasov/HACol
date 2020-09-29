# from tire_RK import configactual, m1, mdot, eta, afac, r_e, dr_e, omega, rstar
import h5py
import os.path
from numpy import arange, size

def entryname(n, ndig = 6):
    entry = str(n).rjust(ndig, '0') # allows for 6 positions (hundreds of thousand of entries)
    return entry

def init(hname, g, configactual): # , m1, mdot, eta, afac, re, dre, omega):
    '''
    writing globals and geometry to the output HDF5 file
    '''
    hfile = h5py.File(hname, "w")
    glo = hfile.create_group("globals")
    print(configactual['outdir']+": omega = "+str(configactual.getfloat('omega')))
    #    input("oo")
    glo.attrs['m1']      = configactual.getfloat('m1')
    glo.attrs['mdot']      = configactual.getfloat('mdot')
    glo.attrs['eta']      = configactual.getfloat('eta')
    glo.attrs['afac']      = configactual.getfloat('afac')
    glo.attrs['re']      = configactual.getfloat('r_e')
    glo.attrs['dre']      = configactual.getfloat('dr_e')
    glo.attrs['omega']      = configactual.getfloat('omega')
    glo.attrs['rstar']      = configactual.getfloat('rstar')
    glo.attrs['umag']      = configactual.getfloat('umag')

    geom = hfile.create_group("geometry")
    geom.create_dataset("l", data=g.l)
    geom.create_dataset("r", data=g.r)
    geom.create_dataset("sth", data=g.sth)
    geom.create_dataset("cth", data=g.cth)
    
    hfile.flush()
    return hfile # returns file stream reference
    
def dump(hfile, nout, t, rho, v, u, qloss):
    '''
    writing one snapshot
    '''
    entry = entryname(nout)
    grp = hfile.create_group("entry"+entry)
    grp.attrs["t"] = t
    grp.create_dataset("rho", data=rho)
    grp.create_dataset("v", data=v)
    grp.create_dataset("u", data=u)
    grp.create_dataset("qloss", data=qloss)
    hfile.flush()
    print("HDF5 output, entry"+entry+"\n", flush=True)

def close(hfile):
    hfile.close()

#########################
def keyshow(filename):
    '''
    showing the list of keys (entries) in a given data file
    '''
    f = h5py.File(filename,'r', libver='latest')
    keys = list(f.keys())
    #    print(list(f.keys()))
    f.close()
    return keys

def read(hname, nentry):
    '''
    read a single entry from an HDF5
    '''
    glosave = dict()
    hfile = h5py.File(hname, "r", libver='latest')
    geom=hfile["geometry"]
    glo=hfile["globals"]
    glosave["rstar"] = glo.attrs["rstar"]
    glosave["mdot"] = glo.attrs["mdot"]
    glosave["umag"] = glo.attrs["umag"]
    rstar=glo.attrs["rstar"]
    entry = entryname(nentry)
    l=geom["l"][:]  ;  r=geom["r"][:] ;  sth=geom["sth"][:] # reading geometry
    data=hfile["entry"+entry]
    rho=data["rho"][:] ; u=data["u"][:] ; v=data["v"][:] # reading the snapshot
    qloss = data["qloss"][:]
    t=data.attrs["t"]
    print("t="+str(t)+" ("+str(nentry)+")")
    hfile.close()
    return entry, t, l, r/rstar, sth, rho, u, v, qloss, glosave

def stitch(hname1, hname2):
    '''
    reads to HDF outputs and stitches them together
    '''
    hfile1 = h5py.File(hname1, "r")
    hfile2 = h5py.File(hname2, "r")
    # globals are taken from the first file:
    glo1=hfile1["globals"] 
    # geometry could be different
    # TODO: added interpolation for the case when geom1 != geom2
    geom1=hfile1["geometry"]

    print(os.path.dirname(hname1)+'/tirecombine.hdf5')
    hnew = h5py.File(os.path.dirname(hname1)+'/tirecombine.hdf5', "w")
    
    glo = hnew.create_group("globals")
    geom = hnew.create_group("geometry")
    
    # hnew.copy(glo1, glo) ; hnew.copy(geom1, geom)
    # group.copy does not work, for some reason
    globalkeys = glo1.attrs.keys()
    for k in globalkeys:
        glo.attrs[k] = glo1.attrs[k]
    geokeys = geom1.keys()
    for k in geokeys:
        geom.create_dataset(k, data=geom1[k])
    print(glo.attrs["rstar"])
    print(geom["l"])
    #    ii=input("k")
   
    # all the entries, excluding globals and geometry
    keys1 = list(hfile1.keys())[:-2] ; keys2 = list(hfile2.keys())[:-2] 

    for k in arange(size(keys1)):
        entry = keys1[k]
        print("From "+hname1+", entry "+entry+"\n", flush=True)
        grp = hnew.create_group(entry)
        data = hfile1[entry]
        grp.attrs["t"] = data.attrs["t"]
        grp.create_dataset("rho", data=data["rho"][:])
        grp.create_dataset("v", data=data["v"][:])
        grp.create_dataset("u", data=data["u"][:])
        grp.create_dataset("qloss", data=data["qloss"][:])
        hnew.flush()

    # removing duplicates:
    keys22 = [i for i in keys2 if i not in keys1]
        
    for k in arange(size(keys22)):
        entry = keys22[k]
        grp = hnew.create_group(entry)
        data = hfile2[entry]
        grp.attrs["t"] = data.attrs["t"]
        grp.create_dataset("rho", data=data["rho"][:])
        grp.create_dataset("v", data=data["v"][:])
        grp.create_dataset("u", data=data["u"][:])
        grp.create_dataset("qloss", data=data["qloss"][:])
        hnew.flush()
        print("From "+hname2+", entry"+entry+"\n", flush=True)
        
    hnew.close()

def toasc(hname='tireout.hdf5', nentry=0):
    '''
    convert a single HDF5 entry to an ascii table
    '''
    entry, t, l, r, sth, rho, u, v = read(hname, nentry)

    nr=size(r)
    # write an ascii file
    fout = open(hname+'_'+entry, 'w')
    fout.write('# t = '+str(t)+'\n')
    fout.write('# format: l -- rho -- v -- u\n')
    for k in arange(nr):
        fout.write(str(r[k])+" "+str(rho[k])+" "+str(v[k])+" "+str(u[k])+"\n")
    fout.close()
    
def multitoasc(n1, n2, no,hname='tireout.hdf5'):
    '''
    running toasc for a set of frames
    '''
    for k in linspace(n1,n2, num=no, dtype=int):
        toasc(hname=hname, nentry=k)
        print(k)
    
