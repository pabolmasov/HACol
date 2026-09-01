# from tire_RK import configactual, m1, mdot, eta, afac, r_e, dr_e, omega, rstar
import h5py
import os.path
from numpy import arange, size, zeros
from scipy.interpolate import interp1d

verbatim = 0

def entryname(n, ndig = 6):
    entry = str(n).rjust(ndig, '0') # allows for 6 positions (hundreds of thousand of entries)
    return entry


def write_low_time_res_hdf (outdir,h_new_name,h_old_name,new_n_entry=500):
    cwd = os.getcwd()
    os.chdir(outdir)
    print ("\nMake hdf with ",new_n_entry," dumps:")
    print ('Source file: ',h_old_name)
    if not os.path.exists(h_old_name):
        print ("There is no file "+h_old_name);
        if os.path.exists(h_new_name):
            h_new_file = h5py.File(h_new_name, 'r', libver='latest')
            new_n_entry = size(h_new_file)-2
            print ("But there is a file "+h_new_name+" with ",new_n_entry," dumps");
            return
        else:
            print ("There is neither "+h_new_name+". Exit.")
            exit()
            
    h_old_file = h5py.File(h_old_name, 'r', libver='latest')
    
    old_n_entry = size(h_old_file)-2
    entry_ar = stored_entries_nums (h_old_name)
    skip = max(1,round(old_n_entry/new_n_entry))
    
    #check that new file does not exists and remove if it does
    if os.path.exists(h_new_name):
        os.remove(h_new_name)
    else:
        print("Can not delete the file",h_new_name, ", as it doesn't exists")
    
    #pathlib.Path(h_new_name).unlink()
    
    # Just make a symbolic link if the new size is the same as the old size
    if (skip==1):
        print("New file would have the same size; make a symbolink link: ",h_old_name, h_new_name)
        os.symlink(h_old_name, h_new_name)
        os.chdir(cwd)
        return

    print ("geo: " )
    print (list(h_old_file['geometry']))
    
    print ("glo: " )
    print (list(h_old_file['globals']))
    
    h_new_file = h5py.File(h_new_name, 'w', libver='latest')
    
    #copy groups to a new file:
    h_old_file.copy('geometry', h_new_file)
    h_old_file.copy('globals', h_new_file)
    
    for i in range(0, old_n_entry, skip):
        entry = entryname(entry_ar[i])
        h_old_file.copy("entry"+entry, h_new_file)
 
    close(h_new_file)
    close(h_old_file)
    
    print ("New number of dumps: ",num_bunches_total(h_new_name))
    print ('New file:',h_new_name)
    os.chdir(cwd)
    return 1
    
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
    
def dump(hfile, nout, t, rho, v, u, qloss, ediff):
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
    if size(ediff) > 1:
        grp.create_dataset("ediff", data=ediff)
    else:
        grp.create_dataset("ediff", data=qloss * 0.)
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
    # print(list(f.keys()))
    f.close()
    return keys

def stored_entries_nums (filename):
    import re
    if not os.path.exists(filename):
        print ("There is no file "+filename);
        exit()
    keys = keyshow(filename)
    #print(size(filename))
    dumps_array = zeros(num_bunches_total(filename),dtype='int')
    i = 0 
    for k in keys:
        #print (k)
        match  = re.search(r"^entry(\d+)", k)
        #print ("M=",match)
        if (match): 
            dumps_array[i] = int(match.group(1))
            i += 1
    return dumps_array    

def ndumps (hname):
    hfile = h5py.File(hname, 'r', libver='latest')
    num = size(hfile)
    hfile.close()
    return num

def num_bunches_total (filename):
    #print(ndumps (filename))
    #print("1st=",get_num_1st_entry (filename))
    #entry_ar = stored_entries_nums (filename)
    #entry_name = "entry"+entryname(entry_ar[-1])
    #print ("The last one is",entry_name+"")
    #exit()
    return ndumps (filename)-2

def get_name_last_entry (filename):
    size = num_bunches_total (filename)
    print("There are",size,"entries altogether")
    entry_ar = stored_entries_nums (filename)
    entry_name = "entry"+entryname(entry_ar[0])
    print ("The first one is",entry_name)
    entry_name = "entry"+entryname(entry_ar[-1])
    print ("The last one is",entry_name)
    #print(read(filename,entry_ar[-1]))
    return (entry_name, entry_ar[-1])

def get_num_1st_entry (filename):
    entry_ar = stored_entries_nums (filename)
    return entry_ar[0]

def check_entry_exists (hname, nentry):
    hfile = h5py.File(hname, 'r', libver='latest')
    entry = entryname(nentry)
    if "entry"+entry in hfile:
        return 1
    else:
        return 0;

def read(hname, nentry):
    '''
    read a single entry from an HDF5
    '''
    glosave = dict()
    hfile = h5py.File(hname, 'r', libver='latest')
    geom=hfile["geometry"]
    glo=hfile["globals"]
    glosave["rstar"] = glo.attrs["rstar"]
    glosave["mdot"] = glo.attrs["mdot"]
    glosave["umag"] = glo.attrs["umag"]
    rstar=glo.attrs["rstar"]
    entry = entryname(nentry)
    l=geom["l"][:]  ;  r=geom["r"][:] ;  sth=geom["sth"][:] # reading geometry
    
    #print ("EN =",entry)
    if "entry"+entry in hfile:
        data=hfile["entry"+entry]
    else:
        return -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
   
    rho=data["rho"][:] ; u=data["u"][:] ; v=data["v"][:] # reading the snapshot
    qloss = data["qloss"][:]
    ediff = data["ediff"][:]
    t=data.attrs["t"]
    if (verbatim) :
        print("t="+str(t)+" ("+str(nentry)+")")
    hfile.close()
    return entry, t, l, r/rstar, sth, rho, u, v, qloss, glosave, ediff

def liststitch(hnamelist):
    '''
    reads HDF5 outputs from the list and stitches them together
    '''

    nfiles = size(hnamelist)
    # globals are taken from the first file:
    hfile0 = h5py.File(hnamelist[0], "r")
    glo0=hfile0["globals"] 
    geom0=hfile0["geometry"]
    print(os.path.dirname(hnamelist[0])+'/tire_lcombine.hdf5')
    hnew = h5py.File(os.path.dirname(hnamelist[0])+'/tire_lcombine.hdf5', "w")
    
    glo = hnew.create_group("globals")
    geom = hnew.create_group("geometry")
    globalkeys = glo0.attrs.keys()
    for k in globalkeys:
        glo.attrs[k] = glo0.attrs[k]
        print(k)
    geokeys = geom0.keys()
    for k in geokeys:
        geom.create_dataset(k, data=geom0[k])
        print(k)
    print(glo.attrs["rstar"])
    nx0 = size(geom["l"])

    keys0 = list(hfile0.keys())[:-2]

    keys = []
    
    for q in arange(nfiles):
        print("reading file "+str(hnamelist[q]))
        hfile1 = h5py.File(hnamelist[q], "r")
        glo1=hfile1["globals"] 
        geom1=hfile1["geometry"]
        nx1 = size(geom1["l"])
        keys1 = list(hfile1.keys())[:-2]       
        keys11 = [i for i in keys1 if i not in keys]
        keys = keys + keys11
        for k in arange(size(keys11)):
            entry = keys11[k]
            print("From "+hnamelist[q]+", entry "+entry+"\n", flush=True)
            grp = hnew.create_group(entry)
            data = hfile1[entry]
            grp.attrs["t"] = data.attrs["t"]
            if nx1 == nx0:
                grp.create_dataset("rho", data=data["rho"][:])
                grp.create_dataset("v", data=data["v"][:])
                grp.create_dataset("u", data=data["u"][:])
                grp.create_dataset("qloss", data=data["qloss"][:])
            else:
                print("interpolating from a "+str(nx1)+" to a "+str(nx0)+" grid")
                rhofun = interp1d(geom1["l"], data["rho"][:])
                vfun = interp1d(geom1["l"], data["v"][:])
                ufun = interp1d(geom1["l"], data["u"][:])
                qfun = interp1d(geom1["l"], data["qloss"][:])
                grp.create_dataset("rho", data=rhofun(geom["l"]))
                grp.create_dataset("v", data=vfun(geom["l"]))
                grp.create_dataset("u", data=ufun(geom["l"]))
                grp.create_dataset("qloss", data=qfun(geom["l"]))
            hnew.flush()
        hfile1.close()
        #  ii = input('file')
    hnew.close()
        
    
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
    
