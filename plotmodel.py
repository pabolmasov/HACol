import g_postpro as pp
import plots
import os
import auxtools as au 
import g_hdfoutput as hdf
from addplot import *  # for massrace
from subsonic import *

import sys   # to direct stdout to a file  @galja


currentdir = os.getcwd()


modellist = ['FIDU','M1100Wdr10','M1100Wdr10x2']
modellist = sys.argv[1:]
print(" ----- Model list consists of:",modellist)

filter_data=1 # flag to make sparse data and plot a bit faster
plot_before_filter=1
#hdf.stitch('SUBSO3/tireout.hdf5','SUBSO3ext1/tireout.hdf5') 



for mod in modellist:
    print ("\n\nPlot model "+mod)

    outdir = currentdir+"/"+au.get_dir_for_model(mod)

    print ("\n\nDir path: "+outdir)

    original_stdout = sys.stdout 
    save_summary = open(outdir+'/summary.txt', 'w') 
    print("\n")
    save_summary.close() # @galja
    sys.stdout = original_stdout
    

    if plot_before_filter : 
        massrace([outdir,],[mod,]);
    

    
    hfdfile = '/tireout'
    
    if plot_before_filter : 
        #Условие на давление на поверхности НЗ:
        pp.boundary_evolution(prefix = outdir+hfdfile,conf=mod, dat=True)
    
    size = au.count_lines(outdir+'/flux.dat')-10

    flux_data = loadtxt(outdir+'/flux.dat', comments='#')
    time_min = flux_data[:, 0].min()
    time_max = flux_data[:, 0].max()
    time_duration = time_max - time_min

    

    print ("\n\n ----- Plot radial distributions: "+outdir)
    nouts = (0,)
    for nout in nouts:
        print ("\n\nFor n=",nout)
        pp.taus(nout, prefix = outdir+'/tireout{:05d}'.format(nout), ifhdf = False, conf = mod)
    
    # get the dat-file name with the last entry and plot distributions from it:
    filelast=pp.last_entry_dat_file (prefix = outdir+'/tireout')
    print("\n\nFor file with last entry: " ,filelast);
    # remove ".dat":
    


    if plot_before_filter : 
        pp.taus(0, prefix = filelast[:-4] , ifhdf = False, conf = mod)
    
    print ("\n\nNum of dumps in flux.dat - 10 =",size)
    
    size_filtered=2000
    if ( (filter_data) and (os.path.exists(outdir+'/tire_small.hdf5')==0 ) ) :
        hdf.write_low_time_res_hdf (outdir,'tire_small.hdf5','tireout.hdf5',size_filtered)
        
        
    if filter_data :
        print ("Plot figures using full file? [Y/n]");
        answer = input();
        if ((answer == "") or (answer == "Y") or (answer == "y")):
            1
        else:
            hfdfile = '/tire_small';
    ext = ".hdf5"    
    
    #refresh number:    
    size_filtered = hdf.num_bunches_total (outdir+hfdfile+ext)    
    print ("Num of dumps in ",outdir+hfdfile+ext, "=",size_filtered)
    hdf.get_name_last_entry(outdir+hfdfile+ext)    
    
    first = hdf.get_num_1st_entry(outdir+hfdfile+ext)  #for filterd hdf5 ALWAYS first = 0(?)
    size = size_filtered
    step =1
    last = first + size - 1
   
    print ("\n first=", first, " size=", size, " last=",last);
    calc = 2
    # timerange = [0., 0.1 ]
    # timerange = [0.022, 0.5]  # out_mdot1100w10x2
    # if ftable does not exist:
    
    
    
   
    
    #print ("*** Plot along analytic solution")
    ##acomparer сравнивает с аналитическим решением
    ##nentry задает диапазон номеров записей для осреднения
    ## to make file avprofile.dat:
    #pp.acomparer(outdir+hfdfile, nentry = [first+round(0.95*size),first+size-1], conf=mod, nocalc=False)
    ## to plot nicely:
    #pp.acomparer(outdir+hfdfile, nentry = [first+round(0.95*size),first+size-1], conf=mod, nocalc=True)
    #exit()
    
    #pp.acomparer(outdir+hfdfile, nentry = [400,600], conf=mod, nocalc=False)
    #### to plot nicely:
    #pp.acomparer(outdir+hfdfile, nentry = [400,600], conf=mod, nocalc=True)
    #exit()
   
    
    # plot tau:
    pp.taus(round(size-(size-first)/2), prefix = outdir+hfdfile, ifhdf = True, conf = mod)
    
    
    # plot masses:
    pp.masstest (outdir, conf = mod) 
    
    
    #after multishock:
    #pp.dynspec(infile = outdir+'/sfront', nbins = 20, ntimes=15, iffront = True,deline=False, fosccol = -1,stnorm = False)
               #trange = [0.,0.5], 

   
    #size=round(size/10)
    if ((calc==2) or (os.path.exists(outdir+'/ftable.dat')==0)):
        print (" ----- Calculate 2D file ftable.dat...")
        # это двумерная картинка, аргументы: имя файла, номер первой записи и количество записей:  
        print ("... plot_model from:",first," to ", size-1,"\n")
        pp.quasi2d(outdir+hfdfile+ext,first,size-1,conf=mod)
        #,trange=timerange)
    
    #for filterd hdf5 ALWAYS first = 0
    #pp.quasi2d(outdir+'/tire_small.hdf5',0, size, conf=mod, trange=timerange, tag='_small')
    
    
    #print ("*** Plot 2D")
    #quasi2d_nocalc нужен, чтобы по двумерной карте, построенной удаленно и сохраненной в ascii, построить рисунок
    #pp.quasi2d_nocalc(outdir+'/ftable.dat', conf=mod, trange=timerange,tag='')
    #pp.gquasiplot3(outdir+'/ftable.dat', conf=mod, trange=timerange)

   
    print ("\n\n ----- Plot shock evolution")
    #положение фронта: 
    pp.multishock(first ,size-1, 1, prefix = outdir+hfdfile,conf=mod)
    #первые два аргумента, как в quasi2d, — номера элементов, начало и количество записей
    pp.dynspec(infile = outdir+'/sfront', nbins = 20, ntimes=15, iffront = True, deline=False, fosccol = -1, trange = [0.,0.5], stnorm = False)
    
    #continue

    
    print ("\n ----- Plot along analytic solution")
    #acomparer сравнивает с аналитическим решением
    #nentry задает диапазон номеров записей для осреднения
    # to make file avprofile.dat:
    pp.acomparer(outdir+hfdfile, nentry = [first+round(0.95*size),first+size-1], conf=mod, nocalc=False)
    # to plot nicely:
    pp.acomparer(outdir+hfdfile, nentry = [first+round(0.95*size),first+size-1], conf=mod, nocalc=True)
   
   
    early_trange = [time_min, time_min + 0.2*time_duration]
    late_trange = [time_min + 0.8*time_duration, time_max]
    # plot early result:
    print("\n ----- Plot 2D for early times")

    early_first = first
    early_size = max(1, int(0.2 * size))


    if early_first + (early_size - 1) <= last:
        print("\n ----- Plot 2D for early times, from", early_first, "with", early_size, "entries")
        # pp.quasi2d(outdir+hfdfile+ext, early_first, early_size, conf=mod, step=step, tag='_early')
        pp.quasi2d(outdir+hfdfile+ext, first, size, conf=mod, step=step, trange=early_trange, tag='_early')
    else:
        print("Early-time range exceeds available entries")


    print("\n ----- Plot 2D for late times")

    late_first = first + int(0.95 * size)

    late_size = size - (late_first - first)


    if late_size > 0:
        print("\n ----- Plot 2D for late times, from", late_first, "with", late_size, "entries")
        pp.quasi2d(outdir+hfdfile+ext, first, size, conf=mod, step=step, trange=late_trange, tag='_late')
        # pp.quasi2d(outdir+hfdfile+ext, late_first, late_size, conf=mod, step=step, tag='_late')
    else:
        print("Late-time range is empty")
    

    


#ok betaeff.png fluxshock.png  lumshocks.png q2d_b.png  q2d_mdmean.png  q2d_m.png  q2d_q.png  q2d_u.png  q2d_vmean.png  q2d_v.png  shockfront.png  uvcheck.png  ux.png  vleap.png 






# tmax=50 примерно какой размер hdf5 дает?
# dtout = 0.001
# tmax/dtout =  5*10^4  - количество дампов (выводов)
# при nx=1200 примерно 0.04МБ на дамп

## EXMAPLES:

# quasi2d('out_fidu/tireout.hdf5', 0,600)
# это двумерная картинка, аргументы: имя файла, номер первой записи и номер последней
# postpro.quasi2d('out_RI/tireout.hdf5', 0,22000, step=10, conf='irr')
# postpro.quasi2d('out_rot/tireout.hdf5', 0,5000, step=1, conf='rot')
# !! обязательно должно быть название конфигурации, чтобы при рисовании использовались правильные глобальные переменные
# quasi2d_nocalc('titania_mdot1/ftable.dat', conf='M1', trange=[0., 0.05])
# quasi2d_nocalc('titania_fidu2/ftable.dat', trange=[0., 0.05])
# quasi2d_nocalc('titania_mdot100/ftable.dat', conf='M100', trange=[0., 0.05])



# postpro.acomparer('out_bs/tireout', nentry = [40000,50000])
# postpro.acomparer('out_narrow2/tireout', nentry = [4000,5000], conf='NARROW2')
# postpro.quasi2d('out_narrow2/tireout.hdf5', 4000,5000, conf = 'NARROW2')
# postpro.acomparer('out_mdot3/tireout', nentry = [15000,20000], conf='M3')
# postpro.acomparer('out_mdot1/tireout', nentry = [20000,26000], conf='M1')
# postpro.quasi2d('out_RI/tireout.hdf5', 0,22000, step=10, conf='irr')
# postpro.quasi2d('out_rot/tireout.hdf5', 0,5000, step=1, conf='rot')
# postpro.quasi2d('out_RI/tireout.hdf5', 0,5000, step=1, conf='RI')
