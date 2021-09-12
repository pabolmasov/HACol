import postpro
import plots
import os

modellist = ['wide']

def titanfetch():

    for a in modellist:
        print("mkdir /Users/pasha/HACol/titania_"+a)
        print("sshpass -p \"i138Sal\" scp pavabo@titan.utu.fi:/home/pavabo/tired/out_"+a+"/\*.\* /Users/pasha/HACol/titania_"+a+"/")
        os.system("mkdir /Users/pasha/HACol/titania_"+a)
        os.system("sshpass -p \"i138Sal\" scp pavabo@titan.utu.fi:/home/pavabo/tired/out_"+a+"/\*.dat /Users/pasha/HACol/titania_"+a+"/")
        
'''
# stitching together:
stitch('titania_dhuge/tireout.hdf5','titania_dhuge1/tireout.hdf5') 
system('mv titania_dhuge/tirecombine.hdf5 titania_dhuge/tire123.hdf5')
stitch('titania_dhuge/tire1234.hdf5','titania_dhuge/tireout.hdf5')
  
  
quasi2d('titania_fidu/tireout.hdf5', 0,600)
quasi2d('titania_mdot30/tireout.hdf5', 0,1500, conf = 'M30')


multishock(0,5000, 1, prefix = 'titania_light/tireout', dat=False, conf='LIGHT')
multishock(0,5000, 1, prefix = 'titania_fidu/tireout', dat=False)
multishock(0,5000, 1, prefix = 'titania_fidu2/tireout', dat=False)
multishock(0,5161, 1, prefix = 'titania_fidu_old/tirecombine', dat=False)
multishock(0,4754, 1, prefix = 'titania_RI/tireout', dat=False, conf='RI', kleap = 13)   
multishock(0,9656, 1, prefix = 'titania_nod/tireout', dat=False, conf='NOD') 
multishock(0,9990, 1, prefix = 'titania_narrow/tireout', dat=False, conf='NARROW', xest=8.)
multishock(0,9990, 1, prefix = 'titania_narrow2/tireout', dat=False, conf='NARROW', xest=8.)
multishock(0,6589, 1, prefix = 'titania_mdot30/tireout', dat=False, conf='M30', xest=7.)
multishock(0,3090, 1, prefix = 'titania_mdot1/tireout', dat=False, conf='M1', xest=1.5)
multishock(0,5000, 1, prefix = 'titania_mdot3/tireout', dat=False, conf='M3')
multishock(0,9492, 1, prefix = 'titania_dhuge_old/tirecombine', dat=False, conf='DHUGE', kleap=13)
multishock(0,9492, 1, prefix = 'titania_mdot100/tireout', dat=False, conf='M100', kleap=13)
multishock(0,1377, 1, prefix = 'titania_widenod/tireout', dat=False, conf='WIDENOD')
multimultishock_plot(["titania_fidu", "titania_fidu2", "titania_light"], parflux = True)
multimultishock_plot(["titania_fidu", "titania_nod"], parflux = True)
multimultishock_plot(["titania_fidu", "titania_rot", "titania_RI"], parflux = True)
multimultishock_plot(["titania_fidu", "titania_irr"], parflux = True, sfilter = 1.2)
dynspec(infile = 'titania_huge/sfront', nbins = 20, ntimes=50, iffront = True,deline=True, fosccol = -1)

quasi2d('titania_fidu2/tireout.hdf5', 0,1500)
mdotmap(0,1500,1, prefix='titania_fidu2/tireout', conf='FIDU') 
quasi2d('titania_mdot30/tireout.hdf5', 0,4000, conf='M30')
mdotmap(0,4000,1, prefix='titania_mdot30/tireout', conf='M30') 

mdotmap(0,500,1, prefix='titania_irr/tireout', conf='IRR')  
quasi2d('titania_irr/tireout.hdf5', 0,500, conf='IRR')

mdotmap(0,500,1, prefix='titania_wide/tireout', conf='WIDE') 
quasi2d('titania_narrow2/tireout.hdf5', 0,8000, conf='NARROW', step=20)
mdotmap(0,8000,20, prefix='titania_narrow2/tireout', conf='NARROW') 

mdotmap(0,500,1, prefix='titania_mdot1/tireout', conf='M1') 
mdotmap(0,500,1, prefix='titania_mdot3/tireout', conf='M3') 
mdotmap(0,500,1, prefix='titania_mdot100/tireout', conf='M100') 
mdotmap(0,500,1, prefix='titania_mdot100w/tireout', conf='M100W') 


multimultishock_plot(["titania_dhuge", "titania_fidu"])

postpro.multishock(0,5000, 1, prefix = 'titania_fidu/tirecombine', dat=False)
plots.quasi2d('titania_fidu/tirecombine.hdf5', 0,3000)
multishock(0,7460, 1, prefix = 'titania_nod/tireout', dat=False, conf='NOD')
postpro.multishock(0,670, 1, prefix = 'titania_irr/tireout', dat=False, conf='IRR')
postpro.multishock(0,1095, 1, prefix = 'titania_rot/tireout', dat=False, conf='ROT')
postpro.multishock(0,4400, 1, prefix = 'titania_wide/tireout', dat=False, conf='WIDE')
postpro.multishock(0,6816, 1, prefix = 'titania_widenod/tireout', dat=False, conf='WIDENOD')
quasi2d('titania_widenod/tireout.hdf5', 0,6816)
postpro.multishock(0,623, 1, prefix = 'titania_mdot1/tireout', dat=False, conf='M1')
postpro.multishock(0,427, 1, prefix = 'titania_mdot3/tireout', dat=False, conf='M3')
postpro.multishock(0,1850, 1, prefix = 'titania_mdot30/tireout', dat=False, conf='M30')
multishock(0,2048, 1, prefix = 'titania_mdot100/tireout', dat=False, conf='M100')
postpro.multishock(0,1119, 1, prefix = 'titania_mdot1n/tireout', dat=False, conf='M1N')

postpro.filteredflux("titania_fidu/tireout.hdf5", 0, 1000, rfraction = 0.5)
postpro.filteredflux("titania_fidu/tireout.hdf5", 0, 670, rfraction = 0.5) 
postpro.filteredflux("titania_rot/tireout.hdf5", 0, 1000, rfraction = 0.5)
postpro.filteredflux("titania_irr/tireout.hdf5", 0, 1000, rfraction = 0.5)
postpro.filteredflux("titania_wide/tireout.hdf5", 0, 1000, rfraction = 0.5)
postpro.filteredflux("titania_xireal/tireout.hdf5", 0, 1000, rfraction = 0.5)
multishock(0,8900, 1, prefix = 'titania_mdot100/tireout', dat=False, mdot=100.*4.*pi, kleap=3)
plots.quasi2d('titania_mdot3/tireout.hdf5', 0,427, conf='M3')
plots.quasi2d('titania_mdot30/tireout.hdf5', 0,1850, conf='M30')
plots.quasi2d('titania_mdot1/tireout.hdf5', 0,182, conf='M1') 
quasi2d('titania_mdot1n/tireout.hdf5', 0,1119, conf='M1N') 
plots.quasi2d('titania_wide/tireout.hdf5', 0,4000, conf='WIDE')
plots.quasi2d('titania_rot/tireout.hdf5', 0,1095, conf='ROT')
plots.quasi2d('titania_irr/tireout.hdf5', 0,670, conf='IRR')
plots.quasi2d('titania_v5/tireout.hdf5', 0,670, conf='V5') 
plots.quasi2d('titania_v30/tireout.hdf5', 0,670, conf='V30') 

plots.quasi2d('titania_rot/tireout.hdf5', 0,300)
plots.quasi2d('titania_mdot100/tireout.hdf5', 0,300)
plots.twomultishock_plot("titania_fidu/flux", "titania_fidu/sfront", "titania_rot/flux", "titania_rot/sfront")
plots.twomultishock_plot("titania_fidu/flux", "titania_fidu/sfront", "titania_irr/flux", "titania_irr/sfront")
postpro.comparer('galia_F/BS_solution_F', 'titania_fidu1/tireout04000')
comparer('galia_M30/BS_solution_M30', 'titania_mdot30/tireout06000', vone = -1.170382e+06/3e10)
comparer('galia_F/BS_solution_F', 'titania_BS/tireout07000', vone = -9.778880e+05/3e10)
comparer('galia_M100/BS_solution_M100', 'titania_mdot100/tireout06000', vone = -1.957280e+06/3e10)
comparer('galia_N/BS_solution_N', 'titania_narrow2/tireout20000', vone = -8.194837e+06/3e10)

'''
