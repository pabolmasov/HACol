import matplotlib
from matplotlib import rc
from matplotlib import axes
from numpy import *
from pylab import *
from scipy.integrate import *
from scipy.interpolate import *

#Uncomment the following if you want to use LaTeX in figures 
rc('font',**{'family':'serif'})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
# #add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"] 

from globals import ifplot

def pds(infile='flux'):
    '''
    makes a power spectrum plot;
    input infile+'.dat' is an ascii, 2+column dat-file with an optional comment sign of #
    '''
    lines = np.loadtxt(infile+".dat", comments="#", delimiter=" ", unpack=False)
    t=lines[:,0] ; l=lines[:,1]
    f=np.fft.rfft((l-l.mean())/l.std())
    freq=np.fft.rfftfreq(size(t),t[1]-t[0])
    
    pds=abs(f)**2
    if ifplot:
        clf()
        plot(freq[1:], pds[1:], 'k')
        xscale('log') ; yscale('log')
        ylabel('PDS') ; xlabel('$f$, Hz')
        savefig(infile+'_pds.png')
    
    # additional ascii output:
    fpds=open(infile+'_pds.dat', 'w')
    for k in arange(size(freq)-1)+1:
        fpds.write(str(freq[k])+' '+str(pds[k])+'\n')
    fpds.close()
    
