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

def pds(infile):
    '''
    makes a power spectrum plot
    '''
    lines = np.loadtxt(infile+".dat", comments="#", delimiter=" ", unpack=False)
    t=lines[:,0] ; l=lines[:,1]
    f=np.fft.rfft((l-l.mean())/l.std())
    freq=np.fft.rfftfreq(size(t),t[1]-t[0])
    
    pds=abs(f)**2
    clf()
    plot(freq[1:], pds[1:], 'k')
    xscale('log') ; yscale('log')
    ylabel('PDS') ; xlabel('$f$, Hz')
    savefig(infile+'_pds.png')
