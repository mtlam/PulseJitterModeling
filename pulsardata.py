import numpy as np
import glob
import sys
sys.path.append("/home/michael/Research/Noise Budget/source/")
import DISS
import utilities as u
import psrcat
import pypulse.par as par
import scipy.interpolate as interpolate
import jittermodels as jm

from matplotlib.pyplot import *

BANDS = [327,430,820,1400,2500] #2500 better for filtering
colors = ['m','r','g','k','b']
def getBand(freq,bands=BANDS):
    diffs=map(lambda f: abs(f-freq),bands)
    minindex=diffs.index(min(diffs))
    return bands[minindex]



class PulsarData:
    def __init__(self,pulsar,mode=None,snrcut=3,residcut=50,setbands=None,eta=0.2,maindir=""):
        self.pulsar = pulsar
        self.mode = mode # for main/interpulse timing
        self.snrcut = snrcut
        self.residcut = residcut
        self.maindir = maindir
        try:
            retval = psrcat.psrcat(pulsar,["P0","DM"])
            self.P = float(retval["P0"])
            self.DM = float(retval["DM"])
        except:
            p = par.Par("/home/michael/Research/Data/NANOGravData/nanograv_timing_2016/release/par/%s_NANOGrav_11yv0.gls.par"%pulsar,numwrap=float)
            self.P = p.getPeriod()
            self.DM = p.getDM()

        self.load()
        self.getWBmodels()
        self.weffs = self.getWeffs()

        #self.bands = np.array(map(getBand,self.freqs))
        if setbands is None:
            setbands = BANDS
        self.bands = np.array(map(lambda x: getBand(x,bands=setbands),self.freqs))



        self.getDISSparameters()

        #self.taudo = 0.028  #manual input for now from optimal freq
        #self.taudo = 0.0
        self.tauds = 1e-3*DISS.scale_taud(self.taud0,1000.0,self.freqs) # in us

        self.niss = (1 + 0.2*self.tobs/DISS.scale_dtd(self.dtd0,1000.0,self.freqs))*(1 + 0.2*self.bws/DISS.scale_dnud(self.dnud0,1000.0,self.freqs))

    

        
    def load(self):
        if self.mode is None:
            npz = np.load("residuals/%s.npz"%self.pulsar)
        else:
            npz = np.load("residuals/%s-%s.npz"%(self.pulsar,self.mode))
        inds = np.where((npz['snrs']>self.snrcut)&(np.abs(npz['resids'])<self.residcut))[0]
        for arr in ['resids','freqs','errs','snrs','tobs','bws']:
            exec("self.%s = npz[\"%s\"][inds]"%(arr,arr))



    def getWBmodels(self):
        filenames = sorted(glob.glob("%sWBmodels/%s*npz"%(self.maindir,self.pulsar)))
        self.funcs = []
        self.WBfilenames = []
        for i,filename in enumerate(filenames):
            npz = np.load(filename)
            self.WBfilenames.append(filename)
            freqs = npz['freqs']
            bins = npz['bins']
            model = npz['model']
            func = interpolate.interp2d(freqs,bins,np.transpose(model),kind='linear')
            self.funcs.append(func) #do this for now
        self.bins = bins #these will all be the same




    def getDISSparameters(self,filename="DISS_parameters.dat"):
        psr = np.loadtxt("%s%s"%(self.maindir,filename),unpack=True,usecols=(0,),dtype=np.str)
        dtd0,dnud0,taud0 = np.loadtxt("%s%s"%(self.maindir,filename),unpack=True,usecols=(1,2,3))
        ind = np.where(self.pulsar==psr)[0][0]
        self.dtd0 = dtd0[ind] #s
        self.dnud0 = dnud0[ind] #MHz
        self.taud0 = taud0[ind] #ns
        
    def getWeff(self,freq): #get Weff at a certain frequency, thus how the models are gotten under the hood will not matter
        
        for i,func in enumerate(self.funcs):
            if min(func.x) <= freq <= max(func.x):
                temp = func(freq,self.bins)

                U = u.normalize(temp,simple=True) #remove baseline?
        
                tot = np.sum(np.power(U[1:]-U[:-1],2))
                return 1e6*self.P/np.sqrt(len(self.bins)*tot) # in us
                
                #plot(self.bins,temp)
                #show()
                #raise SystemExit
        
    #def return_args(self):
    #    return (self.P,self.Weff,self.resids,self.freqs,self.errs,self.snrs,self.tobs)
    
    def getWeffs(self):
        retval = np.zeros_like(self.freqs)
        # This is slow but works
        for i,freq in enumerate(self.freqs):
            retval[i] = self.getWeff(freq)

        # Check for np.nan and interpolate over
        inds = np.where(~np.isnan(retval))[0]

        x = np.array(sorted(list(set(self.freqs[inds]))))
        y = np.zeros_like(x)

        for i,elem in enumerate(x):
            y[i] = self.getWeff(elem)

        wefffunc = u.extrap1d([x,y])

        #testfreqs = np.linspace(400,2000.0,1000)
        #plot(testfreqs,wefffunc(testfreqs),'k.')
        #show()
        #raise SystemExit

        inds = np.where(np.isnan(retval))[0]
        for ind in inds:
            retval[ind] = wefffunc(self.freqs[ind])

        return retval

    def getNp(self):
        return self.tobs/self.P

    def getfac(self,T=1800):
        return np.sqrt(T/self.P)


    def plot(self,filename=None,doshow=True):
        for i,band in enumerate(BANDS):
            inds = np.where(self.bands==band)[0]
            plot(self.snrs[inds],self.resids[inds],'%s.'%colors[i])
        xscale('log')
        xlabel('S/N')
        ylabel('Residual (us)')
        xlim(3,None)
        if filename is not None:
            savefig(filename)
        if doshow:
            show()
        close()

    def frequencyplot(self,snrcut=0.0,filename=None,doshow=True):
        cutinds = np.where(self.snrs>snrcut)[0]
        
        for i,band in enumerate(BANDS):
            inds = np.where(self.bands[cutinds]==band)[0]
            plot(self.freqs[cutinds][inds],self.resids[cutinds][inds],'%s.'%colors[i])
        #xscale('log')
        xlabel('Frequency (MHz)')
        ylabel('Residual (us)')
        #xlim(3,None)
        if filename is not None:
            savefig(filename)
        if doshow:
            show()
        close()

    def get_prior_sigC(self,num=-100):
        '''
        From the data, estimate what sigma_constant is by taking the RMS of the last few data points
        '''
        sortinds = np.argsort(self.snrs)
        return u.RMS(self.resids[sortinds][num:])

    def sigma_clip(self,sigma=6):
        rms = self.get_prior_sigC()
        sigC = np.sqrt(jm.sigma_SN(self.weffs,self.snrs)**2 + rms**2)
        inds = np.where(np.abs(self.resids)<=sigma*sigC)[0]
        for arr in ['resids','freqs','errs','snrs','tobs','bws','weffs','bands','tauds','niss']:
            exec("self.%s = self.%s[inds]"%(arr,arr))

        

        
if __name__ == '__main__':
    psr = sys.argv[1]

    pd = PulsarData(psr)#,)
    pd.plot()

    
    #pd = PulsarData("J1909-3744")
    #print pd.getWeff(1400.0)
