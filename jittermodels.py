'''
This methodology is used because the emcee sampler cannot be embedded into classes if multiprocessing is used (threads>1).
'''

import numpy as np
from collections import namedtuple


JitterModel = namedtuple('JitterModel','lnprior lnlike lnprob pinit')


MAX = 1000 #us

Nphi = 2048
def sigma_SN(Weff,S):
    return Weff / (np.sqrt(Nphi)*S)

def sigma_J(sigmaJ1,P,tobs): 
    Np = tobs/P
    return sigmaJ1/np.sqrt(Np)

def sigma_DISS(taud,niss=1):
    return taud/np.sqrt(niss)



def get_lnL(pd,sig_SN,sig_J,sig_DISS,weight=False):

    V_tot = sig_SN**2 + sig_J**2 + sig_DISS**2
    sig_tot = np.sqrt(V_tot)
    
    N = len(pd.resids)
    if weight:
        lnL = 0
        weights = pd.snrs / np.sum(pd.snrs)
        for i in range(N):
            x = -(1.0/2)*np.log(2*np.pi) - np.log(sig_tot[i]) - 0.5*np.power(pd.resids[i]/sig_tot[i],2) 
            lnL += (weights[i]*x)
    else:
        lnL = -(N/2)*np.log(2*np.pi) - np.log(sig_tot).sum() - 0.5*np.power(pd.resids/sig_tot,2).sum() #can speed up the square root of squaring
    return lnL


# ============================================================
# Constant Model
# ============================================================

def constant_lnprior(theta):
    sigmaJ1 = theta
    if 0 < sigmaJ1 < MAX:
        return 0.0
    return -np.inf


def constant_lnlike(theta,pd,weight=False):
    sigmaJ1 = theta

    sig_SN = sigma_SN(pd.weffs,pd.snrs)
    sig_J = sigma_J(sigmaJ1,pd.P,pd.tobs)
    sig_DISS = sigma_DISS(pd.tauds,pd.niss)
    
    return get_lnL(pd,sig_SN,sig_J,sig_DISS,weight=weight)




def constant_lnprob(theta,pd,weight=False):
    lp = constant_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + constant_lnlike(theta,pd,weight=weight)



constant = JitterModel(lnprior=constant_lnprior,lnlike=constant_lnlike,lnprob=constant_lnprob,pinit=np.array([10.0]))





# ============================================================
# N-Band Constant Model
# ============================================================

def nband_lnprior(theta):
    for elem in theta:
        if (elem <= 0) or (elem >= MAX):
            return -np.inf
    return 0.0


def nband_lnlike(theta,pd,bands,weight=False):
    #sigmaJ1a,sigmaJ1b = theta

    sig_SN = sigma_SN(pd.weffs,pd.snrs)
    sig_J = np.zeros_like(pd.resids)
    for i,b in enumerate(bands): # this will be super slow?
        inds = np.where(pd.bands==b)[0]
        sig_J[inds] = sigma_J(theta[i],pd.P,pd.tobs[inds])

    sig_DISS = sigma_DISS(pd.tauds,pd.niss)

    return get_lnL(pd,sig_SN,sig_J,sig_DISS,weight=weight)
    



def nband_lnprob(theta,pd,bands,weight=False):
    lp = nband_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + nband_lnlike(theta,pd,bands,weight=weight)



nband = JitterModel(lnprior=nband_lnprior,lnlike=nband_lnlike,lnprob=nband_lnprob,pinit=None)








# ============================================================
# Power-Law Model
# ============================================================

def powerlaw_lnprior(theta):
    sigmaJ1,alpha = theta
    if 0 < sigmaJ1 < MAX and -10 < alpha < 10: #us
        return 0.0
    return -np.inf



def powerlaw_lnlike(theta,pd,weight=False):
    sigmaJ1,alpha = theta

    sig_SN = sigma_SN(pd.weffs,pd.snrs)
    sig_J = sigma_J(sigmaJ1,pd.P,pd.tobs) * np.power(pd.freqs/1000.0,alpha)

    sig_DISS = sigma_DISS(pd.tauds,pd.niss)

    return get_lnL(pd,sig_SN,sig_J,sig_DISS,weight=weight)




def powerlaw_lnprob(theta,pd,weight=False):
    lp = powerlaw_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + powerlaw_lnlike(theta,pd,weight=weight)



powerlaw = JitterModel(lnprior=powerlaw_lnprior,lnlike=powerlaw_lnlike,lnprob=powerlaw_lnprob,pinit=np.array([10.0,1.0]))






# ============================================================
# Log-Polynomial Model
# ============================================================

def logpolynomial_lnprior(theta):
    #total = 0
    #for i,elem in enumerate(theta):
        
    for elem in theta:
        if (elem <= -MAX) or (elem >= MAX): #?
            return -np.inf
    return 0.0


def logpolynomial_lnlike(theta,pd,weight=False):
    #sigmaJ1a,sigmaJ1b = theta
    
    
    sig_SN = sigma_SN(pd.weffs,pd.snrs)
    sig_J = np.zeros_like(sig_SN)
    for i,elem in enumerate(theta):
        sig_J += sigma_J(elem,pd.P,pd.tobs) * np.power(np.log10(pd.freqs/1000.0),i)


    sig_DISS = sigma_DISS(pd.tauds,pd.niss)

    return get_lnL(pd,sig_SN,sig_J,sig_DISS,weight=weight)


def logpolynomial_lnprob(theta,pd,weight=False):
    lp = logpolynomial_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + logpolynomial_lnlike(theta,pd,weight=weight)



logpolynomial = JitterModel(lnprior=logpolynomial_lnprior,lnlike=logpolynomial_lnlike,lnprob=logpolynomial_lnprob,pinit=None)


