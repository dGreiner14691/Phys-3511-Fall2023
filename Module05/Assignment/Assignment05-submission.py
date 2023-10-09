#!/usr/bin/env python
# coding: utf-8

# # Assignment 5 Due: Thursday 10/5

# In this assignment you will explore fitting data and assessing how well your fit describes the different data sets.
# 
# Assignment Overview:
# * Fit data and use $\chi^2$ and the $\chi^2$ test to assess 
# * Analyze the efficiency of your data provided different threshold levels using your fit results 
# 
# For this assingment you can make use of the numpy, matplotlib, and the scipy packages.

# In[1]:


import scipy as sp
import numpy as np
import matplotlib.pyplot as plt


# # Problem 1: W Boson Mass
# 
# Finding the *true* values of a quantity relies on analyzing results for many experiments. One quantity that has been measured many times is the W boson mass see Wikipedia https://en.wikipedia.org/wiki/W_and_Z_bosons and the particle data group (PDG) https://pdg.lbl.gov/2018/listings/rpp2018-list-w-boson.pdf 
# 
# **a)** In this problem you will analyze measurements of the W boson from various experiments and determine if the values are consistnet and given this data set, what the best fit value is. Start by reading in the data file Wmass_data.txt, which contains an experiment number, W mass in units of $GeV/c^2$ and its uncertainty.
# 

# In[22]:


expN, Wmass, unc = np.loadtxt("Wmass_data.txt", unpack=True)


# **b)** Compute the error weighted mean of the W mass and its uncertainty. How does the weighted mean compare to the bold faced average of the PDG?

# In[26]:


weightedMean = np.sum(Wmass/(unc**2))/(np.sum(1/unc**2))
uncertainty = np.sqrt(1/np.sum(1/(unc**2)))
print(weightedMean,uncertainty)
print('Value is very similar to the groups given value of 80.379 +- 0.012.')


# **c)** Calculate the $\chi^2$, degrees of freedom, reduced $\chi^2$, and p-value. The p-value can be calculated using *gammaincc(dof / 2.0, chisq / 2.0)* from *scipy.special*. Based on the p-value are the data consistant?

# In[14]:


chi2 = np.sum(((Wmass - weightedMean)/unc)**2)
free = len(Wmass) -1
redChi2 = chi2/free
p = sp.special.gammaincc(free/2,chi2/2)
print(chi2,free,redChi2,p)


# **d)** Plot the measurement number vs. the W mass. Don't forget to include the error bars on the W mass measurements. Then Fit a line of the form $y = p_0$, where $p_0$ is a constant parameter.
# 
# How does your $p_0$ value compare to the weighted mean you calculated earlier in part b)?

# In[58]:


def pofunc(x, p_0):
    return p_0 * np.ones( len(x) )

fig = plt.figure()
axes = fig.add_axes([0.15,0.1,0.8,0.8])

po, pc = sp.optimize.curve_fit(pofunc, expN, Wmass, sigma=unc)

axes.errorbar(expN, Wmass, yerr = unc, fmt='o')

plt.plot(expN, pofunc(expN, *po), 'r-',)

axes.set_xlabel('Experiment')
axes.set_ylabel('W_mass');


# # Problem 2: Proton Charge Radius
# 
# We will carry an identical analysis as we did in Problem 1, but on a different quantity, the proton charge radius. The proton charge radius has been a recent hot topic in the nuclear physics field, as new designed experiments using muonic hydorgen have made very percise measurements of it. See https://www.nature.com/articles/s41586-019-1721-2
# 
# There is an approchable video that reviews the history of the proton size and its measurements: https://www.youtube.com/watch?v=C5B_ZfGy4d0
# 
# **a)** Import the data set proton_radius_data.txt, which includes the experiment number, the proton charge radius, and its uncertainty measured in $fm$. 

# In[24]:


pExpN, pRad, pUnc = np.loadtxt("proton_radius_data.txt", unpack=True)


# **b)** Compute the error weighted mean of the proton charge radius and its uncertainty. 
# 
# You can also compare this to the PDG value (pgs. 6 and 7): https://pdg.lbl.gov/2018/listings/rpp2018-list-p.pdf 

# In[32]:


pWeightedMean = np.sum(pRad/(pUnc**2)) / np.sum(1/(pUnc**2))
pUncertainty = np.sqrt(1/np.sum(1/(pUnc**2)))


# **c)** Calculate the  $\chi^2$, degrees of freedom, reduced $\chi^2$ and p-value. Based on the p-value are the data consistant? Do you see what all of the fuss is about.

# In[35]:


pChi2 = np.sum(((pRad - pWeightedMean)/pUnc)**2)
pFree = len(pRad) -1
pRedChi2 = pChi2/pFree
pP = sp.special.gammaincc(pFree/2,pChi2/2)
print(pChi2,pFree,pRedChi2,pP)
print('P-value looks to be floating point error, therefore 0 which means data is inconsistent')


# **d)** Plot the measurement number vs. the proton charge radius. Don't forget to include the error bars on the proton charge radius measurements. Then Fit a line of the form  $y = p_0$ , where $p_0$ is a constant parameter.
# 
# How does your $p_0$ value compare to the weighted mean you calculated earlier in part b)?

# In[68]:


fig = plt.figure()
axes = fig.add_axes([0.15,0.1,0.8,0.8])

ppo, ppc = sp.optimize.curve_fit(pofunc,pExpN,pRad,sigma=pUnc)

axes.errorbar(pExpN, pRad, yerr = pUnc, fmt='o')
plt.plot(pExpN, const_func(pExpN, *ppo))

plt.xlabel("Experiment")
plt.ylabel("Proton Charge Radius")


# # Problem 3: Selecting Data
# 
# In particle physics we sometimes want to measure a particlular particle that is created from many which result from a collision in a particle collider. In recording these collision events we typically measure other particles which are not the ones we are intersted in. The events we are interested in we refer to as our signal, whereas the ones we are not interested in we refer to as a background. 
# 
# **a)** The provided data set (Ep_data.txt) contains values of particle energy/momentum (E/p), the number of particles, and the uncertainty on the number of particles. Import the data and plot the number of particles vs. E/p and be sure to include the error bars on the particle counts. 

# In[39]:


eVal, eN , eUnc = np.loadtxt("Ep_data.txt", unpack=True)


# **b)** You should notice that there appear to be two clear distributions here. One which seems to be centered E/p = 0.6 and another around E/p = 1. The population at the lower E/p represent pions, whereas the population around E/p = 1 are electrons. For this exersice we will treat the pions as a background and the electrons as our signal. We will model each particle type to have a Gaussian distribution. Define two python functions, one that returns a value computed from a Gaussian functions, and another python function that returns a value computed from the sum of two Gaussian functions. Then make a fit to the data using the sum of two Gaussian functions. Each of your Gaussian functions can take the form of:
# 
# $G_1(x) = p_1 e^{-(x-p_2)^2/(2p_3)}$
# 
# where the $p_1, p_2,$ and $p_3$ are three parameters for the one Gaussian function. You will have 3 more different parameters for the other Gaussian function $G_2(x)$. So we want to fit our E/p distribution with function $G_1(x) + G_2(x)$. The image below shows my fit, with the $G_1(x) + G_2(x)$ fit being the black curve. From this fit I can use the fit parameters to draw $G_1(x)$ (blue curve) and $G_2(x)$ (red curve). 
# 
# Note: Did you get a negative value for the gaussian widths from your fit? We know that a negative value is not physical. Try to give some initial parameters for the fit to start with.
# 
# ![Screen%20Shot%202021-07-15%20at%209.57.45%20AM.png](attachment:Screen%20Shot%202021-07-15%20at%209.57.45%20AM.png)

# In[73]:


def gauss(x, p1, p2, p3):
    return p1*np.exp(-((x-p2)**2)/(2*p3))

def gauss2(x, p1, p2, p3, p4, p5, p6):
    return gauss(x, p1, p2, p3) + gauss(x, p4, p5, p6)

par, cov = sp.optimize.curve_fit(gauss2,eVal,eN,sigma=eUnc, absolute_sigma=True)

pFit = gauss(eVal, *par[0:3])
eFit = gauss(eVal, *par[3:6])
tFit = pFit + eFit

par


# **c)** Calculate your $\chi^2$, degrees of freedom, reduced $\chi^2$, and p-value for the fit to the data.
# Based on those statistics above is this a good fit? Explain.

# In[77]:


eChi2 = np.sum((eN - tFit)**2/eUnc**2)
eFree = len(eN) - len(par)
eRedChi2 = eChi2/eFree
eP = sp.special.gammaincc(eFree/2,eChi2/2)
print(eChi2,eFree,eRedChi2,eP)


# **d)** On the same graph, plot your data, the total fit to it, and the single Gaussian functions computed using the parameter results from your 2 Gaussian function fit (e.g. reproduce my fit figure). 

# In[97]:


fig = plt.figure()
axes = fig.add_axes([0.15,0.1,0.8,0.8])

axes.errorbar(eVal,eN,yerr = eUnc, fmt='+',label = 'data')
axes.plot(eVal, eFit,'r--',label = 'electron fit')
axes.plot(eVal, pFit,'g--',label = 'pion fit')
axes.plot(eVal, tFit,'k-',label = 'total fit')

axes.set_xlabel('energy/momentum')
axes.set_ylabel('Particle Number')
axes.legend()


# **e)** We can use the $E/p$ distribution to try to select the maximum number of electrons while minimizing the number of pions that *leak* into our electron signal. We can do this by requireing our selected sample to be larger than some $E/p$ threshold value. Any data that has an $E/p$ value lower then the threshold we throw it out. In a physics analysis this is called a cut. However we need to be careful, if we place a cut at $E/p$ that is too large we will have a really clean electron sample, but throw away a lot of good electrons. On the other hand if we make the $E/p$ cut too low we will keep most of our electrons, but let in a lot of background (pions). So we must compormise between clean data and statistics. To do this lets calculat the total number of electrons we have from $0.0 < E/p < 2$. This can be obtained by integrating (you can use scipy integrators, I used *integrate.quad* when doing this exersise)the electron contribution from our fit. We will call this number e_tot. Do a similar thing for the total pions and call that number pi_tot. 
# 
# For 10 equally spaced E/p thresholds between 0.3 and 0.8, calculate the number of electrons that are above each of the thresholds, we can call this array e_sig and can be obtained by integrating from the E/p threshold value to the E/p = 2. Do a similar thing for the pion distribution. 
# 
# Below is the your graph in part f) should look like.
# 
# ![Screen%20Shot%202021-07-15%20at%209.57.52%20AM.png](attachment:Screen%20Shot%202021-07-15%20at%209.57.52%20AM.png)
# 
# 

# In[110]:


list = np.linspace(0.3,0.8,10)
pionA = np.array([])
electronA = np.array([])
pi_tot = sp.integrate.quad(gauss,0,2,args =(par[0], par[1], par[2]))[0]
e_tot = sp.integrate.quad(gauss,0,2,args =( par[3], par[4], par[5]))[0]
for i in list:
    pion = sp.integrate.quad(gauss,i,2,args =(par[0], par[1], par[2]))[0]
    pionA = np.append(pionA,pion)
    electron = sp.integrate.quad(gauss,0,2,args =( par[3], par[4], par[5]))[0]
    electronA = np.append(electronA,electron)


# **f)** Plot the ratios e_sig/e_tot and pi_sig/pi_tot as a function of E/p threshold on the same graph. 

# In[114]:


pionR = 100*pionA/pi_tot
electronR = 100*electronA/e_tot
fig = plt.figure()
axes = fig.add_axes([0.15,0.1,0.8,0.8])
axes.scatter(list,pionR,label = 'pion', c='r')
axes.scatter(list,electronR,label = 'electron', c='b')
axes.legend()


# **g)** When the e_sig/etot ratio is 90%, how what percentage of the pion distribution is contaminating our electron sample?

# In[ ]:




