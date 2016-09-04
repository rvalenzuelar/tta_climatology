# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 11:00:36 2016

@author: raulv

check symetrical_sigmoidean model 
in https://mycurvefit.com/index.html

"""
import matplotlib.pyplot as plt
import lmfit
import numpy as np


def symmetrical_sig(x,la,gr,ce,ua):
    '''
        la: lower asymptote
        ua: upper asymptote
        gr: growth rate
        ce: center of symmetry
    '''
    return ua + (la-ua)/(1 + (x/ce)**gr)
    
def curv_fit(x=None,y=None,model=None):
    
    x = np.array(x)    
    y = np.array(y)    
    
    if model == 'gaussian':
        mod  = lmfit.models.GaussianModel()
        params = mod.guess(y, x=x)
        out  = mod.fit(y,params, x=x)
        R_sq = 1 - out.residual.var()/np.var(y)
        
    elif model == 'symmetrical':
        mod = lmfit.Model(symmetrical_sig)
        out = mod.fit(y,x=x,la=1,gr=1,ce=1,ua=1)
        R_sq = 1 - out.residual.var()/np.var(y)      
    
    out.R_sq = R_sq
    return out

def example():

    x = np.array(range(90,280,10))
    
    results = {}
    results['Surf-500m']={
                    'TTczd':[0.88,0.85,1.47,1.57,1.48,1.87,2.66,
                             3.27,3.71,3.83,3.43,2.97,3.07,2.67,
                             2.29,1.55,1.24,1.23,1.07],
                    'TTbby':[0.77,0.56,0.87,1.18,1.15,1.22,1.13,
                             1.04,0.81,1.12,0.91,0.87,0.94,0.84,
                             0.77,0.54,0.45,0.35,0.39],
                    'ratio':[1.14,1.52,1.68,1.33,1.29,1.53,2.35,
                             3.13,4.56,3.41,3.79,3.42,3.25,3.17,
                             2.96,2.89,2.76,3.53,2.72]                         
                    }
    
    y = results['Surf-500m']['ratio'] 
    out = curv_fit(x=x,y=y,model='symmetrical')
    xnew = np.array(range(90,280,1))
    ynew = out.eval(x=xnew)
    plt.plot(x, y,'go')                       
    plt.plot(xnew, ynew, 'g-')
    
    
    
    y = np.array(results['Surf-500m']['TTczd'])   
    out = curv_fit(x=x,y=y,model='gaussian')
    xnew = np.array(range(90,280,1))
    ynew = out.eval(x=xnew)
    plt.plot(x, y,'bo')
    plt.plot(xnew, ynew, 'b-')
    plt.ylim([0,6])
    
    
    y = np.array(results['Surf-500m']['TTbby'])   
    out = curv_fit(x=x,y=y,model='gaussian')
    xnew = np.array(range(90,280,1))
    ynew = out.eval(x=xnew)
    plt.plot(x, y,'ro')
    plt.plot(xnew, ynew, 'r-')
    plt.ylim([0,6])

