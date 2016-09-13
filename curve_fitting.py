# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 11:00:36 2016

@author: raulv

check 4- and 5-parameter logistic model 
in https://mycurvefit.com/index.html

Also in:
Gottschalk and Dunn - 2005:
The five-parameter logistic

"""
import matplotlib.pyplot as plt
import lmfit
import numpy as np


def logistic_4p(x, la, gr, ce, ua):
    '''
        la: lower asymptote
        ua: upper asymptote
        gr: growth rate
        ce: center of symmetry
    '''
    return ua + (la-ua)/(1 + (x/ce)**gr)


def logistic_5p(x, la, gr, ce, ua, sy):
    '''
        la: lower asymptote
        ua: upper asymptote
        gr: growth rate
        ce: center
        sy: symmetry
    '''
    return ua + (la-ua)/(1 + (x/ce)**gr)**sy


def curv_fit(x=None, y=None, model=None):
    
    x = np.array(x)    
    y = np.array(y)    
    params = lmfit.Parameters()
    
    if model == 'gaussian':
        mod = lmfit.models.GaussianModel()
        params = mod.guess(y, x=x)
        out = mod.fit(y,params, x=x)
        r_sq = 1 - out.residual.var()/np.var(y)
        
    elif model == '4PL':
        mod = lmfit.Model(logistic_4p)
        params.add('la', value=1.0)
        params.add('gr', value=120.0, vary=False)
        params.add('ce', value=150.0)
        params.add('ua', value=3.0)
        out = mod.fit(y, params,x=x)
        r_sq = 1 - out.residual.var()/np.var(y)
        
    elif model == '5PL':
        mod = lmfit.Model(logistic_5p)
        params.add('la', value=1.0)
        params.add('gr', value=1.0)
        params.add('ce', value=1.0)
        params.add('ua', value=1.0)
        params.add('sy', value=1.0)
        out = mod.fit(y, params, x=x)
        r_sq = 1 - out.residual.var()/np.var(y)
    
    out.R_sq = r_sq
    return out

def get_params_err(out):

    param_err_bot=dict()
    param_err_top=dict()
    for par in out.params:
        value = out.params[par].value
        stder = out.params[par].stderr
        param_err_bot[par] = value - stder
        param_err_top[par] = value + stder
        
    return param_err_bot,param_err_top

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

    xnew = np.array(range(90,280,1))
  
    fig,ax = plt.subplots(2,1,figsize=(6,8))
    
    y = np.array(results['Surf-500m']['TTczd'])   
    out = curv_fit(x=x,y=y,model='gaussian')
    ynew = out.eval(x=xnew)
    ax[0].plot(x, y,'bo')
    ax[0].plot(xnew, ynew, 'b-')
    ax[0].set_ylim([0,6])
    
    y = np.array(results['Surf-500m']['TTbby'])   
    out = curv_fit(x=x,y=y,model='gaussian')
    ynew = out.eval(x=xnew)
    ax[0].plot(x, y,'ro')
    ax[0].plot(xnew, ynew, 'r-')
    ax[0].set_ylim([0,6])

    y = results['Surf-500m']['ratio'] 
    out = curv_fit(x=x,y=y,model='4PL')
    ynew = out.eval(x=xnew)
    ax[1].plot(x, y,'go')                       
    ax[1].plot(xnew, ynew, 'g-')
    
#    be,te = get_params_err(out)
#    yb = logistic_4P(xnew,be['la'],48+72,be['ce'],be['ua'])
#    yt = logistic_4P(xnew,te['la'],48+72,te['ce'],te['ua'])
#    ax[1].plot(xnew, yb, '--')
#    ax[1].plot(xnew, yt, '--')
        
    ''' add annotations '''
    ypos = 0.5
    for par in out.params:
        tx = '{}: {:2.1f}'.format(par,out.params[par].value)
        ax[1].text(0.1,ypos,tx,fontsize=15,
                    transform=ax[1].transAxes)
        ypos += 0.08
    tx = 'R-sq: {:2.2f}'.format(out.R_sq)
    ax[1].text(0.1,ypos,tx,fontsize=15,
                    transform=ax[1].transAxes)
                    
    ax[1].set_ylim([0,6])
    print(out.fit_report())
    print('R-sq: {}'.format(out.R_sq))
