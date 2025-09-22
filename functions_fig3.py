# -*- coding: utf-8 -*-
"""
Created on Thu May  8 13:36:37 2025

@author: Volll001
"""
import pandas as pd
from io import StringIO
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

NA = 6.02214076*1e23
m = 0.3 #[g]
M0 = 28 #M0: monomer molecular weight PP g/mol
fac = 100

def readin(file, t_min, t_max):
    SEC = readin_simple(file, t_min, t_max)
    new_idx = SEC.index/M0/fac
    SEC.index = new_idx
    max_Mw = t_max/M0/fac
    xa = pd.Index(np.arange(1, max_Mw, 1))
    idx2 = xa.append(SEC.index).sort_values()
    idx2 = np.unique(np.array(idx2))
    SEC = SEC.reindex(idx2).interpolate(method='values')
    xa = pd.Index(np.arange(1, max_Mw, 1))
    SEC = SEC[xa]
    SEC = SEC.fillna(0)
    SEC = np.array(SEC)    
    SEC -= min(SEC)
    SEC /= sum(SEC)
    return SEC, xa, max_Mw

def readin_simple(file, t_min, t_max):
    with open(file) as full:
        data = full.read().split('\t\nMWDstop :\n\n')[0]
        data = data.split(' :\nMolar mass\tIR-5 CH2\tIntegral[%]\n\n ')[-1]
    names = ['Molar mass [g/mol]', 'intensity', 'integral']
    frame = pd.read_table(StringIO(data), sep='\t ', header=None,
                          names=names, engine='python')
    frame.index =frame['Molar mass [g/mol]']
    frame = frame.drop(['integral','Molar mass [g/mol]'], axis=1)
    SEC = pd.Series(frame['intensity']/frame.index)
    SEC = SEC[(SEC.index>t_min) & (SEC.index<t_max)]   
    SEC -= min(SEC)
    SEC /= SEC.sum()
    return SEC


def scissions(ydata, xa, chains0):
    Mi = xa*M0*fac
    ni = ydata/Mi
    Mn = sum(Mi*ni)/sum(ni)
    Mw = sum(ni*Mi**2)/sum(Mi*ni)
    PDI = Mw/Mn
    sciss = m/(Mn)*NA-chains0
    return sciss, Mn, Mw, PDI
#%%
"""definition of functions"""
#P_fun: P(t,x); Probability that a polymer with a DP of x will undergo scission after t scission events
def P_fun(x, M0, fn, s):
    #M0: monomer molecular weight
    #fn: fn(x,t); Number fraction for a polymer with a DP of x after t scission events 2-D array
    #s: power constant to be fitted double
    #A: A(t); normalization coefficient at time t
    #get x from fn array
    P1 = (x*M0)**s*(fn) # a function of x
    #determine A so that sum(P)=1
    if sum(P1)>0:
        A=1/sum(P1)
        P1 = P1*A
    return P1

#Q_fun: Q(y,t); Probability that a polymer with a DP of y will produce a polymer
# with a DP of x upon scission
# this calculates a Gaussian function
def Q_fun(r, x, y):
    #y: degree of polymerization x plus 1
    #r: Gaussian parameter
    Q = (1/(r*y*np.sqrt(2*np.pi)))*np.exp(-(x-y/2)**2/(2*r**2*y**2))
    return Q
    
#fn
def fn_fun(xa, fn, N, t, r, M0, s):
    #t: number of cleavage events, i.e. iterations
    P = P_fun(xa, M0, fn, s)
    counter = []
    fn1 = []
    for x, i in zip(xa, np.arange(0, len(xa), 1)):
        ya = xa[i:] + 1
        Qya = Q_fun(r, x, ya)
        Pya = P_fun(ya, M0, fn[i:], s)            
        counter.append(sum(Pya*Qya))
    counter = np.array(counter)
    fn1 = ((N+t)*fn - P + 2*counter)/(N+t+1)
    return fn1
#%%  

def fit_fun(fn0, tmax, chains0, max_Mw, scissys, N, fig_path, Mn0, ydatas, t_min, t_max, times):
    max_Mw = int(max_Mw)
    def fitty(xas, *vary):
        NA = 6.02214076*1e23        
        xa = xas[:max_Mw]
        m = 0.3
        r = vary[0]
        s = vary[1]
        mlist = []
        scissions = [0]
        Mns = [Mn0]

        for t in np.arange(0, tmax, 1):
            if t == 0:
                fn = fn_fun(xa, fn0, N, t, r, M0, s)
                Mi = xa*M0*fac
                ni = fn/Mi
                Mn = sum(Mi*ni)/sum(ni)
                Mns.append(Mn)
                scissions.append((m/Mn)*NA - chains0)
                mlist.append(np.array(fn/sum(fn)))
    
            else:
                fn = fn_fun(xa, fn, N, t, r, M0, s)
                Mi = xa*M0*fac
                ni = fn/Mi
                Mn = sum(Mi*ni)/sum(ni)
                Mns.append(Mn)
                scissions.append((m/Mn)*NA - chains0)
                mlist.append(np.array(fn/sum(fn)))
                    
        minimas = [0]
        for i in range(0, int(len(ydatas)/max_Mw)):
            diff = [sum(abs(m-ydatas[i*max_Mw:(i+1)*max_Mw])) for m in mlist]
            minimas.append(np.array(diff).argmin())
        mlist = np.array(mlist)[minimas[1:]]    
        print(minimas)

        mpl.rcParams['font.sans-serif'] = 'Arial'
        mpl.rcParams['font.size'] = 9
        mpl.rcParams['lines.linewidth'] = 1
        fig = plt.figure(layout="constrained", figsize = (3.5, 7))

        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :])
        ax3 = ax2.twinx()
        ax1.annotate('a)', xy=(-0.25, 1), xycoords='axes fraction'
                     , verticalalignment='top', fontsize=11)
        ax2.annotate('b)', xy=(-0.25, 1), xycoords='axes fraction'
                     , verticalalignment='top', fontsize=11)
        colors = plt.cm.magma(np.linspace(0, 1, len(mlist)+2))

        j = 0
        for m in mlist:
            ax1.plot(xa*M0*fac/1E3, m*1e2, color=colors[j+1])
            j+=1
        
        ax1.plot(xa*M0*fac/1E3, fn0*1e2, 'k--', label='0.0 h')
        Mns_ex = [Mn0]
        for i in range(0, int(len(ydatas)/max_Mw)):
            ydat = ydatas[i*max_Mw:(i+1)*max_Mw]
            Mi = xa*M0*fac
            ni = ydat/Mi
            Mn = sum(Mi*ni)/sum(ni)
            Mns_ex.append(Mn)
            ax1.plot(Mi/1E3, ydat*1e2, marker='*', markersize=3, linestyle='',
                     color=colors[i+1], 
                     label='{} h'.format(times[i+1]))        
        
        ax1.set_xlabel(r'$\bf{molar\ mass}\ \it{(kg/mol)}$')
        ax1.set_ylabel(r'$\bf{number\ fraction}\ \it{(*10^{2})}$')
        #ax1.title.set_text(fig_path.split('/')[-1]+'r:{:1.2f}, '.format(r)+'s:{:1.2f}, '.format(s)+'N:{}'.format(N))
        ax1.legend(frameon=False)
        ax1.set_ylim(0, )
        ax1.set_xlim(t_min, 3.5e5/1E3)
        
        ax2.plot(times, np.array(scissions)[minimas]/1e18, 'r--', label='fit')
        ax2.plot(times, np.array(scissys)/1e18, marker="o", linestyle=' ', color='k', label='experimental data')
        ax2.legend(frameon=False)
        ax2.set_xlim(0, )
        ax2.set_ylim(0, )
        ax2.set_xlabel(r'$\bf{time\ }\ \it{(h)}$')
        ax2.set_ylabel(r'$\bf{number\ of\ scissions}\ \it{(10^{18})}$')
        
        ax3.plot(times, np.array(Mns_ex)/1E3, marker="o", linestyle=' ', color='k', label='experimental data')
        ax3.plot(times, np.array(Mns)[minimas]/1E3, 'r--', label='fit')
        ax3.set_xlabel(r'$\bf{time\ }\ \it{(h)}$')
        ax3.set_ylabel(r'$\bf{M_{n}}\ \it{(kg/mol)}$')
        path = fig_path + '.svg'
        
        plt.savefig(path, bbox_inches='tight', transparent=True)
        plt.show()
        return np.array(mlist).flatten()
    return fitty

#Q_fun: Q(y,t); Probability that a polymer with a DP of y will produce a polymer
# with a DP of x upon scission
# this calculates a Gaussian function
def Q_fun1(r, x, y):
    #y: degree of polymerization x plus 1
    #r: Gaussian parameter
    Q1 = (1/(r*np.sqrt(2*np.pi)))*np.exp(-(x-y/2)**2/(2*r**2))
    return Q1
    
#fn
def fn_fun1(xa, fn, N, t, r, M0, s):
    #t: number of cleavage events, i.e. iterations
    P = P_fun(xa, M0, fn, s)
    counter = []
    fn1 = []
    for x, i in zip(xa, np.arange(0, len(xa), 1)):
        ya = xa[i:] + 1
        Qya = Q_fun1(r, x, ya)
        Pya = P_fun(ya, M0, fn[i:], s)            
        counter.append(sum(Pya*Qya))
    counter = np.array(counter)
    fn1 = ((N+t)*fn - P + 2*counter)/(N+t+1)
    return fn1    