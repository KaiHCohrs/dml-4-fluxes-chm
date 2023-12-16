import torch
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import gaussian_kde
import seaborn as sb
import matplotlib as mpl
import matplotlib.pyplot as plt
import torchvision
import pylab

from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn


def mean_diurnal_cycle(data, flux, path = False):
    data['grouped_months'] = data.apply(lambda x: (x['Month']-1) // 2 + 1, axis = 1)
    fig, ax = plt.subplots(2, 6, figsize = (40,10), sharex=True)
    for j, months in enumerate(['Jan-Feb', 'Mar-Apr', 'May-Jun', 'Jul-Aug', 'Sep-Oct', 'Nov-Dec']):
        df_temp = data[(data["grouped_months"] == j+1)]
        
        means = df_temp.groupby(['Time']).mean()
        
        ax[0,j].plot(means[flux + '_orth'], color='red', label='Orth')
        ax[0,j].plot(means[flux + '_NT'], color='black', label='NT')
        ax[0,j].plot(means[flux + '_DT'], color='blue', label='DT')
        ax[0,j].vlines(x = 24,  ymin=-1, ymax=20, color='gray', alpha=0.5)
        
        
        ax[1,j].plot(means[flux + '_orth']-means[flux + '_NT'], color='black', label='Orth-NT')
        ax[1,j].plot(means[flux + '_orth']-means[flux + '_DT'], color='blue', label='Orth-DT')
        ax[1,j].plot(means[flux + '_NT']-means[flux + '_DT'], color='green', label='NT-DT')
        ax[1,j].vlines(x = 24,  ymin=-2.5, ymax=2.5, color='gray', alpha=0.5)
        ax[1,j].hlines(y = 0,  xmin=0, xmax=48, color='gray', alpha=0.5)
        
        if flux == 'GPP':
            ax[0,j].set_ylim((-1,20))
            ax[1,j].set_ylim((-2.5, 2.5))
        elif flux == 'RECO':
            ax[0,j].set_ylim((-1,10))
            ax[1,j].set_ylim((-1.5, 1.5))
        ax[1,j].set_xticks(list(range(0,48,8)))
        ax[1,j].set_xticklabels(list(range(0,24,4)), fontsize = 24)
        ax[1,j].set_xlabel("Hour of the Day", fontsize = 24)
            
        if j == 0:
            ax[0,j].set_ylabel(flux, fontsize = 24)
            ax[1,j].set_ylabel('Diff ' + flux, fontsize = 24)
            
        ax[0,j].set_title(months, fontsize=24)
        
        ax[0,j].yaxis.set_tick_params(labelsize=24)        
        ax[1,j].yaxis.set_tick_params(labelsize=24)
        #handles, labels = ax[i].get_legend_handles_labels()
        ax[0,0].legend(loc="upper left", fontsize=24)
        ax[1,0].legend(loc="lower left", fontsize=24)

        
        #fig.legend(handles, labels, bbox_to_anchor=(0.5,-0.1), loc='lower center', ncol=3, fontsize=24)                       
    fig.suptitle(f'Mean diurnal cycle of {flux}', fontsize=24)
    fig.tight_layout()
    if path:
        fig.savefig(f'{path}/mean_diurnal_cycle_{flux}.png', bbox_inches='tight') 


def seasonal_cycle_plot(data, path=False):
    
    fig, ax = plt.subplots(2, 1, figsize = (10,10))
    for i, flux in enumerate(['GPP', 'RECO']):
        ax[i].plot(data.groupby('Month').mean()[flux + '_orth_seasonal_avg'], color='red', label='Orth')
        ax[i].fill_between(range(1,13), data.groupby('Month').mean()[flux + '_orth_seasonal_avg']-data.groupby('Month').std()[flux + '_orth_seasonal_avg'], data.groupby('Month').mean()[flux + '_orth_seasonal_avg']+data.groupby('Month').std()[flux + '_orth_seasonal_avg'], alpha=0.2, color='red')
        #rather: data.groupby(['Month', 'site', 'Year']).mean().groupby(['Month']).std()[flux + '_orth_seasonal_avg'] ????
        #aand do we work with the cycles with yearly on top?

        ax[i].plot(data.groupby('Month').mean()[flux + '_NT_seasonal_avg'], color='black', label='NT')
        ax[i].fill_between(range(1,13), data.groupby('Month').mean()[flux + '_NT_seasonal_avg']-data.groupby('Month').std()[flux + '_NT_seasonal_avg'], data.groupby('Month').mean()[flux + '_orth_seasonal_avg']+data.groupby('Month').std()[flux + '_orth_seasonal_avg'], alpha=0.2, color='black')

        ax[i].plot(data.groupby('Month').mean()[flux + '_DT_seasonal_avg'], color='blue', label='DT')
        ax[i].fill_between(range(1,13), data.groupby('Month').mean()[flux + '_DT_seasonal_avg']-data.groupby('Month').std()[flux + '_DT_seasonal_avg'], data.groupby('Month').mean()[flux + '_orth_seasonal_avg']+data.groupby('Month').std()[flux + '_orth_seasonal_avg'], alpha=0.2, color='blue')
        
        ax[i].set_xticks(list(range(1,13)))
        ax[i].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize = 24)
        ax[i].set_ylabel(flux, fontsize=24)
        ax[i].yaxis.set_tick_params(labelsize=24)
        handles, labels = ax[i].get_legend_handles_labels()
        
        
        fig.legend(handles, labels, bbox_to_anchor=(0.5,-0.1), loc='lower center', ncol=3, fontsize=24)                       
    fig.suptitle(f'Mean Seasonal Cycle (MSC)', fontsize=24)
    fig.tight_layout()
    if path:
        fig.savefig(f'{path}/seasonal_cycle.png', bbox_inches='tight') 



def cross_consistency_plots(data, path=False):
    fig, ax = plt.subplots(6,4, figsize = (10*4,5*6))
    for i, (cond, form) in enumerate(zip([['site', 'Year', 'doy'],['site', 'Year'], ['site', 'Year', 'soy'], ['site', 'Year', 'doy']], ['daily', 'yearly', 'seasonal_without_yearly', 'daily_anomalies'])):
        for j, (flux_orth, flux_other) in enumerate(zip(['GPP_orth', 'GPP_orth', 'GPP_NT', 'RECO_orth', 'RECO_orth', 'RECO_NT'], ['GPP_NT', 'GPP_DT', 'GPP_DT','RECO_NT', 'RECO_DT', 'RECO_DT'])):
            x = data.groupby(cond).mean()[flux_orth + '_' + form + '_avg']
            y = data.groupby(cond).mean()[flux_other + '_' + form + '_avg']
            analysis_plot(x,y, ax[j,i], fluxes = [flux_orth, flux_other], form = form)
    fig.tight_layout()
    if path:
        fig.savefig(f'{path}/cross_consistency.png', bbox_inches='tight') 

def analysis_plot(x, y, ax, fluxes, form):
    #fig, ax = plt.subplots(1,1, figsize = (10,5))
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    ax.scatter(x,y, c=z, s=100)
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m*x + b, color='black')
    ax.text(x=1.0, y=0.1, s= f'R2 {r2_score(y,x): .2}\n RMSE {np.sqrt(mean_squared_error(x,y)): .2}\n Bias {b: .2}\n ',
            horizontalalignment='right',
            verticalalignment='center',
            transform = ax.transAxes,
            fontsize=24)
    ax.set_title(form, fontsize=24)
    ax.set_xlabel(fluxes[0], fontsize=24)
    ax.set_ylabel(fluxes[1], fontsize=24)
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)

def density_scatter( x , y, ax = None, sort = True, bins = 20, fig=None, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True)
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
    
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )

    if fig is None:
        norm = Normalize(vmin = np.min(z), vmax = np.max(z))
        cbar = fig.colorbar(cm.ScalarMappable(norm = norm,cmap='viridis'), ax=ax)
        cbar.ax.set_ylabel('Density')
    else:
        norm = Normalize(vmin = np.min(z), vmax = np.max(z))
        cbar = fig.colorbar(cm.ScalarMappable(norm = norm,cmap='viridis'), ax=ax)
        cbar.ax.set_ylabel('Density')
        
    return ax


def monthly_curves(data, site, year, flux, res=False, path=False, final="", syn=False):
    data['NEE_DT'] = data["RECO_DT"] - data["GPP_DT"]
    data['NEE_NT'] = data["RECO_NT"] - data["GPP_NT"]

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    subdata = data[(data['site'] == site) & (data['Year']==year)]

    fig, axes = plt.subplots(12,1, figsize=(40,50), sharex=True, sharey=True)
    
    for month, ax in enumerate(axes.flatten()):
        
        df_temp = subdata[subdata["Month"] == month+1]
        if syn:
            DT = df_temp[flux + "_DT"]
            NT = df_temp[flux +"_NT"]
            orth = df_temp[flux +"_orth"]
            GT = df_temp[flux + "_syn"]
            
            if flux == 'NEE':
                GT_clean = df_temp['NEE_syn_clean']

            
        else:
            DT = df_temp[flux + "_DT"]
            NT = df_temp[flux +"_NT"]
            orth = df_temp[flux +"_orth"]
        
            if flux == 'NEE':
                GT = df_temp[flux]

            
            
        if (flux == 'RECO') & (res==True):
            orth_res = df_temp[flux +"_orth_res"]
        
        if syn:
            ax.plot(GT.values, color = "green", label = "Ground Truth")
            if flux == 'NEE':
                ax.plot(GT_clean.values, color = "orange", label = "Ground Truth")
            
        else:
            if flux == 'NEE':
                ax.plot(GT.values, color = "green", label = "Ground Truth")
        
        ax.plot(DT.values, color = "blue", label = "Daytime Method")
        ax.plot(NT.values, color = "black", label = "Nighttime Method")
        ax.plot(orth.values, color = "red", label = "Orthogonal ML")

        if (flux == 'RECO') & (res==True):
            ax.plot(orth_res.values, color = "orange", label = "Orthogonal ML residuals")
        
        #if data_type == 'train':
        #    ax.plot(nuisance.values, color = "green", label = "Orthogonal nuisance")
        if flux == 'NEE':
            ax.set_ylim([-20, 20])
        elif flux == 'GPP': 
            ax.set_ylim([-5, 30])
        elif flux == 'RECO':
            ax.set_ylim([-1, 15])
        
        ax.xaxis.set_tick_params(labelsize=24)
        ax.yaxis.set_tick_params(labelsize=24)


        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc=(0.45, 0.94), fontsize=24)
        ax.set_title(f'{month_names[month]}', fontsize=24)
    fig.suptitle(f'Comparison of the predicted {flux} in monthly curves for different flux partitioning methods in {site} in {year}', fontsize=24)
    fig.tight_layout()
    if path:
        fig.savefig(f'{path}/images/{site}_{flux}_{year}{final}.png', bbox_inches='tight',  transparent=False)         
            