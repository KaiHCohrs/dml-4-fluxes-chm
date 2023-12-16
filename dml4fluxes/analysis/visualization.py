import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
from itertools import product

import numpy as np
from skill_metrics import taylor_diagram

import dml4fluxes.datasets.relevant_variables as relevant_variables
from dml4fluxes.datasets.preprocessing import unwrap_time, standardize_column_names
from dml4fluxes.analysis.postprocessing import load_partition

#met_colors = met_brewer.met_brew(name="Signac", n=14)


def mean_diurnal_cycle(data, flux, path = False, method='orth', index=''):
    data['grouped_months'] = data.apply(lambda x: (x['Month']-1) // 2 + 1, axis = 1)
    fig, ax = plt.subplots(2, 6, figsize = (40,10), sharex=True)
    
    for j, months in enumerate(['Jan-Feb', 'Mar-Apr', 'May-Jun', 'Jul-Aug', 'Sep-Oct', 'Nov-Dec']):
        df_temp = data.loc[(data["grouped_months"] == j+1),:]
        means = df_temp.groupby(['Time']).mean()
        
        ax[0,j].plot(means[f'{flux}_{method}'], color='red', label=method)
        ax[0,j].plot(means[flux + '_NT'], color='black', label='NT')
        ax[0,j].plot(means[flux + '_DT'], color='blue', label='DT')
        ax[0,j].vlines(x = 24,  ymin=-1, ymax=20, color='gray', alpha=0.5)
        
        
        ax[1,j].plot(means[f'{flux}_{method}']-means[flux + '_NT'], color='black', label=f'{method}-NT')
        ax[1,j].plot(means[f'{flux}_{method}']-means[flux + '_DT'], color='blue', label=f'{method}-DT')
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
        fig.savefig(f'{path}/mean_diurnal_cycle_{flux}_{method}{index}.png', bbox_inches='tight') 


def seasonal_cycle_plot(data, path=False, method='orth', index=""):
    fig, ax = plt.subplots(2, 1, figsize = (10,10))
    for i, flux in enumerate(['GPP', 'RECO']):
        ax[i].plot(data.groupby('Month').mean()[flux + f'_{method}_seasonal_avg'], color='red', label=f'{method}')
        ax[i].fill_between(range(1,13), data.groupby('Month').mean()[flux + f'_{method}_seasonal_avg']-data.groupby('Month').std()[flux + f'_{method}_seasonal_avg'], data.groupby('Month').mean()[flux + f'_{method}_seasonal_avg']+data.groupby('Month').std()[flux + f'_{method}_seasonal_avg'], alpha=0.2, color='red')
        #rather: data.groupby(['Month', 'site', 'Year']).mean().groupby(['Month']).std()[flux + f'_{method}_seasonal_avg'] ????
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
        fig.savefig(f'{path}/seasonal_cycle_{method}{index}.png', bbox_inches='tight')


def without_nans(x,y):
    return x[~np.isnan(x) & ~np.isnan(y)], y[~np.isnan(x) & ~np.isnan(y)]

def cross_consistency_plots(data, path=False, method='orth', index=1, QC=True):
    fig, ax = plt.subplots(6,4, figsize = (10*4,5*6))
    for i, (cond, form) in enumerate(zip([['site', 'Year', 'doy'],['site', 'Year'], ['site', 'Year', 'soy'], ['site', 'Year', 'doy']], ['daily', 'yearly', 'seasonal_without_yearly', 'daily_anomalies'])):
        for j, (flux_orth, flux_other) in enumerate(zip([f'GPP_{method}', f'GPP_{method}', 'GPP_NT', f'RECO_{method}', f'RECO_{method}', 'RECO_NT'], ['GPP_NT', 'GPP_DT', 'GPP_DT','RECO_NT', 'RECO_DT', 'RECO_DT'])):
            if QC:
                subdata = data.loc[(data[f'quality_mask_{form}']==1) & (data['QC']==0),:]
            else:
                subdata = data
            x = subdata.groupby(cond).mean()[flux_orth + '_' + form + '_avg']
            y = subdata.groupby(cond).mean()[flux_other + '_' + form + '_avg']
            x, y = without_nans(x,y)
            analysis_plot(x,y, ax[j,i], fluxes = [flux_orth, flux_other], form = form)
    fig.tight_layout()
    if path:
        fig.savefig(f'{path}/cross_consistency_{method}{index}.png', bbox_inches='tight') 

def analysis_plot(x, y, ax, fluxes, form):
    #fig, ax = plt.subplots(1,1, figsize = (10,5))
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    ax.scatter(x,y, c=z, s=100)
    #m, b = np.polyfit(x, y, 1)
    #ax.plot(x, m*x + b, color='black')
    plot_min = min([x.min(), y.min()])
    plot_max = max([x.max(), y.max()])
    ax.plot([plot_min, plot_max],[plot_min, plot_max], color='black')
    b = np.mean(x-y)
    ax.text(x=1.0, y=0.1, s= f'R2 {r2_score(y,x): .2}\n RMSE {np.sqrt(mean_squared_error(x,y)): .2}\n Bias {b: .2}\n ',
            horizontalalignment='right',
            verticalalignment='center',
            transform = ax.transAxes,
            fontsize=24)
    ax.set_title(form, fontsize=24)
    ax.set_xlabel(fluxes[0], fontsize=24)
    ax.set_ylabel(fluxes[1], fontsize=24)
    #ax.set_xlim(x.min(), x.max())
    #ax.set_ylim(y.min(), y.max())
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
       
#def light_curve(data, path = False, input='VPD'):
    

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

def cross_consistency(data, path=False, method='orth', index=1, QC=True):
    dataframe = pd.DataFrame()
    data = load_partition(experiment_name, site, year=2015, syn=syn)
    
    data['DateTime'] = data.index
    data = unwrap_time(data)
    data = standardize_column_names(data)
    data = data[list(set(data.columns)
                & set(relevant_variables.variables))]
    
    fig, ax = plt.subplots(6,4, figsize = (10*4,5*6))
    for i, (cond, form) in enumerate(zip([['site', 'Year', 'doy'],['site', 'Year'], ['site', 'Year', 'soy'], ['site', 'Year', 'doy']], ['daily', 'yearly', 'seasonal_without_yearly', 'daily_anomalies'])):
        for j, (flux_orth, flux_other) in enumerate(zip([f'GPP_{method}', f'GPP_{method}', 'GPP_NT', f'RECO_{method}', f'RECO_{method}', 'RECO_NT'], ['GPP_NT', 'GPP_DT', 'GPP_DT','RECO_NT', 'RECO_DT', 'RECO_DT'])):
            if QC:
                subdata = data.loc[(data[f'quality_mask_{form}']==1) & (data['QC']==0),:]
            else:
                subdata = data
            x = subdata.groupby(cond).mean()[flux_orth + '_' + form + '_avg']
            y = subdata.groupby(cond).mean()[flux_other + '_' + form + '_avg']
            x, y = without_nans(x,y)
            analysis_plot(x,y, ax[j,i], fluxes = [flux_orth, flux_other], form = form)
    fig.tight_layout()
    if path:
        fig.savefig(f'{path}/cross_consistency_{method}{index}.png', bbox_inches='tight') 


def monthly_curves(flux, data, compare_to=['DT', 'NT'], results_path=None, suffix=""):
    
    # LOAD DATA from 
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    colors = ['#d73027', '#fc8d59','#91bfdb','tab:green']
    line_styles = ['-', '--', '-.', '-']

    fig, axes = plt.subplots(12,1, figsize=(40,50), sharex = True, sharey=True)
    
    for month, ax in enumerate(axes.flatten()):
        df_temp = data[data["Month"] == month+1]
        mean = df_temp[flux + "_mean"]
        std = df_temp[flux +"_std"]
        df_temp = df_temp.sort_values("tom")
        # Check if there are duplicates in df_temp['tom']
        if df_temp['tom'].duplicated().any():
            print("Duplicates in tom")
        
        ax.fill_between(df_temp["tom"].unique(), mean.values - std*1.96, mean.values + std*1.96, color = 'blue', alpha=0.2)
        ax.plot(df_temp["tom"].unique(), mean.values, color = 'blue', label = "Mean")
        ax.set_title(month_names[month])
        
        # Make xticks time of month tom
        ax.set_xticks(df_temp["tom"].unique()[::48])
        # Make every 48th tick a label that corresponds to the day
        ax.set_xticklabels(df_temp["dom"][::48], rotation=90)
        
        
        for i, compare in enumerate(compare_to):
            ax.plot(df_temp[f'{flux}_{compare}'].values, color = colors[i], linestyle=line_styles[i] , label = compare)
        
        for i, qc in enumerate(df_temp['NEE_QC']):
            if qc == 0:
                ax.axvspan(i-0.5, i+0.5, color='grey', alpha=0.3, lw=0)
        
        #Todo: make limits depending on max and min values
        if flux == 'NEE':
            ax.set_ylim([-25, 20])
        elif flux == 'GPP': 
            ax.set_ylim([-5, 30])
        elif flux == 'RECO':
            ax.set_ylim([-5, 30])
        
        ax.xaxis.set_tick_params(labelsize=24)
        ax.yaxis.set_tick_params(labelsize=24)


    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.45, 0.90), fontsize=24)

    fig.suptitle(f'Comparison of the predicted {flux} in monthly curves for different flux partitioning methods.', fontsize=24)
    #fig.tight_layout()
    if results_path:
        fig.savefig(f'{results_path}/{flux}{suffix}.pdf', bbox_inches='tight', facecolor='white',  transparent=False)   

def taylor_plot(NEE, GPP, RECO, ensemble_size, filtered=True, results_path=None, suffix=""):
#filtered = True
#ensemble_size = 100
#obs_label = 'NEE_NT'
    data = {'NEE': NEE, 'GPP': GPP, 'RECO': RECO}
            
    compare_to = ['mean', 'DT', 'NT']
    fig, axes = plt.subplots(2,3, figsize=(18,12))
    
    for j, (obs, flux) in enumerate(product(['NT', 'DT'], ['NEE', 'GPP', 'RECO'])):
        compare_to.remove(obs)
        flux_data = data[flux]
        ax = axes.flatten()[j]
        if filtered:
            mask = flux_data['NEE_QC'] == 0
        else:
            True

        # Generate some dummy data
        observed = flux_data[flux+'_'+obs].values[mask]
        models = [flux_data[f'{flux}_{i}'].values[mask] for i in range(ensemble_size)] + [flux_data[method].values[mask] for method in [flux+'_'+method for method in compare_to]]

        # Compute standard deviations and correlation coefficients
        stddevs = np.array([np.std(observed), *[np.std(model) for model in models]])

        # Compute centered root mean squared error for each model to the observed data
        observations_centered = observed - np.mean(observed)
        models_centered = [model - np.mean(model) for model in models]
        RMSE_centered = [mean_squared_error(observations_centered, model_centered, squared=False) for model_centered in models_centered]
        CRMSES = np.array([0] + RMSE_centered)

        # Compute correlation coefficient
        corrcoefs = np.array([1]+[*[np.corrcoef(observed, model)[0, 1] for model in models]])

        # Create the Taylor diagram

        labels = ['mean', 'DT']
        colors = ['black', 'black']
        markers = ['x', 'x']

        # Add title to subplot
        ax.set_title(f'{flux} {obs}', fontsize=16)

        taylor_diagram(ax, stddevs[:ensemble_size+1], CRMSES[:ensemble_size+1], corrcoefs[:ensemble_size+1], markerLabel=['Observation', *([f'' for i in range(ensemble_size)])], styleOBS = '-', colOBS = 'r', markerobs = 'o', titleOBS = 'observation', 
                    markerColor='orange', alpha=0.5, markersymbol='.', markerSize=3, colRMS='red', widthRMS=2.0, 
                    checkstats='on')
        for i in range(2):
            taylor_diagram(ax, stddevs[[0,ensemble_size+1+i]], CRMSES[[0,ensemble_size+1+i]], corrcoefs[[0,ensemble_size+1+i]], markerLabel=['Observation',labels[i]], styleOBS = '-', colOBS = 'r', markerobs = 'o', titleOBS = 'observation', 
                        markerColor=colors[i], markersymbol=markers[i], markerSize=6, colRMS='red', widthRMS=2.0, 
                        checkstats='on')
        compare_to = compare_to + [obs]
        
    fig.suptitle(f'Taylor plot of estimations over ensemble and other models.', fontsize=12)

    if results_path:
        fig.savefig(f'{results_path}/Taylor_diagram{suffix}.pdf', bbox_inches='tight', facecolor='white',  transparent=False)   

def monthly_Q10_model_curves(data, site, year, res=False, path=False, final="", syn=False):
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    subdata = data[(data['site'] == site) & (data['Year']==year)]

    fig, axes = plt.subplots(12,1, figsize=(40,50), sharex=True, sharey=True)
    
    for month, ax in enumerate(axes.flatten()):
        
        df_temp = subdata[subdata["Month"] == month+1]
        Q10 = df_temp["Q10"]
        #Rb = df_temp["Rb"]
        RECO = df_temp["RECO_orth"]    
            
        if res:
            Q10_res = df_temp["Q10_res"]
            #Rb_res = df_temp["Rb_res"]
            RECO_res = df_temp["RECO_orth_res"]
        
        ax.plot(Q10.values, color = "purple", label = "Q10")
        #ax.plot(Rb.values, color = "red", label = "Rb")
        ax.plot(RECO.values, color = "blue", label = "RECO")
        
        if res:
            ax.plot(Q10_res.values, color = "purple", linestyle="dotted",  label = "Q10_res")
            #ax.plot(Rb_res.values, color = "red", linestyle="dotted", label = "Rb_res")
            ax.plot(RECO_res.values, color = "blue", linestyle="dotted", label = "RECO_res")
        
        #if data_type == 'train':
        #    ax.plot(nuisance.values, color = "green", label = "Orthogonal nuisance")
        ax.set_ylim([-1, 15])
        
        ax.xaxis.set_tick_params(labelsize=24)
        ax.yaxis.set_tick_params(labelsize=24)


        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc=(0.45, 0.94), fontsize=24)
        ax.set_title(f'{month_names[month]}', fontsize=24)
    fig.suptitle(f'Comparison of RECO, Rb and Q10 in monthly curves for different flux partitioning methods in {site} in {year}', fontsize=24)
    fig.tight_layout()
    if path:
        fig.savefig(f'{path}/images/{site}_{flux}_{year}{final}.png', bbox_inches='tight',  transparent=False)         
        

def monthly_curves_cutout(data, site, year, flux, res=False, path=False, final="", syn=False, day=0):
    
    data['NEE_DT'] = data["RECO_DT"] - data["GPP_DT"]
    data['NEE_NT'] = data["RECO_NT"] - data["GPP_NT"]

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    subdata = data[(data['site'] == site) & (data['Year']==year)]
    plt.style.use('ggplot')

    fig, axes = plt.subplots(3,4, figsize=(15,15), sharex=True, sharey=True)
    
    for month, ax in enumerate(axes.flatten()):
        df_temp = subdata[subdata["Month"] == month+1].copy()
        df_temp = df_temp.reset_index().loc[48*day:48*(day+3),:]
        
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
            ann = df_temp[flux +"_ann"]
            
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
        ax.plot(ann.values, color = "orange", label = "ANN")
        ax.plot(orth.values, color = "red", label = "Orthogonal ML")

        if (flux == 'RECO') & (res==True):
            ax.plot(orth_res.values, color = "orange", label = "Orthogonal ML residuals")
        
        #if data_type == 'train':
        #    ax.plot(nuisance.values, color = "green", label = "Orthogonal nuisance")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc=(0.45, 0.87))
    
        if flux == 'NEE':
            ax.set_ylim([-20, 20])
        elif flux == 'GPP': 
            ax.set_ylim([-5, 30])
        elif flux == 'RECO':
            ax.set_ylim([-1, 15])
#        ax.set_xticks(list(range(0,48,8)))
#        ax.set_xticklabels(list(range(0,24,4)), fontsize = 10)
        if month // 4 == 2:
            ax.set_xlabel("3 days time window")
        if month % 4 == 0:
            ax.set_ylabel("GPP")
        ax.set_title(month_names[month])

    fig.suptitle(f'Comparison of the predicted {flux} over a three days window for different flux partitioning methods in {site} in {year}', fontsize=24)
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    if path:
        fig.savefig(f'{path}/images/{site}_{flux}_{year}{final}.png', bbox_inches='tight',  transparent=False)         