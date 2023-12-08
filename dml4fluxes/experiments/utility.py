from cmath import inf
import numpy as np
from os import listdir
from os.path import isdir, join
import json

from sklearn.linear_model import LinearRegression
from scipy.optimize import least_squares, minimize

        
def get_available_sites(path):
    fluxsites = [f[4:] for f in listdir(path) if isdir(join(path, f)) and f.startswith("FLX")]
    return fluxsites

def get_igbp_of_site(site):
    IGBP_of_site = {'ES-Abr': 'SAV',
                    'DE-Hai': 'DBF',
                    'FR-FBn': 'MF',
                    'DE-Hzd': 'DBF',
                    'CH-Cha': 'GRA',
                    'AU-Cpr': 'SAV',
                    'AU-DaP': 'GRA',
                    'AU-Dry': 'SAV',
                    'AU-How': 'WSA',
                    'AU-Stp': 'GRA',
                    'BE-Lon': 'CRO',
                    'BE-Vie': 'MF',
                    'CA-Qfo': 'ENF',
                    'DE-Geb': 'CRO',
                    'DE-Gri': 'GRA',
                    'DE-Kli': 'CRO',
                    'DE-Obe': 'ENF',
                    'DE-Tha': 'ENF',
                    'DK-Sor': 'DBF',
                    'FI-Hyy': 'ENF',
                    'FR-LBr': 'ENF',
                    'GF-Guy': 'EBF',
                    'IT-BCi': 'CRO',
                    'IT-Cp2': 'EBF',
                    'IT-Cpz': 'EBF',
                    'IT-MBo': 'GRA',
                    'IT-Noe': 'CSH',
                    'IT-Ro1': 'DBF',
                    'IT-SRo': 'ENF',
                    'NL-Loo': 'ENF',
                    'RU-Fyo': 'ENF',
                    'US-ARM': 'CRO',
                    'US-GLE': 'ENF',
                    'US-MMS': 'DBF',
                    'US-NR1': 'ENF',
                    'US-SRG': 'GRA',
                    'US-SRM': 'WSA',
                    'US-UMB': 'DBF',
                    'US-Whs': 'OSH',
                    'US-Wkg': 'GRA',
                    'ZA-Kru': 'SAV',
    }
    return IGBP_of_site[site]

def clean_outliers(x, mean=True, times=2):
    if mean:
        crit = np.mean(x)
    else:
        crit = np.median(x)
    y = (np.roll(x,-1,0) + np.roll(x, 1, 0))/2
    conditions = [np.abs(x-y) < times*crit, np.abs(x - y) >= times*crit]
    choicelist = [x, y]
    x_new = np.select(conditions, choicelist)

    return x_new 

def transform_t(x, delta='heuristic8', data=None, threshold=2000, 
                month_wise=False, parameter=None, moving_window=None,
                doy=None, optimizer=None, target=None):
    if delta == 'heuristic3':
        sub_data = data
        if month_wise:
            if not parameter:
                parameter = list()
                xs=list()
                for i in range(1,13):
                    indices = data['Month']==i
                    sub_data = data[indices]
                    t = sub_data['SW_IN']
                    if target:
                        y = sub_data[target]
                    else:
                        y = sub_data['NEE']
                    x_monthly  = x[indices]
                    res_lsq = least_squares(rectangular_hyp, [0.1,0.001,-2], bounds=([0, 0, -inf], [10, 1, 0]), args=(t, y))
                    alpha, delta, b = res_lsq['x']
                    print(res_lsq['x'])
                    xs.append(alpha * x_monthly/(delta*x_monthly+1))
                    parameter.append([alpha, delta])             
                return np.concatenate(xs), parameter
            else:
                xs=list()
                for i in range(1,13):
                    indices = data['Month']==i
                    x_monthly  = x[indices]
                    alpha, delta = parameter[i-1]
                    xs.append(alpha * x_monthly/(delta*x_monthly+1))
                return np.concatenate(xs)
        elif moving_window:
            training_windows, window_size = moving_window
            data['woy'] = data['doy'].apply(lambda x: x//window_size)
            if not parameter:
                parameter = list()
                xs=list()
                #for i in data['woy'].unique():
                for i in range((366 // window_size)+1):
                    indices_fit = np.isin(data['woy'], range(i-training_windows,i+training_windows+1))
                    indices = data['woy']==i
                    sub_data_fit = data[indices_fit]
                    t = sub_data_fit['SW_IN']
                    if target:
                        y = sub_data_fit[target]
                    else:
                        y = sub_data_fit['NEE']
                    x_window  = x[indices]
                    res_lsq = least_squares(rectangular_hyp, [0.1,0.001,-2], bounds=([0, 0, -inf], [10, 1, 0]), args=(t, y))
                    alpha, delta, b = res_lsq['x']
                    #print(res_lsq['x'])
                    xs.append(alpha * x_window/(delta*x_window+1))
                    parameter.append([alpha, delta])
                return np.concatenate(xs), parameter
            else:
                training_windows, window_size = moving_window
                data['woy'] = data['doy'].apply(lambda x: x//window_size)
                xs=list()
                alphas=list()
                deltas=list()
                for i in data['woy'].unique():
                    indices = data['woy']==i
                    x_window  = x[indices]
                    #alpha, delta = parameter[i-1]
                    len(parameter)
                    alpha, delta = parameter[i]
                    xs.append(alpha * x_window/(delta*x_window+1))
                    alphas += len(x_window) * [alpha]
                    deltas += len(x_window) * [delta]
                    
                return np.concatenate(xs), np.array(alphas), np.array(deltas)
        else:
            if not parameter:
                t = sub_data['SW_IN']
                y = sub_data['NEE']
                res_lsq = least_squares(rectangular_hyp, [0.1,0.001,-2], bounds=([0, 0, -inf], [10, 1, 0]), args=(t, y))
                alpha, delta, b = res_lsq['x']
                #print(res_lsq['x'])
                parameter = alpha, delta
                return alpha * x/(delta*x+1), parameter
            else:
                alpha, delta = parameter
                return alpha * x/(delta*x+1)    
    elif delta == 'heuristic7':
        sub_data = data
        training_windows, window_size = moving_window
        if data is not None:
            data['woy'] = data['doy'].apply(lambda x: x//window_size)
            woy = data['woy']
        else:
            woy = doy//window_size    
            woy = woy.astype(int)
        if not parameter:
            parameter = list()
            xs=list()
            #for i in data['woy'].unique():
            for i in range((366 // window_size)+1):
                indices_fit = np.isin(woy, range(i-training_windows,i+training_windows+1))
                indices = woy==i
                sub_data_fit = data[indices_fit]
                t = sub_data_fit['SW_IN']
                if target:
                    y = sub_data_fit[target]
                else:
                    y = sub_data_fit['NEE']
                x_window  = x[indices]
                res_lsq = least_squares(rectangular_hyp_neg_simple, [0.5,0.5], bounds=([0, 0], [100, inf]), args=(t, y))
                delta, b = res_lsq['x']
                #print(res_lsq['x'])
                #print(alpha/beta)
                xs.append(x_window/(delta*x_window+1))
                parameter.append([delta])
            return np.concatenate(xs), parameter
        else:
            training_windows, window_size = moving_window
            if data is not None:
                data['woy'] = data['doy'].apply(lambda x: x//window_size)
                woy = data['woy']
            else:
                woy = doy//window_size
                woy = woy.astype(int)
            xs=list()
            deltas=list()
            for i in np.unique(woy):
                indices = woy==i
                x_window  = x[indices]
                #alpha, delta = parameter[i-1]
                delta= parameter[i]
                
                xs.append(x_window/(delta*x_window+1))
                deltas += len(x_window) * [delta]
            return np.concatenate(xs), np.array(deltas), np.array(deltas)
    elif delta == 'heuristic8':
        sub_data = data
        training_windows, window_size = moving_window
        if data is not None:
            data['woy'] = data['doy'].apply(lambda x: x//window_size)
            woy = data['woy']
        else:
            woy = doy//window_size    
            woy = woy.astype(int)
        if not parameter:
            parameter = list()
            xs=list()
            #for i in data['woy'].unique():
            for i in range((366 // window_size)+1):
                indices_fit = np.isin(woy, range(i-training_windows,i+training_windows+1))
                indices = woy==i
                sub_data_fit = data[indices_fit]
                t = sub_data_fit['SW_IN']
                if target:
                    y = sub_data_fit[target]
                else:
                    y = sub_data_fit['NEE']
                x_window  = x[indices]
                if optimizer:
                    f_MSE_loss = partial(rectangular_hyp_original_MSE_loss, t=t, y=y)
                    res = minimize(f_MSE_loss, x0=[-0.05, -10, 0.5] ,method=optimizer, bounds=[(-10,0), (-100, 0), (0, inf)])
                    alpha, beta, b = res['x']
                    #print(f'alpha: {alpha:.3f}, beta: {beta:.3f}')
                else:
                    res_lsq = least_squares(rectangular_hyp_original, [-0.05,-10,0.5], bounds=([-10, -100, 0], [0, 0, inf]), args=(t, y))
                    alpha, beta, b = res_lsq['x']
                    #print(f'alpha: {alpha:.3f}, beta: {beta:.3f}')
                #print(res_lsq['x'])
                #print(alpha/beta)
                xs.append(alpha*x_window/(alpha/beta*x_window+1))
                parameter.append([alpha, beta])
            return np.concatenate(xs), parameter
        else:
            training_windows, window_size = moving_window
            if data is not None:
                data['woy'] = data['doy'].apply(lambda x: x//window_size)
                woy = data['woy']
            else:
                woy = doy//window_size
                woy = woy.astype(int)
            xs=list()
            alphas=list()
            betas=list()
            for i in np.unique(woy):
                indices = woy==i
                x_window  = x[indices]
                #alpha, delta = parameter[i-1]
                alpha, beta= parameter[i]
                
                xs.append(alpha*x_window/(alpha/beta*x_window+1))
                alphas += len(x_window) * [alpha]
                betas += len(x_window) * [beta]
            return np.concatenate(xs), np.array(alphas), np.array(betas)
    else:
        return x / (delta * x + 1)
    
from functools import partial
    
def simple_rectangular_hyp(x, t, y, b):
    return t / (x/b * t + 1) + y
    
def rectangular_hyp(x, t, y):
    return x[0] * t / (1 + x[1]*t) + x[2] + y

def rectangular_hyp_original(x, t, y):
    return y - x[0] * x[1] * t / (x[0] * t + x[1]) - x[2]

def rectangular_hyp_original_MSE_loss(x, t, y):
    return np.mean((y - x[0] * x[1] * t / (x[0] * t + x[1]) - x[2])**2)

def rectangular_hyp_neg_simple(x, t, y):
    return  y + t / (1 + x[0]*t) - x[1]

def gen_rectangular_hyp(x, t, y):
    return 2 * x[0] * t / (t + x[1] + ((t + x[1])**2 - 4*x[1]*x[2]*t)**(1/2)) + x[3] + y


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_json'):
            return obj.to_json(orient='records')
        return json.JSONEncoder.default(self, obj)


# =================== depricated ============================
            
    # elif delta == 'heuristic4':
    #     if not parameter:
    #         b = x.max()
    #         parameter = b
    #         return x / (5/b * x + 1), parameter
    #     else:
    #         b = parameter
    #         return x / (5/b * x + 1)
    
    # elif delta == 'heuristic5':
    #     if not parameter:
    #         sub_data = data
    #         #sub_data = data[data['NIGHT']==0]
    #         #sub_data = sub_data[sub_data['SW_IN']<threshold]
    #         t = sub_data['SW_IN']
    #         y = sub_data['NEE']
    #         res_lsq = least_squares(gen_rectangular_hyp, [10,50,0.5,-5], args=(t, y))
    #         c = res_lsq['x']
    #         parameter = c
    #         return 2 * c[0] * x / (x + c[1] + ((x + c[1])**2 - 4*c[1]*c[2]*x)**(1/2)+1e-10), parameter
    #     else:
    #         c = parameter
    #         return 2 * c[0] * x / (x + c[1] + ((x + c[1])**2 - 4*c[1]*c[2]*x)**(1/2)+1e-10)
            
    # elif delta == 'heuristic6':
    #     if not parameter:
    #         sub_data = data
    #         b = x.max()
    #         t = sub_data['SW_IN']
    #         y = sub_data['NEE']
    #         res_lsq = least_squares(simple_rectangular_hyp, 5, args=(t, y, b))
    #         c = res_lsq['x']
    #         parameter = b, c
    #         return x / (c/b * x + 1), parameter
    #     else:
    #         b, c = parameter
    #         return x / (c/b * x + 1)
    
    
    #     if delta == 'heuristic1':
    #     if not parameter:
    #         model = LinearRegression()
    #         sub_data = data[data['NIGHT']==0]
    #         sub_data = sub_data[sub_data['SW_IN']<threshold]
    #         model.fit(sub_data['T'].values.reshape(-1, 1), sub_data['NEE'])
    #         gamma = -model.coef_**(-1)
    #         parameter = gamma
    #         return x / (x + gamma), parameter
    #     else:
    #         gamma = parameter
    #         return x / (x + gamma)
    
    # elif delta == 'heuristic2':
    #     if not parameter:
    #         model = LinearRegression()
    #         sub_data = data[data['NIGHT']==0]
    #         sub_data = sub_data[sub_data['SW_IN']<threshold]
    #         model.fit(sub_data['T'].values.reshape(-1, 1), sub_data['NEE'])
    #         a = -model.coef_
    #         b = x.max()
    #         parameter = a, b
    #         return a*x / (a/b * x + 1), parameter
    #     else:
    #         a, b = parameter
    #         return a*x / (a/b * x + 1)