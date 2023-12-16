import numpy as np
from os import listdir
from os.path import isdir, join
import json

from sklearn.linear_model import LinearRegression
from scipy.optimize import least_squares

        
def get_available_sites(path):
    fluxsites = [f[4:] for f in listdir(path) if isdir(join(path, f)) and f.startswith("FLX")]
    return fluxsites

def get_IGBP_of_site(site):
    IGBP_of_site = {'IT-Cp2': 'EBF',
                    'ES-Abr': 'SAV',
                    'DK-Sor': 'DBF',
                    'DE-Hai': 'DBF',
                    'BE-Lon': 'CRO',
                    'FR-FBn': 'MF',
                    'DE-Hzd': 'DBF',
                    'CH-Cha': 'GRA',
                    'DE-Gri': 'GRA',
                    'FI-Hyy': 'ENF',
        
    }
    return IGBP_of_site[site]
    

def transform_T(x, delta='heuristic1', data=None, threshold=2000, month_wise=False, parameter=None):
    if delta == 'heuristic1':
        if not parameter:
            model = LinearRegression()
            sub_data = data[data['NIGHT']==0]
            sub_data = sub_data[sub_data['SW_IN']<threshold]
            model.fit(sub_data['T'].values.reshape(-1, 1), sub_data['NEE'])
            gamma = -model.coef_**(-1)
            parameter = gamma
            return x / (x + gamma), parameter
        else:
            gamma = parameter
            return x / (x + gamma)
    
    elif delta == 'heuristic2':
        if not parameter:
            model = LinearRegression()
            sub_data = data[data['NIGHT']==0]
            sub_data = sub_data[sub_data['SW_IN']<threshold]
            model.fit(sub_data['T'].values.reshape(-1, 1), sub_data['NEE'])
            a = -model.coef_
            b = x.max()
            parameter = a, b
            return a*x / (a/b * x + 1), parameter
        else:
            a, b = parameter
            return a*x / (a/b * x + 1)

    elif delta == 'heuristic3':
        sub_data = data
        #sub_data = data[data['NIGHT']==0]
        #sub_data = sub_data[sub_data['SW_IN']<threshold]
        if month_wise:
            if not parameter:
                parameter = list()
                xs=list()
                for i in range(1,13):
                    indices = data['Month']==i
                    sub_data = data[indices]
                    t = sub_data['SW_IN']
                    y = sub_data['NEE']
                    x_monthly  = x[indices]
                    res_lsq = least_squares(rectangular_hyp, [0.1,0.001,-2], args=(t, y))
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
        else:
            if not parameter:
                t = sub_data['SW_IN']
                y = sub_data['NEE']
                #res_lsq = least_squares(rectangular_hyp, [0.1,0.001,-2], args=(t, y))
                res_lsq = least_squares(rectangular_hyp, [0.1,0.001,-2], args=(t, y))
                alpha, delta, b = res_lsq['x']
                print(res_lsq['x'])
                parameter = alpha, delta
                return alpha * x/(delta*x+1), parameter
            else:
                alpha, delta = parameter
                return alpha * x/(delta*x+1)
    
    elif delta == 'heuristic4':
        if not parameter:
            b = x.max()
            parameter = b
            return x / (5/b * x + 1), parameter
        else:
            b = parameter
            return x / (5/b * x + 1)
    
    elif delta == 'heuristic5':
        if not parameter:
            sub_data = data
            #sub_data = data[data['NIGHT']==0]
            #sub_data = sub_data[sub_data['SW_IN']<threshold]
            t = sub_data['SW_IN']
            y = sub_data['NEE']
            res_lsq = least_squares(gen_rectangular_hyp, [10,50,0.5,-5], args=(t, y))
            c = res_lsq['x']
            parameter = c
            return 2 * c[0] * x / (x + c[1] + ((x + c[1])**2 - 4*c[1]*c[2]*x)**(1/2)+1e-10), parameter
        else:
            c = parameter
            return 2 * c[0] * x / (x + c[1] + ((x + c[1])**2 - 4*c[1]*c[2]*x)**(1/2)+1e-10)
            
    elif delta == 'heuristic6':
        if not parameter:
            sub_data = data
            b = x.max()
            t = sub_data['SW_IN']
            y = sub_data['NEE']
            res_lsq = least_squares(simple_rectangular_hyp, 5, args=(t, y, b))
            c = res_lsq['x']
            parameter = b, c
            return x / (c/b * x + 1), parameter
        else:
            b, c = parameter
            return x / (c/b * x + 1)
    else:
        return x / (delta * x + 1)
    
def simple_rectangular_hyp(x, t, y, b):
    return t / (x/b * t + 1) + y
    
def rectangular_hyp(x, t, y):
    return x[0] * t / (1 + x[1]*t) + x[2] + y

def gen_rectangular_hyp(x, t, y):
    return 2 * x[0] * t / (t + x[1] + ((t + x[1])**2 - 4*x[1]*x[2]*t)**(1/2)) + x[3] + y


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_json'):
            return obj.to_json(orient='records')
        return json.JSONEncoder.default(self, obj)
