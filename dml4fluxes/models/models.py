import sys
import numpy as np

from econml.dml import NonParamDML
import doubleml as dml
from doubleml import DoubleMLData
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import SGDRegressor, LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.base import BaseEstimator, RegressorMixin

from copy import deepcopy
from dml4fluxes.experiments.utility import transform_t
#sys.path.append('../../bayes-q10')
from src.models.models import EnsembleCustomJNN
from econml.utilities import WeightedModelWrapper


class dml_fluxes():
    def __init__(self, model_y_config, model_t_config, model_final_config, dml_config, reco_config=None):

        self.dml_config = dml_config
        self.model_t_config = model_t_config
        self.model_y_config = model_y_config
        self.model_final_config = model_final_config
        self.model_y_name = model_y_config['model']
        self.reco_config = reco_config
                
        self.est = None
        model_y = getattr(sys.modules[__name__], model_y_config['model'])
        config = model_y_config.copy()
        del config['model']
        self.model_y = model_y(**config)
    
        model_t = getattr(sys.modules[__name__], model_t_config['model'])
        config = model_t_config.copy()
        del config['model']
        self.model_t = model_t(**config)
        
        model_final = getattr(sys.modules[__name__], model_final_config['model'])
        config = model_final_config.copy()
        del config['model']
        self.model_final = WeightedModelWrapper(model_final(**config))

        if self.reco_config:
            self.reco_name = reco_config['model']
            model_reco = getattr(sys.modules[__name__], reco_config['model'])
            config = reco_config.copy()
            del config['model']
            self.model_reco = model_reco(**config)
    
    
    def fit(self, Y, X, T, W=None, X_reco=None):
        #TODO: Build in model selection
        #TODO: Way to go is probably: Hand over test set. Run several NonParamDML model
        #TODO: And score them on the test set. Take the best one. Also try out the
        #TODO: sklearn.model_selection.TimeSeriesSplit (but this only fits the single models)
        self.est = NonParamDML(model_y=self.model_y,
                                model_t=self.model_t,
                                model_final = self.model_final, 
                                **self.dml_config)
        
        self.est.fit(Y=Y, X=X, W=W, T=T, cache_values=True)
        self.gpp, self.reco, self.reco_res, self.nee, self.t, self.lue = self.get_estimators()
        
        if self.reco_config:
            # Compute the residuals on the training set
            res = Y + self.gpp(X, T)
            self.model_reco.fit(X_reco, res)
            
                        
        
    def get_estimators(self):
        def gpp(x, t):
            return -self.est.const_marginal_effect(x)*t
            
        def reco(x, t=None, w=None, y=None, ta=None):
            return np.mean([self.est._models_nuisance[0][i]._model_y.predict(x, w) 
                            - self.est.const_marginal_effect(x) 
                            * (self.est._models_nuisance[0][i]._model_t.predict(x, w)) \
                            for i in range(len(self.est._models_nuisance[0]))], axis=0)
    
        def reco_res(nee, x, t):
            return nee + gpp(x, t)
            
        def nee(x, t, w=None, y=None, ta=None):
            if self.model_y_name=='Reco_DML':
                return -gpp(np.concatenate([np.expand_dims(ta,axis=1), x],axis=1), t) \
                            + self.model_y.reco(x, ta, w)
            else:
                return -gpp(x, t) + reco(x, t, w)

        def t(x, w=None):
            return np.mean([self.est._models_nuisance[0][i]._model_t.predict(x, w) \
                            for i in range(len(self.est._models_nuisance[0]))], axis=0)
        
        def lue(x):
            return -self.est.const_marginal_effect(x)
        
        return gpp, reco, reco_res, nee, t, lue
    
    def get_score(self, x, t, y, w=None):
        """
        Computes the score for the fitted model on given data.


        Args:
            x (ndarray): _description_
            t (ndarray): _description_
            w (ndarray): _description_
            y (ndarray): _description_

        Returns:
            list: _description_
        """
        y_pred = self.nee(x, t, w)
        RMSE = mean_squared_error(y, y_pred, squared=False)
        R2 = r2_score(y, y_pred)
        
        return RMSE, R2


class LightResponseCurve():
    def __init__(self, moving_window, parameter, delta, lue):
        self.moving_window = moving_window
        self.parameter = parameter
        self.delta = delta
        self.lue = lue
        
    def predict(self, rg, x, doy):
        t = transform_t(x=rg,
                        delta=self.delta,
                        moving_window=self.moving_window,
                        parameter = self.parameter,
                        doy=doy)
        return self.lue(x) * t
        

class RECO_DML(BaseEstimator, RegressorMixin):
    def __init__(self, model_y_config, model_t_config,
                    model_final_config, dml_config, Q10_fix=False, split=(1,3,2),
                    copy_model=None):
        self.dml_config = dml_config
        self.Q10_fix = Q10_fix
        self.model_y_config = model_y_config
        self.model_t_config = model_t_config
        self.model_final_config = model_final_config
        self.fitting_done = False
        self.split = split
        self.copy_model = copy_model
        
        if copy_model:
            self.Rb=copy_model[0]
            self.Q10=copy_model[1]
            self.reco=copy_model[2]
            self.fitting_done=True
        else:
            if not self.Q10_fix:
                model_y = getattr(sys.modules[__name__], 'GradientBoostingRegressor')
                config = model_y_config.copy()
                del config['model']
                self.model_y = model_y(**config)

                model_t = getattr(sys.modules[__name__], model_t_config['model'])
                config = model_t_config.copy()
                del config['model']
                self.model_t = model_t(**config)

                model_final = getattr(sys.modules[__name__], model_final_config['model'])
                config = model_final_config.copy()
                del config['model']
                self.model_final = model_final(**config)

                self.est = NonParamDML(model_y=self.model_y,
                        model_t=self.model_t,
                        model_final = self.model_final, 
                        **self.dml_config)

    def fit(self, Y, X):
        if not self.fitting_done:
            if self.Q10_fix:
                dml_procedure = "dml1"
                score = "partialling out"
                n_folds = 5
                n_rep = 1
                
                TA_trans, X = X[:,0], X[:,1:]
                
                self.obj_dml_data = DoubleMLData.from_arrays(X, Y, TA_trans)

                self.dml_plr_obj = dml.DoubleMLPLR(self.obj_dml_data, self.model_y, self.model_t, 
                                                    dml_procedure=dml_procedure,
                                                    score=score,
                                                    n_folds=n_folds, 
                                                    n_rep=n_rep)
                self.obj_dml_data = DoubleMLData.from_arrays(X, Y, TA_trans)
                self.dml_plr_obj.fit(store_predictions=True)
            else:
                self.X = X[:,1:]
                TA_trans, X, W = X[:,0], X[:,1:self.split[1]+1], X[:,1+self.split[1]:1+self.split[1]+self.split[2]]
                self.est.fit(Y=Y, X=X, W=W, T=TA_trans, cache_values=True)
                self.Rb, self.Q10, self.reco = self.get_estimators()
            self.fitting_done = True
        
        return None

    def predict(self, X):
        TA_trans, X, W = X[:,0], X[:,1:self.split[1]+1], X[:,1+self.split[1]:1+self.split[1]+self.split[2]]
        return self.reco(X, TA_trans, W)
    
    def get_estimators(self):
        def Q10(x):
            return np.exp(self.est.const_marginal_effect(x))
            
        def Rb(x, w=None):
            return np.exp(np.mean([self.est._models_nuisance[0][i]._model_y.predict(x, w)-self.est.const_marginal_effect(x)*(self.est._models_nuisance[0][i]._model_t.predict(x, w)) for i in range(len(self.est._models_nuisance[0]))], axis=0))
            #return np.mean([self.est._models_nuisance[0][i]._model_y.predict(x, w)-self.est.const_marginal_effect(x)*t for i in range(len(self.est._models_nuisance[0]))], axis=0)
            #return np.mean([y-self.est.const_marginal_effect(x)*(self.est._models_nuisance[0][i]._model_t.predict(x, w)) for i in range(len(self.est._models_nuisance[0]))], axis=0)       
                    
        def reco(x, ta_trans, w=None):
            return Rb(x, w)*Q10(x)**ta_trans
            
        def ta(x, w=None):
            return np.mean([self.est._models_nuisance[0][i]._model_t.predict(x, w) for i in range(len(self.est._models_nuisance[0]))], axis=0)
                
        return Rb, Q10, reco