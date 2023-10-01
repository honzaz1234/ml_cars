import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from abc import ABC, abstractmethod
import json

class PredictionSummary():


    def __init__(self, iteration=0, cv=3, refit="neg_mean_squared_error"):

        self.iteration = iteration
        self.summary_df = pd.DataFrame(
            columns=["method", "MSE_train", "MAD_train", "R2_train",
                     "MSE_valid", "MAD_valid", "R2_valid", "MSE_test", "MAD_test", "R2_test", "params", "test_size", "cv_n", "iteration"]
            )
        self.train_x = pd.DataFrame()
        self.test_x = pd.DataFrame()
        self.train_y = pd.DataFrame()
        self.test_y = pd.DataFrame()
        self.train_share = None
        self.cv = cv
        self.refit = refit
        self.df_index = 0 

    def load_data(self, X, y, test_size=0.1, random_state=42):
        self.iteration += 1
        self.train_share = 1 - test_size
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(X, y, test_size=test_size, random_state=random_state
            )
        
    def find_best_model(self, method, param_grid, n_jobs=None):
        m_factory = MethodFactory()
        method_o = m_factory.create_method_object(method=method) 
        grid_search = GridSearchCV(
        estimator=method_o, 
        param_grid=param_grid, 
        cv=self.cv, 
        scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
        refit=self.refit,
        return_train_score=True,
        n_jobs=n_jobs
            )
        grid_search.fit(self.train_x, self.train_y)
        self.save_best_model_stats(stat_dict=grid_search.cv_results_,
                                   best_ind=grid_search.best_index_,
                                   method=method)
        return grid_search

    def estimate_test(self, model, param_dict):
        y_pred = model.predict(self.test_x)
        self.save_test_stats(y_pred=y_pred, param_dict=param_dict)

    def save_test_stats(self, y_pred, param_dict):
        row_test = {"MSE_test": mean_squared_error(self.test_y, y_pred), 
                    "MAD_test": mean_absolute_error(self.test_y, y_pred), 
                    "R2_test": r2_score(self.test_y, y_pred)}
        for key in row_test:
            self.summary_df.loc[self.summary_df["params"]==json.dumps(param_dict), key] = row_test[key]
        
    def perform_method(self, method, **kwargs):
        factory_o = MethodFactory()
        method_object = factory_o.create_method_object(
            method=method, **kwargs)
        method_object.fit(
            X=self.train_x, y=self.train_y
            ) 

    def save_best_model_stats(self, stat_dict, best_ind, method):
        test_prefix = "mean_test_"
        train_prefix = "mean_train_"
        MSE_valid = test_prefix + "neg_mean_squared_error"
        MSE_train= train_prefix + "neg_mean_squared_error"
        MAD_valid = test_prefix + "neg_mean_absolute_error"
        MAD_train = train_prefix + "neg_mean_absolute_error"
        R2_valid = test_prefix + "r2"
        R2_train = train_prefix + "r2"
        print(stat_dict['params'][best_ind])
        stat_row = {"method": method,
                    "MSE_train": stat_dict[MSE_train][best_ind],
                    "MSE_valid": stat_dict[MSE_valid][best_ind],
                    "MAD_train": stat_dict[MAD_train][best_ind],
                    "MAD_valid": stat_dict[MAD_valid][best_ind],
                    "R2_valid": stat_dict[R2_valid][best_ind],
                    "R2_train": stat_dict[R2_train][best_ind], 
                    "params": json.dumps(stat_dict['params'][best_ind]),
                    "iteration": self.iteration,
                    "cv_n" : self.cv,
                    "test_size" : 1-self.train_share
                    }
        print(stat_row)
        print(self.summary_df)
        self.summary_df = pd.concat([self.summary_df, pd.DataFrame(stat_row, index=[self.df_index])])
        self.df_index +=1


class MethodFactory():

    def __init__(self):
        pass

    def create_method_object(self, method, **kwargs):
        if method == "ols":
            return LinearRegression(**kwargs)
        elif  method == "knreg":
            return KNeighborsRegressor(**kwargs)
        elif method == "knclass":
            return KNeighborsClassifier(**kwargs)
        elif method == "dtreg":
            return DecisionTreeRegressor(**kwargs)
        elif method == "dtclass":
            return DecisionTreeClassifier(**kwargs)
        elif method == "rfreg":
            return RandomForestRegressor(**kwargs)
        elif method == "rfclass":
            return RandomForestClassifier(**kwargs)
        elif method == "svr":
            return SVR(**kwargs)
        elif method == "svm":
            return SVC(**kwargs)
        

class MyMethod(ABC):


    @abstractmethod
    def perform_cross_validation(self):
        pass
        
    @abstractmethod  
    def perform_estimation(self):
        pass


class MyLinearRegression(MyMethod):

    def __init__(self):
        self.stat_row = {"MSE_train": None,
                         "MAD_train": None,
                         "MSE_valid": None, 
                         "MAD_valid": None, 
                         "MSE_test": None, 
                         "MAD_test": None, 
                         "test_size": None, 
                         "validation_size": None
        }

    def perform_cross_validation(self, X, y):
        pass

        
    def perform_estimation(self):
        pass