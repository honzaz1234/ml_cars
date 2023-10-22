import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRFRegressor
from abc import ABC, abstractmethod
import json

class PredictionSummary():


    def __init__(self, iteration=0, cv=3, refit="neg_mean_squared_error"):

        self.iteration = iteration
        self.summary_df = pd.DataFrame(
            columns=["method", "MSE_train", "MAE_train", "R2_train",
                     "MSE_valid", "MAE_valid", "R2_valid", "MSE_test", "MAE_test", "R2_test", "params", "test_size", "cv_n", "iteration"]
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
        
    def find_best_model(self, method, param_grid, n_jobs=None, **kwargs):
        method_o = create_method_object(method=method, **kwargs) 
        grid_search = GridSearchCV(
        estimator=method_o, 
        param_grid=param_grid, 
        cv=self.cv, 
        scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
        refit=self.refit,
        return_train_score=True,
        n_jobs=n_jobs,
        verbose=4
            )
        grid_search.fit(self.train_x, self.train_y)
        for ind in range(len(grid_search.cv_results_['params'])):
            self.save_model_stats(stat_dict=grid_search.cv_results_,
                                        ind=ind,
                                        method=method)
        return grid_search

    def estimate_test(self, model, param_dict):
        y_pred = model.predict(self.test_x)
        self.save_test_stats(y_pred=y_pred, param_dict=param_dict)

    def save_test_stats(self, y_pred, param_dict):
        row_test = {"MSE_test": mean_squared_error(self.test_y, y_pred), 
                    "MAE_test": mean_absolute_error(self.test_y, y_pred), 
                    "R2_test": r2_score(self.test_y, y_pred)}
        for key in row_test:
            self.summary_df.loc[self.summary_df["params"]==json.dumps(param_dict), key] = row_test[key]
        
    def perform_method(self, method, **kwargs):
        method_object = create_method_object(
            method=method, **kwargs)
        method_object.fit(
            X=self.train_x, y=self.train_y
            ) 

    def save_model_stats(self, stat_dict, ind, method):
        test_prefix = "mean_test_"
        train_prefix = "mean_train_"
        MSE_valid = test_prefix + "neg_mean_squared_error"
        MSE_train= train_prefix + "neg_mean_squared_error"
        MAE_valid = test_prefix + "neg_mean_absolute_error"
        MAE_train = train_prefix + "neg_mean_absolute_error"
        R2_valid = test_prefix + "r2"
        R2_train = train_prefix + "r2"
        stat_row = {"method": method,
                    "MSE_train": stat_dict[MSE_train][ind],
                    "MSE_valid": stat_dict[MSE_valid][ind],
                    "MAE_train": stat_dict[MAE_valid][ind],
                    "MAE_valid": stat_dict[MAE_train][ind],
                    "R2_valid": stat_dict[R2_valid][ind],
                    "R2_train": stat_dict[R2_train][ind], 
                    "params": json.dumps(stat_dict['params'][ind]),
                    "iteration": self.iteration,
                    "cv_n" : self.cv,
                    "test_size" : 1-self.train_share
                    }
        self.summary_df = pd.concat([self.summary_df, pd.DataFrame(stat_row, index=[self.df_index])])
        self.df_index +=1



def create_method_object(method, **kwargs):
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
    elif method == "svc":
        return SVC(**kwargs)
    elif method == "xboostreg":
        return XGBRFRegressor(**kwargs)
    elif method == "xboostclass":
        return XGBClassifier(**kwargs)
    elif method == "adaboostreg":
        return AdaBoostRegressor(**kwargs)
    elif method == "adaboostclass":
        return AdaBoostClassifier(**kwargs)





