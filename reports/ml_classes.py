import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRFRegressor

class PredictionSummary():
    """class whose purpose is collecting metrics data from all the model specifications in one table 
    """


    def __init__(self, iteration=0, cv=3, refit="neg_mean_squared_error"):
        """attributes:
                    iteration - auxiliary variable denoting different times the data set was divided into training and testing
                    cv - number of subset used in cross validation when training model
                    refit - metric based on which the best model specification is decided
                    train_x/y,test_x/y subset of predictor/target variables used for estimation in training and testing
                    summary_df - df used for storing metric of all the models estimated
                    """
                    
        self.iteration = iteration
        self.summary_df = pd.DataFrame(
            columns=[
                "method", "MSE_train", "MAE_train", "R2_train",
                "MSE_valid", "MAE_valid", "R2_valid", "MSE_test", "MAE_test", "R2_test", "params", "test_size", "cv_n", "iteration"
            ]
            )
        self.train_x = pd.DataFrame()
        self.test_x = pd.DataFrame()
        self.train_y = pd.DataFrame()
        self.test_y = pd.DataFrame()
        self.train_share = None
        self.cv = cv
        self.refit = refit
        self.df_index = 0 

    def load_data(
            self, X: pd.DataFrame, y: pd.DataFrame, test_size: int = 0.1, random_state: int = 42) -> None:
        """method whose purpose is to divide dataset into training and testing sub sets
        """

        self.iteration += 1
        self.train_share = 1 - test_size
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
    def find_best_model(
            self, method, param_grid: dict, n_jobs:int = None, **kwargs):
        """method for selecting the best model specification
            parameters:
                    method - sklearn model object
                    param_grid - dict with different values of parameters that should be tried 
                    n_jobs - how many processor cores to use for estimation
        """

        method_o = self.create_method_object(method=method, **kwargs) 
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
            self._save_model_stats(stat_dict=grid_search.cv_results_,
                                        ind=ind,
                                        method=method)
        return grid_search

    def estimate_test(self, model, param_dict: dict) -> None:
        """method for estimating predictions on test data 
          parameters:
                    model - sklearn model object
                    param_dict - dict with paramters of the model that    
                                 should be used in estiamtion"""


        y_pred = model.predict(self.test_x)
        self._save_test_stats(y_pred=y_pred, param_dict=param_dict)

    def _save_test_stats(self, y_pred: pd.DataFrame, param_dict: dict) -> None:
        """method for saving stats from estimation on test data set in   summary_df
        """

        row_test = {"MSE_test": mean_squared_error(self.test_y, y_pred), 
                    "MAE_test": mean_absolute_error(self.test_y, y_pred), 
                    "R2_test": r2_score(self.test_y, y_pred)}
        for key in row_test:
            self.summary_df.loc[self.summary_df["params"]==json.dumps(param_dict), key] = row_test[key]
        
    def _save_model_stats(self, stat_dict: dict, ind: int, method) -> None:
        """save metrics of the training and validation estimation"""

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
                    "MAE_train": stat_dict[MAE_train][ind],
                    "MAE_valid": stat_dict[MAE_valid][ind],
                    "R2_valid": stat_dict[R2_valid][ind],
                    "R2_train": stat_dict[R2_train][ind], 
                    "params": json.dumps(stat_dict['params'][ind]),
                    "iteration": self.iteration,
                    "cv_n" : self.cv,
                    "test_size" : 1-self.train_share
                    }
        self.summary_df = pd.concat([self.summary_df, pd.DataFrame(stat_row, index=[self.df_index])])
        self.df_index +=1



    def create_method_object(self, method: str, **kwargs):
        """factory method for creating skelarn method object based on string value
        """

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





