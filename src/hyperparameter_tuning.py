from sklearn.model_selection import GridSearchCV, RepeatedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import catboost as cb
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from src import logger
from datetime import datetime

time = datetime.now().strftime("%d_%m_%Y")


class GetHyperParametersForModels:
    def __init__(self):
        self.logger = logger.get_logger(__name__, f"HyperparameterTuning.txt_{time}")

    def hyperparams_random_forest(self, X, y):

        # Define the hyperparameter configuration space
        rf_params = {
            "n_estimators": [10, 50,100],
            "max_features": ["sqrt", 0.5],
            "max_depth": [15, 20, 30,100],
            "min_samples_leaf": [2,8,16],
            "bootstrap": [True, False],
            "criterion": ["squared_error", "absolute_error"],
        }
        clf = RandomForestRegressor(random_state=0)
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        grid = GridSearchCV(
            clf,
            rf_params,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=2,
        )
        self.logger.info("Tuning Started for Random Forest")
        grid.fit(X, y)
        return grid.best_params_

    def hyperparams_SVR(self, X, y):
        # Define the hyperparameter configuration space
        rf_params = {
            "C": [1, 10, 100],
            "kernel": ["poly", "rbf", "sigmoid"],
            "epsilon": [0.01, 0.1, 1],
        }
        clf = SVR()
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        grid = GridSearchCV(
            clf,
            rf_params,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=2,
        )
        self.logger.info("Tuning Started for Support Vector Regression")
        grid.fit(X, y)
        return grid.best_params_

    def hyperparams_KNN(self, X, y):
        # Define the hyperparameter configuration space
        rf_params = {"n_neighbors": [2, 3, 5, 10,15]}
        clf = KNeighborsRegressor()
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        grid = GridSearchCV(
            clf,
            rf_params,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=2,
        )
        self.logger.info("Tuning Started for KNN")
        grid.fit(X, y)
        return grid.best_params_

    def hyperparams_Lasso(self, X, y):
        # Define the hyperparameter configuration space
        rf_params = {"alpha": [0.001, 0.01, 0.1, 1, 100, 1000]}
        clf = Lasso()
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        grid = GridSearchCV(
            clf,
            rf_params,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=2,
        )
        self.logger.info("Tuning Started for Lasso regression")
        grid.fit(X, y)
        return grid.best_params_

    def hyperparams_ridge(self, X, y):
        # Define the hyperparameter configuration space
        rf_params = {"alpha": [0.001, 0.01, 0.1, 1, 100, 1000]}
        clf = Ridge()
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        grid = GridSearchCV(
            clf,
            rf_params,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=2,
        )
        self.logger.info("Tuning Started for Ridge regression")
        grid.fit(X, y)
        return grid.best_params_

    def hyperparams_adaboost(self, X, y):
        # Define the hyperparameter configuration space
        rf_params = {
            "n_estimators": np.arange(10, 300, 10),
            "learning_rate": [0.01, 0.1, 0.5,1],
        }
        clf = AdaBoostRegressor()
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        grid = GridSearchCV(
            clf,
            rf_params,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=2,
        )
        self.logger.info("Tuning Started for Adaboost")
        grid.fit(X, y)
        return grid.best_params_

    def hyperparams_gradientboost(self, X, y):
        rf_params = {
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.9, 0.4, 0.1],
            "n_estimators": [100, 500,1000],
            "max_depth": [4, 6, 8],
        }
        clf = GradientBoostingRegressor()
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        grid = GridSearchCV(
            clf,
            rf_params,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=2,
        )
        self.logger.info("Tuning Started for GradientBoost")
        grid.fit(X, y)
        return grid.best_params_

    def hyperparams_extratrees(self, X, y):
        rf_params = {
            "n_estimators": [10, 50, 100],
            "min_samples_split": [2, 6, 10],
            "max_depth": [2,4,8],
        }
        clf = ExtraTreesRegressor()
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        grid = GridSearchCV(
            clf,
            rf_params,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=2,
        )
        self.logger.info("Tuning Started for ExtraTreesRegressor")
        grid.fit(X, y)
        return grid.best_params_

    def hyperparams_xgboost(self, X, y):
        rf_params = {
            "n_estimators": [400, 500],
            "colsample_bytree": [0.7, 0.2,1],
            "max_depth": [15, 20,40],
            "reg_alpha": [1.1,0.3],
            "reg_lambda": [1.1,0.5],
            # "eta": [0.1,0.03],
            "subsample": [0.7,0.1],
        }
        clf = xgb.XGBRegressor()
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        grid = GridSearchCV(
            clf,
            rf_params,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=2,
        )
        self.logger.info("Tuning Started for XGBoost")
        grid.fit(X, y)
        return grid.best_params_

    def hyperparams_lightgbm(self, X, y):
        rf_params = {
            "learning_rate": [0.01, 0.01, 0.03],
            "boosting_type": ["gbdt", "dart", "goss"],
            "objective": ["regression"],
            "metric": ["mae","mse"],
            "num_leaves": [20, 40],
            "reg_alpha": [0.01, 0.03],
            "max_depth": [10, 20],
        }
        clf = lgb.LGBMRegressor()
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        grid = GridSearchCV(
            clf,
            rf_params,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=2,
        )
        self.logger.info("Tuning Started for LightGBM")
        grid.fit(X, y)
        return grid.best_params_

    def hyperparams_catboost(self, X, y):
        rf_params = {
            "iterations": [100, 150],
            "learning_rate": [0.03, 0.1],
            "depth": [2, 4, 6],
            "l2_leaf_reg": [0.2, 1, 3],
        }
        clf = cb.CatBoostRegressor()
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        grid = GridSearchCV(
            clf,
            rf_params,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=2,
        )
        self.logger.info("Tuning Started for CatBoost")
        grid.fit(X, y)
        return grid.best_params_

    def hyperparams_stacking(self, X, y):
        rf_params = {"final_estimator__C": [0, 1]}
        estimators = [
            ("rf", RandomForestRegressor(n_estimators=10, random_state=42)),
            ("adareg", AdaBoostRegressor()),
        ]
        clf = StackingRegressor(estimators, final_estimator=SVR())
        cv = RepeatedKFold(n_splits=3, n_repeats=3, random_state=0)
        grid = GridSearchCV(
            clf,
            rf_params,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=2,
        )
        self.logger.info("Tuning Started for Stacking")
        grid.fit(X, y)
        return grid.best_params_
