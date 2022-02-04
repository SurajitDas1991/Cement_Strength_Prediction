from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import textwrap
from datetime import datetime
from src import utils
from src import hyperparameter_tuning
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate

time = datetime.now().strftime("%d_%m_%Y")
import os
from src import logger
from src import file_operations


class ModelFinder:
    """
    This class shall  be used to find the model with best accuracy and AUC score.
    """

    def __init__(self):
        self.logger = logger.get_logger(__name__, f"bestmodelfinder.txt_{time}")
        self.hyperparametertuning = hyperparameter_tuning.GetHyperParametersForModels()
        self.linearReg = LinearRegression()
        self.RandomForestReg = RandomForestRegressor()
        self.models_objects = []

    def get_best_params_for_Random_Forest_Regressor(self, train_x, train_y):
        """
        Get the best params for random forest
        """
        self.logger.info(
            "Get the best params for random forest",
        )
        try:
            # initializing with different combination of parameters
            self.param_grid_Random_forest_Tree = {
                "n_estimators": [10, 20, 30, 40],
                "max_features": ["auto", "sqrt", "log2"],
                "min_samples_split": [2, 4, 8, 16],
                "bootstrap": [True, False],
            }

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(
                self.RandomForestReg,
                self.param_grid_Random_forest_Tree,
                verbose=3,
                cv=5,
            )
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.n_estimators = self.grid.best_params_["n_estimators"]
            self.max_features = self.grid.best_params_["max_features"]
            self.min_samples_split = self.grid.best_params_["min_samples_split"]
            self.bootstrap = self.grid.best_params_["bootstrap"]

            # creating a new model with the best parameters
            self.RandomForestRegressor = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                bootstrap=self.bootstrap,
            )

            # training the new models
            self.RandomForestRegressor.fit(train_x, train_y)
            self.models_objects.append(self.RandomForestRegressor)
            self.logger.info(
                "RandomForestReg best params: " + str(self.grid.best_params_)
            )
            return self.RandomForestRegressor
        except Exception as e:
            self.logger.error(
                "Exception while finding the best params for Random forest   " + str(e),
            )

            raise Exception()

    def get_best_params_for_linearReg(self, train_x, train_y):

        """
        Get the parameters for LinearReg Algorithm which give the best accuracy.
        """
        self.logger.info(
            "Get the best params for linear regression",
        )
        try:
            # initializing with different combination of parameters
            self.param_grid_linearReg = {
                "fit_intercept": [True, False],
                "normalize": [True, False],
                "copy_X": [True, False],
            }
            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(
                self.linearReg, self.param_grid_linearReg, verbose=3, cv=5
            )
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.fit_intercept = self.grid.best_params_["fit_intercept"]
            self.normalize = self.grid.best_params_["normalize"]
            self.copy_X = self.grid.best_params_["copy_X"]

            # creating a new model with the best parameters
            self.linReg = LinearRegression(
                fit_intercept=self.fit_intercept,
                normalize=self.normalize,
                copy_X=self.copy_X,
            )

            # training the mew model
            self.linReg.fit(train_x, train_y)
            self.models_objects.append(self.linReg)
            self.logger.info(
                "LinearRegression best params: " + str(self.grid.best_params_)
            )
            return self.linReg
        except Exception as e:
            self.logger.error(
                "Exception occured in get_best_params_for_linearReg method of the Model_Finder class. Exception message:  "
                + str(e)
            )

            raise Exception()

    def get_best_model(self, train_x, train_y, test_x, test_y):
        """
        Find out the Model which has the best AUC score.

        """
        self.logger.info(
            "Entered the get_best_model method of the Model_Finder class",
        )
        # create best model for Linear Regression
        try:

            self.LinearReg = self.get_best_params_for_linearReg(train_x, train_y)
            self.prediction_LinearReg = self.LinearReg.predict(
                test_x
            )  # Predictions using the LinearReg Model
            self.LinearReg_error = r2_score(test_y, self.prediction_LinearReg)

            # create best model for XGBoost
            self.randomForestReg = self.get_best_params_for_Random_Forest_Regressor(
                train_x, train_y
            )
            self.prediction_randomForestReg = self.randomForestReg.predict(
                test_x
            )  # Predictions using the randomForestReg Model
            self.prediction_randomForestReg_error = r2_score(
                test_y, self.prediction_randomForestReg
            )

            # comparing the two models
            if self.LinearReg_error < self.prediction_randomForestReg_error:
                return "RandomForestRegressor", self.randomForestReg
            else:
                return "LinearRegression", self.LinearReg

        except Exception as e:
            self.logger.error(
                "Exception occured in get_best_model method of the Model_Finder class. Exception message:  "
                + str(e)
            )

            raise Exception()

    def check_r2_squared_results(self, x_test, y_test, dict_model_results, i):
        # Model = model_fit(x_train, x_test, y_train, y_test)

        sns.set()
        R_square_num = []
        for name, m in dict_model_results.items():
            # Cross validation of the model
            model = m["model"]
            R_square = r2_score(y_test, model.predict(x_test))
            lst = [name, R_square]
            R_square_num.append(lst)
        df_results = pd.DataFrame(R_square_num, columns=["model", "R-Squared"])
        df_results.sort_values(by="R-Squared", ascending=False, inplace=True)
        df_results.reset_index(inplace=True, drop=True)
        df_results.to_csv(
            os.path.join(
                utils.PREDICTION_OUTPUT_FILE,
                f"R-SquaredResultComparison_cluster_{i}.csv",
            ),
            header=True,
        )

        # for i in range(2):
        #     R_square = r2_score(y_test, self.models_objects[i].predict(x_test))

        #     R_square_num.append(R_square)
        #     # R_square_num.append(r2_score(y_test, self.models_objects[i].predict(x_test)))
        plt.figure(figsize=(20, 10))
        plt.xlabel("R Square Score", fontsize=12)
        plt.ylabel("Model Type", fontsize=12)
        # plt.xticks(rotation=90, fontsize=12)
        plt.title("The R Square Score Comparsion")
        chart = sns.barplot(x="R-Squared", y="model", data=df_results)
        utils.wrap_labels(chart, 10)
        for container in chart.containers:
            chart.bar_label(container)
        plt.savefig(
            os.path.join(utils.VISUALIZATION_PATH, f"R Square Score_cluster_{i}.png")
        )
        # plt.show()

    def mean_square_error_for_models(self, x_test, y_test, dict_model_results, i):

        sns.set()

        rmse_num = []
        for name, m in dict_model_results.items():
            # Cross validation of the model
            model = m["model"]
            mse = mean_squared_error(y_test, model.predict(x_test),squared=False)
            lst = [name,mse]
            rmse_num.append(lst)
        df_results = pd.DataFrame(rmse_num, columns=["model", "Root Mean Squared Error"])
        df_results.sort_values(by="Root Mean Squared Error", ascending=True, inplace=True)
        df_results.reset_index(inplace=True, drop=True)
        df_results.to_csv(
            os.path.join(
                utils.PREDICTION_OUTPUT_FILE,
                f"RootMeanSquaredErrorResultComparison_cluster_{i}.csv",
            ),
            header=True,
        )

        plt.figure(figsize=(20, 10))
        plt.xlabel("Root mean_square_error", fontsize=12)
        plt.ylabel("Model Type", fontsize=12)
        # plt.xticks(rotation=90, fontsize=12)
        plt.title("The Root mean_square_error Comparison")
        chart = sns.barplot(x="Root Mean Squared Error", y="model", data=df_results)
        utils.wrap_labels(chart, 10)
        for container in chart.containers:
            chart.bar_label(container)
        plt.savefig(
            os.path.join(utils.VISUALIZATION_PATH, f"RootMeanSquareError_cluster_{i}.png")
        )

        # Get the best model name as per RMSE
        best_model_name = df_results.iloc[0]["model"]
        # models[best_model_name[0]]['model']
        model_name=df_results['model']
        best_model=None
        for name, m in dict_model_results.items():
            if name==best_model_name:
                best_model=m['model']
                break
        return best_model_name,best_model
        #Get the actual model


    def base_model_checks(self, train_x, train_y, x_test, y_test, i):
        # Create a dictionary with the model which will be tested
        models = {
            "StackingRegressor": {"model": StackingRegressor(estimators = [
            ("rf", RandomForestRegressor()),
            ("catreg", cb.CatBoostRegressor()),
        ],final_estimator=SVR()).fit(train_x, train_y)},
            "LinearRegression": {"model": LinearRegression().fit(train_x, train_y)},
            "KNN": {
                "model": KNeighborsRegressor(
                    **self.hyperparametertuning.hyperparams_KNN(train_x, train_y)
                ).fit(train_x, train_y)
            },
            "Ridge": {"model": Ridge(**self.hyperparametertuning.hyperparams_ridge(train_x,train_y)).fit(train_x, train_y)},
            "Lasso": {"model": Lasso(**self.hyperparametertuning.hyperparams_Lasso(train_x,train_y)).fit(train_x, train_y)},
            "SVR": {"model": SVR(**self.hyperparametertuning.hyperparams_SVR(train_x,train_y)).fit(train_x, train_y)},

            "Catboost": {"model": cb.CatBoostRegressor(**self.hyperparametertuning.hyperparams_catboost(train_x,train_y)).fit(train_x, train_y)},
            "RandomForest": {"model": RandomForestRegressor(**self.hyperparametertuning.hyperparams_random_forest(train_x,train_y)).fit(train_x, train_y)},
            "GradientBoost": {"model": GradientBoostingRegressor(**self.hyperparametertuning.hyperparams_gradientboost(train_x,train_y)).fit(train_x, train_y)},
            "XGBoost": {"model": xgb.XGBRegressor(**self.hyperparametertuning.hyperparams_xgboost(train_x,train_y)).fit(train_x, train_y)},
            "LightGBM": {"model": lgb.LGBMRegressor(**self.hyperparametertuning.hyperparams_lightgbm(train_x,train_y)).fit(train_x, train_y)},
            "AdaBoost": {"model": AdaBoostRegressor(**self.hyperparametertuning.hyperparams_adaboost(train_x,train_y)).fit(train_x, train_y)},
             "ExtraTrees": {"model": ExtraTreesRegressor(**self.hyperparametertuning.hyperparams_extratrees(train_x,train_y)).fit(train_x, train_y)},
        }

        # Use the 10-fold cross validation for each model
        # to get the mean validation accuracy and the mean training time
        for name, m in models.items():
            # Cross validation of the model
            model = m["model"]
            result = cross_validate(model, train_x, train_y, cv=10)

            # Mean accuracy and mean training time
            mean_val_accuracy = round(
                sum(result["test_score"]) / len(result["test_score"]), 4
            )
            mean_fit_time = round(sum(result["fit_time"]) / len(result["fit_time"]), 4)

            # Add the result to the dictionary witht he models
            m["val_accuracy"] = mean_val_accuracy
            m["Training time (sec)"] = mean_fit_time

            # Display the result
            print(
                f"{name:27} For cluster {i} mean accuracy using 10-fold cross validation: {mean_val_accuracy*100:.2f}% - mean training time {mean_fit_time} sec"
            )

        # Create a DataFrame with the results
        models_result = []

        for name, v in models.items():
            lst = [name, v["val_accuracy"], v["Training time (sec)"]]
            models_result.append(lst)

        df_results = pd.DataFrame(
            models_result,
            columns=["model", "val_accuracy", "Training time (sec)"],
        )
        df_results.sort_values(
            by="val_accuracy", ascending=False, inplace=True
        )
        df_results.reset_index(inplace=True, drop=True)
        df_results.to_csv(
            os.path.join(
                utils.PREDICTION_OUTPUT_FILE,
                f"CrossValidationByvalAccuracyResults_cluster_{i}.csv",
            ),
            header=True,
        )

        plt.figure(figsize=(30, 5))
        chart = sns.barplot(x="model", y="val_accuracy", data=df_results)
        utils.wrap_labels(chart, 10)
        for container in chart.containers:
            chart.bar_label(container)
        plt.title(
            "Mean Validation Accuracy for each Model\ny-axis between 0.6 and 1.0",
            fontsize=12,
        )
        plt.ylim(0.6, 1)
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("val_accuracy", fontsize=12)
        # plt.xticks(rotation=90, fontsize=12)
        plt.savefig(
            os.path.join(
                utils.VISUALIZATION_PATH, f"CV_validationaccuracy_cluster_{i}.png"
            )
        )
        # plt.show()

        self.check_r2_squared_results(x_test, y_test, models, i)
        return self.mean_square_error_for_models(x_test, y_test, models, i)
