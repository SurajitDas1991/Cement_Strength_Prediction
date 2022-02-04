from src import logger
from datetime import datetime
from src import utils
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

time = datetime.now().strftime("%d_%m_%Y")
import os


class Preprocessor:
    def __init__(self):
        self.logger = logger.get_logger(__name__, f"Preprocessing.txt_{time}")

    def standardScalingData(self, X):
        scalar = StandardScaler()
        X_scaled = scalar.fit_transform(X)
        return X_scaled

    def logTransformation(self, X):
        for column in X.columns:
            X[column] += 1
            X[column] = np.log(X[column])

        return X

    def impute_missing_values(self, data):
        self.logger.info("Imputing the data for missing values with KNN imputer")
        self.data = data
        try:
            imputer = KNNImputer(
                n_neighbors=3, weights="uniform", missing_values=np.nan
            )
            self.new_array = imputer.fit_transform(
                self.data
            )  # impute the missing values
            # convert the nd-array returned in the step above to a Dataframe
            self.new_data = pd.DataFrame(data=self.new_array, columns=self.data.columns)
            return self.new_data
        except Exception as e:
            self.logger.error(
                "Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  "
                + str(e)
            )
            raise Exception()

    def is_null_present(self, data):
        """This method checks whether there are null values present in the pandas Dataframe or not."""
        self.logger.info("Checking if there are any null values in the dataset")
        self.null_present = False
        self.cols_with_missing_values = []
        self.cols = data.columns
        try:
            self.null_counts = data.isna().sum()
            for i in range(len(self.null_counts)):
                if self.null_counts[i] > 0:
                    self.null_present = True
                    self.cols_with_missing_values.append(self.cols[i])
            if (
                self.null_present
            ):  # write the logs to see which columns have null values
                self.dataframe_with_null = pd.DataFrame()
                self.dataframe_with_null["columns"] = data.columns
                self.dataframe_with_null["missing values count"] = np.asarray(
                    data.isna().sum()
                )
                self.dataframe_with_null.to_csv(
                    os.path.join(utils.return_full_path("reports"), "null_values.csv")
                )  # storing the null column information to file
            return self.null_present, self.cols_with_missing_values
        except Exception as e:
            self.logger.error(
                "Exception occured in is_null_present method of the Preprocessor class. Exception message:  "
                + str(e)
            )
            raise Exception()

    def separate_label_feature(self, data, label_column_name):
        """Method Name: separate_label_feature Description: This method separates the features and a Label Coulmns."""
        self.logger.info(
            "Entered the separate_label_feature method of the Preprocessor class"
        )
        try:
            self.X = data.drop(
                labels=label_column_name, axis=1
            )  # drop the columns specified and separate the feature columns
            self.Y = data[label_column_name]  # Filter the Label columns
            self.logger.info("Label Separation Successful.")
            return self.X, self.Y
        except Exception as e:
            self.logger.error(
                "Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  "
                + str(e)
            )
            raise Exception()
