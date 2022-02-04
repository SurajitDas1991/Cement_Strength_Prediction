from src import logger
from datetime import datetime
from src import utils
from src import raw_data_validation
from src import database_operation
from src import file_operations
from src import clustering
from src import model_tuner
from src import preprocessing
from src import load_data

time = datetime.now().strftime("%d_%m_%Y")
import os
import pandas as pd
import numpy as np


class PredictFromData:
    def __init__(self, data):
        self.df = data

    def predict(self):

        # Preprocessing
        preprocessor = preprocessing.Preprocessor()

        # check if missing values are present in the dataset
        is_null_present, cols_with_missing_values = preprocessor.is_null_present(
            self.df
        )

        # if missing values are there, replace them appropriately.
        if is_null_present:
            data = preprocessor.impute_missing_values(
                self.df
            )  # missing value imputation

        self.df = preprocessor.logTransformation(self.df)

        data_scaled = pd.DataFrame(
            preprocessor.standardScalingData(self.df), columns=self.df.columns
        )

        file_op = file_operations.FileOperations()
        kmeans = file_op.load_model("KMeans")

        clusters = kmeans.predict(data_scaled)
        data_scaled["clusters"] = clusters
        clusters = data_scaled["clusters"].unique()
        result = []
        for i in clusters:
            cluster_data = data_scaled[data_scaled["clusters"] == i]
            cluster_data = cluster_data.drop(["clusters"], axis=1)
            model_name = file_op.find_correct_model_file(i)
            model = file_op.load_model(model_name)
            for val in model.predict(cluster_data.values):
                result.append(val)
        result = pd.DataFrame(result, columns=["Predictions"])
        result.to_csv(
            os.path.join(utils.PREDICTION_OUTPUT_FILE, "Predictions.csv"), header=True
        )
