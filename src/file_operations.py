import pickle
import os
import shutil
from datetime import datetime
from src import utils
import pandas as pd

time = datetime.now().strftime("%d_%m_%Y")
import os
from src import logger


class FileOperations:
    def __init__(self):
        self.model_directory = utils.MODELS_PATH
        self.logger = logger.get_logger(__name__, f"FileOperations.txt_{time}")

    def save_model(self, model, filename):
        """
        Saves the model file to directory
        """
        self.logger.info("Saving model")
        try:
            path = os.path.join(
                self.model_directory, filename
            )  # create seperate directory for each cluster
            if os.path.isdir(
                path
            ):  # remove previously existing models for each clusters
                shutil.rmtree(self.model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path)  #
            with open(path + "/" + filename + ".sav", "wb") as f:
                pickle.dump(model, f)  # save the model to file
            self.logger.info("Model File " + filename + " saved.")
            return "success"
        except Exception as e:
            self.logger.error("Exception while saving the model" + str(e))
            raise Exception()

    def load_model(self, filename):
        """
        Loads the model file in directory
        """
        self.logger.info("Loading model from the directory")
        try:
            with open(
                os.path.join(self.model_directory, filename, filename + "." + "sav"),
                "rb",
            ) as f:
                self.logger.info("Model File " + filename + " loaded.")
                return pickle.load(f)
        except Exception as e:
            self.logger.error(
                "Exception occured in load_model method of the Model_Finder class. Exception message:  "
                + str(e)
            )
            raise Exception()

    def find_correct_model_file(self, cluster_number):
        """
        Select the correct model based on cluster number
        """
        self.logger.info("Get the correct model file based on the cluster number")
        try:
            self.cluster_number = cluster_number
            self.folder_name = self.model_directory
            self.list_of_model_files = []
            self.list_of_files = os.listdir(self.folder_name)
            for self.file in self.list_of_files:
                try:
                    if self.file.index(str(self.cluster_number)) != -1:
                        self.model_name = self.file
                except:
                    continue
            self.model_name = self.model_name.split(".")[0]
            return self.model_name
        except Exception as e:
            self.logger.error(
                "Unable to get the correct model based on cluster" + str(e)
            )
            raise Exception()
