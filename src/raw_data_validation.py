import sqlite3
from datetime import datetime
from os import listdir
import os
import re
import json
import shutil
import pandas as pd

from src.utils import *
from src import logger

time = datetime.now().strftime("%d_%m_%Y")
# Need to validate data from users
class RawDataValidation:
    def __init__(self, path) -> None:
        self.batch_directory = path
        self.schema_path = return_full_path("schema_training.json")
        self.logger = logger.get_logger(__name__, f"RawDataValidation.txt_{time}")

    def values_from_schema(self):
        """
        Read the schema file in json format. This contains the details regarding the standard column format available in the dataset
        """
        length_of_date_stamp_in_file = 0
        length_of_time_stamp_in_file = 0
        column_names = 0
        number_of_columns = 0
        try:

            with open(self.schema_path, "r") as f:
                dic = json.load(f)
                f.close()
            pattern = dic["SampleFileName"]
            length_of_date_stamp_in_file = dic["LengthOfDateStampInFile"]
            length_of_time_stamp_in_file = dic["LengthOfTimeStampInFile"]
            column_names = dic["ColName"]
            number_of_columns = dic["NumberofColumns"]
            self.logger.info("Completed processing of training json file")
            return (
                length_of_date_stamp_in_file,
                length_of_time_stamp_in_file,
                column_names,
                number_of_columns,
            )
        except Exception as e:
            self.logger.error(
                f"Error while processing the training schema file - str{e}"
            )
        return (
            length_of_date_stamp_in_file,
            length_of_time_stamp_in_file,
            column_names,
            number_of_columns,
        )

    def manual_regex_Creation(self):
        """
        This method contains a manually defined regex based on the "FileName" given in "Schema" file.This Regex is used to validate the filename of the training data.
        """
        regex = "['cement_strength']+['\_'']+[\d_]+[\d]+\.csv"
        return regex

    def create_good_bad_raw_data_folder(self):
        """
        Creates folders for placing the good and bad processed data files
        """
        try:
            if not os.path.isdir(GOOD_RAW_DATA_FOLDER):
                os.makedirs(GOOD_RAW_DATA_FOLDER)
            if not os.path.isdir(BAD_RAW_DATA_FOLDER):
                os.makedirs(BAD_RAW_DATA_FOLDER)
            self.logger.info("Created folders for processed good and bad raw files")
        except Exception as e:
            self.logger.error(
                f"Error in creating folders for separating good and bad raw data folders - str{e}"
            )

    def delete_existing_good_data_training_folder(self):
        """This method deletes the directory made to store the good raw data files."""
        try:
            if os.path.isdir(GOOD_RAW_DATA_FOLDER):
                shutil.rmtree(GOOD_RAW_DATA_FOLDER)
                self.logger.info("Existing Good raw data folder deleted")
        except Exception as e:
            self.logger.error(
                f"Error in deleting existing good raw data folder - str{e}"
            )

    def delete_existing_bad_data_training_folder(self):
        """This method deletes the directory made to store the bad raw data files."""
        try:
            if os.path.isdir(BAD_RAW_DATA_FOLDER):
                shutil.rmtree(BAD_RAW_DATA_FOLDER)
                self.logger.info("Existing Bad raw data folder deleted")
        except Exception as e:
            self.logger.error(
                f"Error in deleting existing bad raw data folder - str{e}"
            )

    def move_bad_files_to_archived(self):
        """
        This method deletes the directory made to store the Bad Data after moving the data in an archive folder. Archived data is sent back to the client.
        """
        now = datetime.now()
        date = now.date()
        time = now.strftime("%H%M%S")
        try:
            source = BAD_RAW_DATA_FOLDER
            if os.path.isdir(source):
                path = ARCHIVED_RAW_DATA_FOLDER
                if not os.path.isdir(path):
                    os.makedirs(path)
                dest = os.path.join(
                    ARCHIVED_RAW_DATA_FOLDER, "BAD_DATA_", str(date), "_", str(time)
                )
                # dest = ARCHIVED_RAW_DATA_FOLDER + str(date)+"_"+str(time)
                if not os.path.isdir(dest):
                    os.makedirs(dest)
                files = os.listdir(source)
                for f in files:
                    if f not in os.listdir(dest):
                        shutil.move(source + f, dest)
                self.logger.info("Bad raw files moved to archive")
                self.delete_existing_bad_data_training_folder()
        except Exception as e:
            self.logger.error(f"Error while moving bad raw files to archive - str{e}")

    def validate_raw_file_name(
        self, regex, length_of_date_stamp_in_file, length_of_time_stamp_in_file
    ):
        """
        This function validates the name of the training csv files as per given name in the schema! Regex pattern is used to do the validation.If name format do not match the file is moved to Bad Raw Data folder else in Good raw data.
        """
        self.delete_existing_bad_data_training_folder()
        self.delete_existing_good_data_training_folder()
        only_files = [f for f in listdir(self.batch_directory)]
        try:
            self.create_good_bad_raw_data_folder()
            for filename in only_files:
                if re.match(regex, filename):
                    splitAtDot = re.split(".csv", filename)
                    splitAtDot = re.split("_", splitAtDot[0])
                    if len(splitAtDot[2]) == length_of_date_stamp_in_file:
                        if len(splitAtDot[3]) == length_of_time_stamp_in_file:
                            shutil.copy(
                                os.path.join(RAW_DATA_FOLDER, filename),
                                GOOD_RAW_DATA_FOLDER,
                            )
                            self.logger.info(
                                f"Valid File name!! File moved to GoodRaw Folder {filename}"
                            )
                        else:
                            shutil.copy(
                                os.path.join(RAW_DATA_FOLDER, filename),
                                BAD_RAW_DATA_FOLDER,
                            )
                            self.logger.info(
                                f"Invalid File name!! File moved to Bad raw folder {filename}"
                            )
                    else:
                        shutil.copy(
                            os.path.join(RAW_DATA_FOLDER, filename), BAD_RAW_DATA_FOLDER
                        )
                        self.logger.info(
                            f"Invalid File name!! File moved to Bad raw folder {filename}"
                        )
                else:
                    shutil.copy(
                        os.path.join(RAW_DATA_FOLDER, filename), BAD_RAW_DATA_FOLDER
                    )
                    self.logger.info(
                        f"Invalid File name!! File moved to Bad raw folder {filename}"
                    )

        except Exception as e:
            self.logger.error(
                f"Error occured while validating the raw data files - str{e}"
            )

    def validate_column_length(self, number_of_columns):
        """
        This function validates the number of columns in the csv files.It is should be same as given in the schema file.
        If not same file is not suitable for processing and thus is moved to Bad Raw Data folder. If the column number matches, file is kept in Good Raw Data for processing.The csv file is missing the first column name, this function changes the missing name to "Wafer".
        """
        try:
            self.logger.info("Column Length Validation Started")
            for file in listdir(GOOD_RAW_DATA_FOLDER):
                csv = pd.read_csv(os.path.join(GOOD_RAW_DATA_FOLDER, file))
                if csv.shape[1] == number_of_columns:
                    pass
                else:
                    shutil.move(
                        os.path.join(GOOD_RAW_DATA_FOLDER, file), BAD_RAW_DATA_FOLDER
                    )
                    self.logger.info(
                        f"Invalid Column Length for the file!! File moved to Bad Raw Folder {file}"
                    )
            self.logger.info("Column Length Validation Completed")
        except Exception as e:
            self.logger.error(f"Error occured while validating column length - str{e}")

    def validate_missing_values_in_whole_column(self):
        """
        This function validates if any column in the csv file has all values missing.If all the values are missing, the file is not suitable for processing.Such files are moved to bad raw data.
        """
        try:
            self.logger.info("Missing Values Validation Started")
            for file in listdir(GOOD_RAW_DATA_FOLDER):
                csv = pd.read_csv(os.path.join(GOOD_RAW_DATA_FOLDER, file))
                count = 0
                for columns in csv:
                    if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                        count += 1
                        shutil.move(
                            os.path.join(GOOD_RAW_DATA_FOLDER, file),
                            BAD_RAW_DATA_FOLDER,
                        )
                        self.logger.info(
                            f"Invalid Column Length for the file .File moved to Bad Raw Folder {file}"
                        )
                        break
                if count == 0:
                    csv.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                    csv.to_csv(
                        os.path.join(GOOD_RAW_DATA_FOLDER, file),
                        index=None,
                        header=True,
                    )
        except Exception as e:
            self.logger.error(
                f"Error occured while validating data for missing values in whole column - str{e}"
            )
