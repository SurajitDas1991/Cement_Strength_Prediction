from src import logger
from datetime import datetime
from src import utils
import pandas as pd

time = datetime.now().strftime("%d_%m_%Y")
import os


class LoadData:
    def __init__(self):
        self.logger = logger.get_logger(__name__, f"LoadFinalData.txt_{time}")
        self.training_file = os.path.join(
            utils.FINAL_INPUT_FILE_FROM_DB, "InputFile.csv"
        )

    def get_data(self):
        self.logger.info("Getting the data which is the output from the db")
        try:
            self.data = pd.read_csv(self.training_file)  # reading the data file
            return self.data
        except Exception as e:
            pass
