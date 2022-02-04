from src import logger
from datetime import datetime
from src import utils
from src import train

time = datetime.now().strftime("%d_%m_%Y")
import sqlite3
import os
from os import listdir
import csv
import shutil


class DbOperation:
    def __init__(self):
        self.logger = logger.get_logger(__name__, f"dboperation.txt_{time}")
        self.database_path = utils.DATABASE_FOLDER

    def database_connection(self, database_name):
        """
        Creates the database with the given name and if Database already exists then opens the connection to the DB.
        """
        conn = None
        try:
            conn = sqlite3.connect(
                os.path.join(self.database_path, database_name + "." + "db")
            )
            self.logger.info(f"Opened database successfully{database_name}")

        except Exception as e:
            self.logger.error(f"Error while connecting to database{database_name} {e}")
        return conn

    def create_table_db(self, database_name, column_names):
        """
        Creates a table in the given database which will be used to insert the Good data after raw data validation.
        """
        try:
            conn = self.database_connection(database_name)
            c = conn.cursor()
            c.execute(
                "SELECT count(name)  FROM sqlite_master WHERE type = 'table'AND name = 'Good_Raw_Data'"
            )
            if c.fetchone()[0] == 1:
                conn.close()

                self.logger.info("Tables created successfully!!")

            else:

                for key in column_names.keys():
                    type = column_names[key]

                    # in try block we check if the table exists, if yes then add columns to the table
                    # else in catch block we will create the table
                    try:
                        conn.execute(
                            'ALTER TABLE Good_Raw_Data ADD COLUMN "{column_name}" {dataType}'.format(
                                column_name=key, dataType=type
                            )
                        )
                    except:
                        conn.execute(
                            "CREATE TABLE  Good_Raw_Data ({column_name} {dataType})".format(
                                column_name=key, dataType=type
                            )
                        )
                self.logger.info(f"Tables created successfully!!")
                conn.close()

                self.logger.info(f"Closed database successfully {database_name}")
        except Exception as e:
            self.logger.error(
                f"Error in creation / modifying database {database_name} {e}"
            )

    def insert_into_table_good_data(self, database):
        """
        Method inserts the Good data files from the Good_Raw folder into the created table in Database
        """
        conn = self.database_connection(database)

        only_files = [f for f in listdir(utils.GOOD_RAW_DATA_FOLDER)]
        for file in only_files:
            try:
                with open(os.path.join(utils.GOOD_RAW_DATA_FOLDER, file), "r") as f:
                    next(f)
                    reader = csv.reader(f, delimiter="\n")
                    for line in enumerate(reader):
                        for list_ in line[1]:
                            try:
                                conn.execute(
                                    "INSERT INTO Good_Raw_Data values ({values})".format(
                                        values=(list_)
                                    )
                                )
                                self.logger.info(f"File loaded successfully {file}")
                                conn.commit()
                            except Exception as e:
                                raise e

            except Exception as e:

                conn.rollback()
                self.logger.error(f"Error while creating table: - str{e}")
                shutil.move(
                    os.path.join(utils.GOOD_RAW_DATA_FOLDER, file),
                    utils.BAD_RAW_DATA_FOLDER,
                )
                self.logger.info(f"File Moved Successfully {file}")
                conn.close()

        conn.close()

    def select_data_from_table_into_csv(self, database):
        """
        Method exports the data in GoodData table as a CSV file. in a given location.
        """
        self.fileName = "InputFile.csv"
        try:
            conn = self.database_connection(database)
            sqlSelect = "SELECT *  FROM Good_Raw_Data"
            cursor = conn.cursor()

            cursor.execute(sqlSelect)

            results = cursor.fetchall()
            # Get the headers of the csv file
            headers = [i[0] for i in cursor.description]

            # Make the CSV ouput directory
            if not os.path.isdir(utils.FINAL_INPUT_FILE_FROM_DB):
                os.makedirs(utils.FINAL_INPUT_FILE_FROM_DB)

            # Open CSV file for writing.
            csvFile = csv.writer(
                open(
                    os.path.join(utils.FINAL_INPUT_FILE_FROM_DB, self.fileName),
                    "w",
                    newline="",
                ),
                delimiter=",",
                lineterminator="\r\n",
                quoting=csv.QUOTE_ALL,
                escapechar="\\",
            )

            # Add the headers and data to the CSV file.
            csvFile.writerow(headers)
            csvFile.writerows(results)

            self.logger.info("File exported successfully!!!")

        except Exception as e:
            self.logger.error(f"File exporting failed. Error : {e}")
