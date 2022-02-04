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
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

time = datetime.now().strftime("%d_%m_%Y")


class TrainModel:
    def __init__(self, path):
        self.logger = logger.get_logger(__name__, f"trainmodel.txt_{time}")
        self.raw_data = raw_data_validation.RawDataValidation(path)
        self.db_operation = database_operation.DbOperation()

    def start_training(self):
        try:
            self.logger.info("Validation of raw files started ")
            # Get values from training schema
            (
                length_of_date_stamp_in_file,
                length_of_time_stamp_in_file,
                column_names,
                number_of_columns,
            ) = self.raw_data.values_from_schema()
            # Get the regex pattern for checking the file name
            regex = self.raw_data.manual_regex_Creation()
            # Check the filename pattern via regex
            self.raw_data.validate_raw_file_name(
                regex, length_of_date_stamp_in_file, length_of_time_stamp_in_file
            )
            # Validate the column length
            self.raw_data.validate_column_length(number_of_columns)
            # validate if any column has all the values empty
            self.raw_data.validate_missing_values_in_whole_column()
            self.logger.info("Validation of raw files completed")
            self.logger.info(
                "Creating Training_Database and tables on the basis of given schema"
            )
            self.db_operation.create_table_db("Training", column_names)
            self.logger.info("Table creation Completed")
            self.logger.info("Insertion of Data into Table started")
            # insert csv files in the table
            self.db_operation.insert_into_table_good_data("Training")
            # Delete the good data folder after loading files in table
            self.raw_data.delete_existing_good_data_training_folder()
            self.logger.info("Good data folder deleted")
            self.logger.info("Moving bad files to Archive and deleting Bad_Data folder")
            self.raw_data.move_bad_files_to_archived()
            self.logger.info("Bad files moved to archive!! Bad folder Deleted")
            self.logger.info("Validation Operation completed")
            self.logger.info("Extracting csv file from table")
            self.db_operation.select_data_from_table_into_csv("Training")
        except Exception as e:
            self.logger.error(f"Error in the training process - str{e}")

    def train_model(self):
        self.logger.info("Start of training..")
        try:
            # Data from source
            data_getter = load_data.LoadData()
            data = data_getter.get_data()

            # Preprocessing

            preprocessor = preprocessing.Preprocessor()

            # check if missing values are present in the dataset
            is_null_present, cols_with_missing_values = preprocessor.is_null_present(
                data
            )

            # if missing values are there, replace them appropriately.
            if is_null_present:
                data = preprocessor.impute_missing_values(
                    data
                )  # missing value imputation

            # get encoded values for categorical data

            # data = preprocessor.encodeCategoricalValues(data)

            # create separate features and labels
            X, Y = preprocessor.separate_label_feature(
                data, label_column_name="Concrete_compressive _strength"
            )
            # drop the columns obtained above
            # X=preprocessor.remove_columns(X,cols_to_drop)

            X = preprocessor.logTransformation(X)
            """ Applying the clustering approach"""

            kmeans = clustering.KMeansClustering()  # object initialization.
            number_of_clusters = kmeans.elbow_plot(
                X
            )  #  using the elbow plot to find the number of optimum clusters

            # Divide the data into clusters
            X = kmeans.create_clusters(X, number_of_clusters)

            # create a new column in the dataset consisting of the corresponding cluster assignments.
            X["Labels"] = Y

            # getting the unique clusters from our dataset
            list_of_clusters = X["Cluster"].unique()

            """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""

            for i in list_of_clusters:
                cluster_data = X[X["Cluster"] == i]  # filter the data for one cluster

                # Prepare the feature and Label columns
                cluster_features = cluster_data.drop(["Labels", "Cluster"], axis=1)
                cluster_label = cluster_data["Labels"]

                # splitting the data into training and test set for each cluster one by one
                x_train, x_test, y_train, y_test = train_test_split(
                    cluster_features, cluster_label, test_size=1 / 3, random_state=36
                )

                x_train_scaled = preprocessor.standardScalingData(x_train)
                x_test_scaled = preprocessor.standardScalingData(x_test)

                model_finder = model_tuner.ModelFinder()  # object initialization

                # getting the best model for each of the clusters
                # best_model_name, best_model = model_finder.get_best_model(
                #     x_train_scaled, y_train, x_test_scaled, y_test
                # )

                best_model_name, best_model=model_finder.base_model_checks(
                    x_train_scaled, y_train, x_test_scaled, y_test, i
                )
                # model_finder.check_model_results(x_test_scaled, y_test)

                # saving the best model to the directory.
                file_op = file_operations.FileOperations()
                save_model = file_op.save_model(best_model, best_model_name + str(i))
                self.logger.info(f"Completed training for cluster {i}")

            # logging the successful Training
            self.logger.info("Successful End of Training")

        except Exception:
            # logging the unsuccessful Training
            self.logger.error("Training did not complete succesfully")
            raise Exception
