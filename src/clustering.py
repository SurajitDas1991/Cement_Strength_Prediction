import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from datetime import datetime
from src import utils
import pandas as pd

time = datetime.now().strftime("%d_%m_%Y")
import os
from src import logger
from src import file_operations


class KMeansClustering:
    def __init__(self):
        self.logger = logger.get_logger(__name__, f"Clustering.txt_{time}")

    def elbow_plot(self, data):
        """
        This method saves the plot to decide the optimum number of clusters to the file.
        """
        self.logger.info("Getting the best number of clusters for the data provided")
        wcss = []  # initializing an empty list
        try:
            for i in range(1, 11):
                kmeans = KMeans(
                    n_clusters=i, init="k-means++", random_state=42
                )  # initializing the KMeans object
                kmeans.fit(data)  # fitting the data to the KMeans Algorithm
                wcss.append(kmeans.inertia_)
            plt.plot(
                range(1, 11), wcss
            )  # creating the graph between WCSS and the number of clusters
            plt.title("The Elbow Method")
            plt.xlabel("Number of clusters")
            plt.ylabel("WCSS")
            # plt.show()
            plt.savefig(
                os.path.join(
                    utils.return_full_path("visualizations"), "K-Means_Elbow.PNG"
                )
            )
            self.kn = KneeLocator(
                range(1, 11), wcss, curve="convex", direction="decreasing"
            )
            self.logger.info(
                "The optimum number of clusters is: "
                + str(self.kn.knee)
                + " . Exited the elbow_plot method of the KMeansClustering class"
            )
            return self.kn.knee

        except Exception as e:
            self.logger.error(
                "Exception occured in elbow_plot method of the KMeansClustering class. Exception message:  "
                + str(e)
            )
            raise Exception()

    def create_clusters(self, data, number_of_clusters):
        """
        Create a new dataframe consisting of the cluster information.
        """
        self.logger.info(
            "Entered the create_clusters method of the KMeansClustering class"
        )
        self.data = data
        try:
            self.kmeans = KMeans(
                n_clusters=number_of_clusters, init="k-means++", random_state=42
            )

            self.y_kmeans = self.kmeans.fit_predict(data)  #  divide data into clusters

            self.file_op = file_operations.FileOperations()
            self.save_model = self.file_op.save_model(
                self.kmeans, "KMeans"
            )  # saving the KMeans model to directory passing 'Model' as the functions need three parameters

            self.data[
                "Cluster"
            ] = (
                self.y_kmeans
            )  # create a new column in dataset for storing the cluster information
            self.logger.info(
                "Succesfully created "
                + str(self.kn.knee)
                + "clusters. Exited the create_clusters method of the KMeansClustering class"
            )
            return self.data
        except Exception as e:
            self.logger.error(
                "Exception occured in create_clusters method of the KMeansClustering class. Exception message:  "
                + str(e)
            )
            raise Exception()
