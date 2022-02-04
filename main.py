from src import logger
from datetime import datetime
from src import utils
from src import train
from src import predict

time = datetime.now().strftime("%d_%m_%Y")
import pandas as pd

# First check if the data provided is in the agreed format or not


def predict_result():
    ls = [[540, 0, 0, 162, 2.5, 1040, 676, 28]]

    df_pred = pd.DataFrame(
        ls,
        columns=[
            "Cement",
            "Blast Furnace Slag _component_2",
            "Fly Ash _component_3",
            "Water_component_4",
            "Superplasticizer_component_5",
            "Coarse Aggregate_component_6",
            "Fine Aggregate_component_7",
            "Age_day",
        ],
    )
    print(df_pred.shape)
    return df_pred


if __name__ == "__main__":
    utils.prepare_for_training()
    train_model = train.TrainModel(utils.RAW_DATA_FOLDER)
    train_model.start_training()
    train_model.train_model()

    # predict_data=predict.PredictFromData(predict_result())
    # predict_data.predict()
    # print(utils.return_full_path("asd", "jdfsjf"))
