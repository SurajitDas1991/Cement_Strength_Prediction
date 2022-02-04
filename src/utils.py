import os
from pathlib import Path
import textwrap
import shutil



def prepare_for_training():
    delete_create_existing_folders(VISUALIZATION_PATH)
    delete_create_existing_folders(PREDICTION_OUTPUT_FILE)
    delete_create_existing_folders(MODELS_PATH)
    #delete_create_existing_folders(LOGS_PATH)

def delete_create_existing_folders(path):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            if not os.path.isdir(path):
                os.makedirs(path)

        except Exception as e:
            pass


def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(
            textwrap.fill(text, width=width, break_long_words=break_long_words)
        )
    ax.set_xticklabels(labels, rotation=0)


def return_full_path(*path):
    cwd = os.path.abspath(os.getcwd())
    temp_path = ""
    for i in path:
        cwd = os.path.join(cwd, i)
    return cwd

LOGS_PATH=return_full_path("logs")
VISUALIZATION_PATH = return_full_path("reports", "figures")
PREDICTION_OUTPUT_FILE = return_full_path("reports","metrics")
MODELS_PATH = return_full_path("models")
FINAL_INPUT_FILE_FROM_DB = return_full_path("data", "final")
DATABASE_FOLDER = return_full_path("database")
RAW_DATA_FOLDER = return_full_path("data", "raw")
GOOD_RAW_DATA_FOLDER = return_full_path("data", "processed", "good_raw")
BAD_RAW_DATA_FOLDER = return_full_path("data", "processed", "bad_raw")
ARCHIVED_RAW_DATA_FOLDER = return_full_path("data", "processed", "archived_raw")
