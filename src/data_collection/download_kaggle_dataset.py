"""Utility file to download kaggle dataset

Note: Requires setting KAGGLE_KEY and KAGGLE_USERNAME environment variables

Functions
---------
download_kaggle_dataset: Download kaggle dataset helper function
"""
import os
from pathlib import Path

# Getting the requires environment vars from google colab
try:
    from google.colab import userdata
    os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
    os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
except ModuleNotFoundError:
    print("Not running from google colab, cannot import userdata")

import kaggle

def download_kaggle_dataset(
    dataset_name: str,
    data_path: Path,
):
    """Download kaggle dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the kaggle dataset.
    data_path : Path
        Path to save the kaggle dataset to.
    """
    if data_path.is_dir() and data_path.exists():
        print(f"Folder {data_path.name} is found. Skipping download data from kaggle...")
        return
    print(f"Folder {data_path.name} is not found. Downloading from kaggle...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset_name, path=data_path, unzip=True)
    print(f"Downloaded {dataset_name} dataset to {data_path.as_posix()}")
