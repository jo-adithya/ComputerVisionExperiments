"""
This file contains data collection utility function for removing unused data,
and transforming the folder names.

Functions
---------
remove_data_folder: remove unused data and transform folder names.
"""

import shutil
from pathlib import Path

from typing import List, Dict


# Remove all other data except for the dogs
def remove_data_folder(dataset_path: Path, folders_to_keep: Dict[str, str] | List[str]):
    """
    Remove unused data and transform the folder names into the specified names.

    Parameters
    ----------
    dataset_path: Path
        Path to the dataset to be cleaned.
    folders_to_keep : Dict[str, str] | List[str]
        List of folder names to be saved. Other folder names will be deleted.
        If type is a `Dict[str, str]`, the keys should be the current folder names
        to be saved, and the values should the the names that the key wnat to be
        transformed to.
    """
    if isinstance(folders_to_keep, dict):
        folders_to_keep_set = set(folders_to_keep.keys()) | set(
            folders_to_keep.values()
        )
    elif isinstance(folders_to_keep, list):
        folders_to_keep_set = set(folders_to_keep)

    for folder in dataset_path.iterdir():
        if not folder.is_dir() or folder.name not in folders_to_keep_set:
            print(f"Deleting file/folder: {folder.name}...")
            shutil.rmtree(folder)
        elif isinstance(folders_to_keep, dict):
            if folder.name in folders_to_keep.values():
                continue
            shutil.move(folder, Path(dataset_path / folders_to_keep[folder.name]))


__all__ = ["remove_data_folder"]
