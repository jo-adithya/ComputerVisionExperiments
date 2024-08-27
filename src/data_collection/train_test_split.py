"""
Module: data_collection.train_test_split

This module provides a function to split a dataset of images into training and testing sets. 
The images are organized by class, and the function ensures that each class's images are divided 
according to the specified train-test ratio. The resulting splits are moved to designated directories.

Functions
---------
train_test_split(dataset_path: Path, limit_num_images: int, train_ratio: float = 0.9, train_path: Path = None, test_path: Path = None)
    Splits the images in the dataset into training and testing sets based on the specified ratio.

"""

import random
import shutil
from pathlib import Path
from collections import defaultdict

from typing import Dict, List, Optional


def train_test_split(
    dataset_path: Path,
    train_path: Optional[Path],
    test_path: Optional[Path],
    train_ratio=0.9,
    limit_num_images=0,
):
    """
    Splits images in a dataset into training and testing sets based on the given ratio.

    This function organizes images into subdirectories for each class within the provided
    dataset path. The images are split into training and testing sets according to
    the specified ratio, with a limit on the maximum number of images considered per class.

    Parameters
    ----------
    dataset_path : Path
        The root directory containing images organized by class folders. Each subfolder represents
        a class and contains images belonging to that class.
    train_path : Path, optional
        The directory where the training images will be moved. If not provided, defaults to
        `dataset_path / "train"`.
    test_path : Path, optional
        The directory where the testing images will be moved. If not provided, defaults to
        `dataset_path / "test"`.
    train_ratio : float, optional
        The ratio of images to be used for the training set. The value should be between 0 and 1.
        The default is 0.9, meaning 90% of the images will be used for training and 10% for testing.
    limit_num_images : int
        The maximum number of images to consider per class. If there are more images than this limit,
        only a random sample of this size will be used.

    Raises
    ------
    ValueError
        If `train_ratio` is not between 0 and 1.
    """

    # Set default paths if not provided
    if train_path is None:
        train_path = dataset_path / "train"
    if test_path is None:
        test_path = dataset_path / "test"

    # Validate train_ratio
    if train_ratio < 0 or train_ratio > 1:
        raise ValueError("train_ratio must be between 0 and 1")

    # Dictionary of class label and list of images' Path in that class
    class_images_dict: Dict[str, List[Path]] = defaultdict(list)
    for image_path in dataset_path.rglob("*.*"):
        class_name = image_path.parent.name
        class_images_dict[class_name].append(image_path)

    # Split the images into train test split for each class
    for class_label, images in class_images_dict.items():
        if limit_num_images <= 0:
            num_samples = len(images)
        else:
            num_samples = min(len(images), limit_num_images)
        selected_images = random.sample(images, num_samples)

        num_train_samples = int(train_ratio * num_samples)
        train_images = selected_images[:num_train_samples]
        test_images = selected_images[num_train_samples:]

        # Create new directory for each class
        train_class_path = train_path / class_label
        if train_class_path.exists():
            shutil.rmtree(train_class_path)
        train_class_path.mkdir(parents=True, exist_ok=True)

        test_class_path = test_path / class_label
        if test_class_path.exists():
            shutil.rmtree(test_class_path)
        test_class_path.mkdir(parents=True, exist_ok=True)

        for img in train_images:
            shutil.copy(img, train_class_path / img.name)
        for img in test_images:
            shutil.copy(img, test_class_path / img.name)


__all__ = ["train_test_split"]
