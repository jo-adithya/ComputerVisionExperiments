"""
This file contains data collection utility function for fine tuning a prebuilt,
computer vision model on the data.

Functions
---------
fine_tune_data : fine tune a prebuilt computer vision model on the data.
"""

import shutil
from pathlib import Path
from dataclasses import dataclass

from fastai.data.all import *
from fastai.vision.all import *
from fastai.vision.widgets import *


@dataclass
class DataCleaner:
    dls: DataLoaders
    learn: Learner
    model_interp: ClassificationInterpretation
    cleaner: ImageClassifierCleaner

    def clean(self, dataset_path: Path):
        for idx in self.cleaner.delete():
            self.cleaner.fns[idx].unlink()
        for idx, class_ in self.cleaner.change():
            shutil.move(str(self.cleaner.fns[idx]), dataset_path / class_)


def fine_tune_data(
    dataset_path: Path, epoch=3, image_size=128, device="cpu"
) -> DataCleaner:
    """
    Fine tune a prebuilt computer vision model from fastai on the data.
    Return a ImageClassifierCleaner, so user can choose which data to keep and which data
    to remove.

    Parameters
    ----------
    dataset_path: Path
        Path to the dataset.
    epoch: int
        Number of epochs to fine tune.
    image_size: int
        Size for the image transformation for the model.
    device: str
        Device to use for training. Default is "cpu".
    """
    data_block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),  # Define the blocks: images and labels
        get_items=get_image_files,  # Function to get the items (file paths)
        get_y=parent_label,  # Function to get labels (from the parent folder name)
        item_tfms=Resize(
            image_size, method="pad", pad_mode="zeros"
        ),  # Resize each image to 128x128
    )
    dls = data_block.dataloaders(
        source=dataset_path, shuffle=True, num_workers=4, pin_memory=True
    ).to(device)
    dls.show_batch(max_n=8, figsize=(8, 5), nrows=2)

    learn = vision_learner(dls=dls, arch=resnet18, metrics=error_rate).to(device)
    print("Fine tuning vision learner...")
    learn.fine_tune(epoch)

    model_interp = ClassificationInterpretation.from_learner(learn)
    model_interp.plot_confusion_matrix()

    cleaner = ImageClassifierCleaner(learn)

    return DataCleaner(dls=dls, learn=learn, model_interp=model_interp, cleaner=cleaner)


__all__ = ["fine_tune_data"]
