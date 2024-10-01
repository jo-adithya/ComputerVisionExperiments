"""
Contains utility function for downloading images from DuckDuckGo.
"""

import requests
from pathlib import Path

from io import BytesIO
from PIL import Image

from duckduckgo_search import DDGS

from typing import List


def download_images(
    target_path: str,
    keywords: List[str],
    filenames: List[str],
    n: int = 1,
):
    """Download images from DuckDuckGo.

    Parameters
    ----------
    target_path: str
        Path for the images to be stored into.
    keywords: List[str]
        List of image keywords to be searched and dowloaded.
    filenames: List[str]
        List of filename for the downloaded images.
        Just need one filename per keyword.
        If n > 1, images will be saved in
        filename[0]_0.jpg, filename[0]_1.jpg, filename[1]_0.jpg, etc.
    n: int
        Number of images to be downloaded for each keyword
    """
    ddg = DDGS()
    total_images = len(keywords) * n

    # Search for image results
    for i, keyword in enumerate(keywords):
        results = ddg.images(keyword, max_results=n)

        # Download all the images
        for j, result in enumerate(results):
            while True:
                image_url = result["image"]
                response = requests.get(image_url)
                filename = f"{filenames[i]}_{j}.jpg"
                target_image_path = target_path / filenames[i]
                target_image_path.mkdir(parents=True, exist_ok=True)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    image.save(target_image_path / filename)
                    print(f"Downloaded image {filename}")
                    break
                else:
                    print(f"Failed to download image {filename}")


__all__ = ["download_images"]
