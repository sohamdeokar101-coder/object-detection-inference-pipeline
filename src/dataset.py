import os
import urllib.request
import cv2
import numpy as np
from typing import List


def download_online_sample(url: str, save_path: str) -> str:
    """Downloads an online sample image to local disk if it does not exist."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        print(f"📥 Downloading sample image from {url}...")
        urllib.request.urlretrieve(url, save_path)
        print(f"✅ Saved online image to '{save_path}'")
    return save_path


def load_images_from_directory(directory_path: str) -> List[str]:
    """Retrieves all valid image paths from a given directory."""
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_paths = []
    
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(directory_path, filename))
                
    return sorted(image_paths)