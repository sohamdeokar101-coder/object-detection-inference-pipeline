import os
import cv2
import numpy as np

def create_local_sample():
    os.makedirs("sample_images", exist_ok=True)
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (220, 220), (255, 0, 0), -1)
    cv2.circle(img, (400, 200), 80, (0, 0, 255), -1)
    
    path = "sample_images/synthetic_sample.jpg"
    cv2.imwrite(path, img)
    print(f"✅ Synthetic local image created at '{path}'")

if __name__ == "__main__":
    create_local_sample()