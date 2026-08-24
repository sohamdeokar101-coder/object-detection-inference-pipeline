from src.dataset import download_online_sample

ONLINE_URL = "https://ultralytics.com/images/bus.jpg"
LOCAL_PATH = "sample_images/real_bus.jpg"

if __name__ == "__main__":
    download_online_sample(ONLINE_URL, LOCAL_PATH)