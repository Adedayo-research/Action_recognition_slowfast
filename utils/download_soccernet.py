import os
from SoccerNet.Downloader import SoccerNetDownloader

# The NAS volume we mounted to the container
NAS_TARGET_DIR = "/mnt/data/SoccerNet_Dataset"
os.makedirs(NAS_TARGET_DIR, exist_ok=True)

print(f"Initializing SoccerNet Downloader straight into {NAS_TARGET_DIR}...")

# Initialize downloader without a password (required for public v2 datasets)
# We are downloading ALL splits: train, valid, test, and challenge.
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=NAS_TARGET_DIR)

# Requesting ONLY the 224p downscaled videos and the Labels-v2.json annotations
files_to_download = ["1_224p.mkv", "2_224p.mkv", "Labels-v2.json"]

print("Starting high-speed cloud download of the 500-game dataset (~400GB)...")
mySoccerNetDownloader.downloadGames(files=files_to_download, split=["train", "valid", "test", "challenge"])

print("\n--- DOWNLOAD COMPLETE ---")
print("The entire SoccerNet-v2 dataset is now permanently stored in the university NAS!")
