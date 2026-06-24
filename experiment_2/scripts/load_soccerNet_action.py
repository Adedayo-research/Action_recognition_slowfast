import os
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames

# 2. Initialize Downloader pointing to your clean local directory
ROOT_PATH = r"/Action_recognition_slowfast/experiment_2/SoccerNet_Dataset"
myDownloader = SoccerNetDownloader(LocalDirectory=ROOT_PATH)

# Set your NDA password as an object property
myDownloader.password = "s0cc3rn3t"

# 3. Use getListGames directly (not as a method of myDownloader) to grab our 25 games
train = getListGames(split="train")[:15]
valid = getListGames(split="valid")[:5]
test = getListGames(split="test")[:5]

micro_games = train + valid + test

# 4. Download the label annotations for all three splits
myDownloader.downloadGames(files=["Labels-v2.json"], split=["train", "valid", "test"])

# 5. Download ONLY the 224p video files for the 25 matches
for idx, game in enumerate(micro_games):
    print(f"\n[ Progress: {idx + 1} / 25 ] Fetching video files for: {game}")

    # downloadGame relies on the internal .password set above
    myDownloader.downloadGame(game, files=["1_224p.mkv"])

print("downloads completed")