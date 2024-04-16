import wget
import os

# Define the folder path where you want to download the files
os.makedirs(os.path.dirname("./datasets/SQuAD_2.0/"), exist_ok=True)

# Download the files directly into the folder
train_file = wget.download("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json")
val_file = wget.download("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json")
test_file = wget.download("https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/")
