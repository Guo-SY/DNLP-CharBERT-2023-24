import datasets
import pandas as pd
import random
import os

DIV = 20

def merge_and_shuffle(file1_path, file2_path, output_file_path):
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        content1 = file1.readlines()
        content2 = file2.readlines()
    merged_content = content1 + content2
    random.shuffle(merged_content)

    with open(output_file_path, 'w') as output_file:
        output_file.writelines(merged_content)


dataset = datasets.load_dataset("wikipedia", "20220301.simple")
# save the dataset to csv
# FULL ENGLISH DATASET
dataset = dataset['train']
dataset = dataset.select(range(12000))
dataset = dataset.to_pandas()
# divide the dataset into 3 parts
train = dataset.iloc[:10000]
val = dataset.iloc[10000:11000]
test = dataset.iloc[11000:12000]
os.makedirs(os.path.dirname("./wikipedia_resized/"), exist_ok=True)
train.to_csv('wiki_eng_train.csv', index=False)
val.to_csv('wiki_eng_val.csv', index=False)
test.to_csv('wiki_eng_test.csv', index=False)
# LITTLE ENGLISH DATASET
little_train = dataset.iloc[:3500]
little_train.to_csv('wikil_eng_train.csv', index=False)
# LITTLE ITALIAN DATASET
dataset = datasets.load_dataset("wikipedia", "20220301.it")
dataset = dataset['train']
dataset = dataset.select(range(2000))
dataset = dataset.to_pandas()
train = dataset.iloc[:1300]
val = dataset.iloc[1300:1600]
test = dataset.iloc[1600:2000]
train.to_csv('wiki_ita_train.csv', index=False)
val.to_csv('wiki_ita_val.csv', index=False)
test.to_csv('wiki_ita_test.csv', index=False)

# LITTLE ENGLISH AND ITALIAN DATASET AGAIN
files = ["wiki_eng_train.csv", "wiki_eng_val.csv", "wiki_eng_test.csv", "wikil_eng_train.csv", "wiki_ita_train.csv", "wiki_ita_val.csv", "wiki_ita_test.csv"]
for file_ in files:
  dataset = pd.read_csv(file_)
  path = f"{file_.split('/')[-1].split('.')[0]}.txt"
  #export DataFrame to text file
  with open(path, 'w') as f:
      for row in dataset['text'][:len(dataset)//DIV]:
        f.write(row)

# MERGE AND SHUFFLE
file1_path = 'wikil_eng_train.txt'
file2_path = 'wiki_ita_train.txt'
output_file_path = './wikipedia_resized/wikil_eng_wiki_ita_train.txt'
merge_and_shuffle(file1_path, file2_path, output_file_path)
file1_path = 'wiki_eng_val.txt'
file2_path = 'wiki_ita_val.txt'
output_file_path = './wikipedia_resized/wikil_eng_wiki_ita_val.txt'
merge_and_shuffle(file1_path, file2_path, output_file_path)
file1_path = 'wiki_eng_test.txt'
file2_path = 'wiki_ita_test.txt'
output_file_path = './wikipedia_resized/wikil_eng_wiki_ita_test.txt'
merge_and_shuffle(file1_path, file2_path, output_file_path)
os.replace("wiki_eng_train.txt", "./wikipedia_resized/wiki_eng_train.txt")
os.replace("wiki_eng_val.txt", "./wikipedia_resized/wiki_eng_val.txt")
os.replace("wiki_eng_test.txt", "./wikipedia_resized/wiki_eng_test.txt")
# clean folder
files = ["wiki_eng_train.csv", "wiki_eng_val.csv", "wiki_eng_test.csv", "wikil_eng_train.csv", "wikil_eng_train.txt", "wiki_ita_train.csv", "wiki_ita_val.csv", "wiki_ita_test.csv", "wiki_ita_train.txt", "wiki_ita_val.txt", "wiki_ita_test.txt"]
for f in files:
  os.remove(f)