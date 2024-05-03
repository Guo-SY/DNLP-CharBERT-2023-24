import datasets
import pandas as pd
import random
import os

DIV = 20

def merge_and_shuffle(file1_path, file2_path, file3_path, file4_path, output_file_path):
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2, open(file3_path, 'r') as file3, open(file4_path, 'r') as file4:
        content1 = file1.readlines()
        content2 = file2.readlines()
        content3 = file3.readlines()
        content4 = file3.readlines()

    merged_content = content1 + content2 + content3 + content4
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


# LITTLE ITALIAN DATASET
dataset = datasets.load_dataset("wikipedia", "20220301.fr")
dataset = dataset['train']
dataset = dataset.select(range(2000))
dataset = dataset.to_pandas()
train = dataset.iloc[:1300]
val = dataset.iloc[1300:1600]
test = dataset.iloc[1600:2000]
train.to_csv('wiki_fr_train.csv', index=False)
val.to_csv('wiki_fr_val.csv', index=False)
test.to_csv('wiki_fr_test.csv', index=False)

# LITTLE ITALIAN DATASET
dataset = datasets.load_dataset("wikipedia", "20220301.de")
dataset = dataset['train']
dataset = dataset.select(range(2000))
dataset = dataset.to_pandas()
train = dataset.iloc[:1300]
val = dataset.iloc[1300:1600]
test = dataset.iloc[1600:2000]
train.to_csv('wiki_de_train.csv', index=False)
val.to_csv('wiki_de_val.csv', index=False)
test.to_csv('wiki_de_test.csv', index=False)


# LITTLE ENGLISH AND ITALIAN DATASET AGAIN
files = ["wiki_eng_train.csv", "wiki_eng_val.csv", "wiki_eng_test.csv", "wikil_eng_train.csv",
         "wiki_ita_train.csv", "wiki_ita_val.csv", "wiki_ita_test.csv",
         "wiki_fr_train.csv", "wiki_fr_val.csv", "wiki_fr_test.csv",
         "wiki_de_train.csv", "wiki_de_val.csv", "wiki_de_test.csv"
        ]


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
file3_path = 'wiki_fr_train.txt'
file4_path = 'wiki_de_train.txt'

output_file_path = './wikipedia_resized/wikil_eng_fr_ita_de_train.txt'
merge_and_shuffle(file1_path, file2_path, file3_path, file4_path, output_file_path)


file1_path = 'wiki_eng_val.txt'
file2_path = 'wiki_ita_val.txt'
file3_path = 'wiki_fr_val.txt'
file4_path = 'wiki_de_val.txt'

output_file_path = './wikipedia_resized/wikil_eng_fr_ita_de_val.txt'
merge_and_shuffle(file1_path, file2_path, file3_path, file4_path, output_file_path)

file1_path = 'wiki_eng_test.txt'
file2_path = 'wiki_ita_test.txt'
file3_path = 'wiki_fr_test.txt'
file4_path = 'wiki_de_test.txt'

output_file_path = './wikipedia_resized/wikil_eng_fr_ita_de_test.txt'
merge_and_shuffle(file1_path, file2_path, file3_path, file4_path, output_file_path)
os.replace("wiki_eng_train.txt", "./wikipedia_resized/wiki_eng_train.txt")
os.replace("wiki_eng_val.txt", "./wikipedia_resized/wiki_eng_val.txt")
os.replace("wiki_eng_test.txt", "./wikipedia_resized/wiki_eng_test.txt")

# clean folder
files = ["wiki_eng_train.csv", "wiki_eng_val.csv", "wiki_eng_test.csv", "wikil_eng_train.csv", "wikil_eng_train.txt",
         "wiki_ita_train.csv", "wiki_ita_val.csv", "wiki_ita_test.csv",
         "wiki_ita_train.txt", "wiki_ita_val.txt", "wiki_ita_test.txt",

          "wiki_fr_train.csv", "wiki_fr_val.csv", "wiki_fr_test.csv",
         "wiki_fr_train.txt", "wiki_fr_val.txt", "wiki_fr_test.txt",

          "wiki_de_train.csv", "wiki_de_val.csv", "wiki_de_test.csv",
         "wiki_de_train.txt", "wiki_de_val.txt", "wiki_de_test.txt"
         ]
         
for f in files:
  os.remove(f)
