import datasets
import os

dataset = datasets.load_dataset("pubmed_qa", "pqa_unlabeled")
dataset = dataset['train'].shuffle(seed = 42).select(range(len(dataset["train"])//2)) 
full = dataset["long_answer"]

train_len = len(full)
val_len = test_len = int(train_len * 0.1)
train_len = train_len - val_len - test_len
train = full[:train_len]
val = full[train_len:train_len+val_len]
test = full[train_len+val_len:]

general_path = 'datasets/medical_domain/mlm/'

if not os.path.exists(general_path):
    os.makedirs(general_path)

# save as txt file
with open(general_path+'train_pubmed_full.txt', 'w', encoding="utf-8") as f:
    for text in train:
        f.write(text + '\n')
with open(general_path+'val_pubmed_full.txt', 'w', encoding="utf-8") as f:
    for text in val:
        f.write(text + '\n')
with open(general_path+'test_pubmed_full.txt', 'w', encoding="utf-8") as f:
    for text in test:
        f.write(text + '\n')