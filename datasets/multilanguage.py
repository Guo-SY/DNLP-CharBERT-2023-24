import datasets
import pandas as pd
import random
import os

DIV = 20

def merge_and_shuffle(file_paths, output_file_path):
    merged_content = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            merged_content.extend(file.readlines())
    random.shuffle(merged_content)
    with open(output_file_path, 'w') as output_file:
        output_file.writelines(merged_content)

def split_and_save(dataset, prefix):
    train = dataset.iloc[:10000]
    val = dataset.iloc[10000:11000]
    test = dataset.iloc[11000:12000]
    os.makedirs(os.path.dirname(f"./wikipedia_resized/"), exist_ok=True)
    train.to_csv(f'{prefix}_train.csv', index=False)
    val.to_csv(f'{prefix}_val.csv', index=False)
    test.to_csv(f'{prefix}_test.csv', index=False)
    return f"{prefix}_train.csv", f"{prefix}_val.csv", f"{prefix}_test.csv"

def export_to_txt(dataset, filename):
    with open(filename, 'w') as f:
        for row in dataset['text'][:len(dataset) // DIV]:
            f.write(row)

datasets_list = [
    ("20220301.simple", "wiki_eng", 12000),
    ("20220301.it", "wiki_ita", 2000),
    ("20220301.frr", "wiki_frr", 2000),
    ("20220301.de", "wiki_de", 2000)
]

csv_files = []

for dataset_name, prefix, size in datasets_list:
    dataset = datasets.load_dataset("wikipedia", dataset_name)['train']
    dataset = dataset.select(range(size)).to_pandas()
    csv_files.extend(split_and_save(dataset, f'./wikipedia_resized/{prefix}'))

for file_ in csv_files:
    dataset = pd.read_csv(file_)
    export_to_txt(dataset, f"{file_.split('.')[0]}.txt")

merge_and_shuffle([f'{prefix}.txt' for prefix in csv_files if 'train' in prefix], './wikipedia_resized/wikil_eng_wiki_ita_train.txt')
merge_and_shuffle([f'{prefix}.txt' for prefix in csv_files if 'val' in prefix], './wikipedia_resized/wikil_eng_wiki_ita_val.txt')
merge_and_shuffle([f'{prefix}.txt' for prefix in csv_files if 'test' in prefix], './wikipedia_resized/wikil_eng_wiki_ita_test.txt')

# Clean folder
for f in csv_files:
    os.remove(f)
for f in os.listdir('.'):
    if f.endswith('.txt'):
        os.remove(f)