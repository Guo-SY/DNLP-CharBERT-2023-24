import wget
import os

def conllu_to_txt(conllu_file, n_stop=100_000):
    name_new_file = conllu_file.split(".")[0] + ".txt"
    f = open("./CoNLL2003_ita/" + name_new_file, "w")
    with open(conllu_file, 'r', encoding='utf-8') as f_in:
        for i, line in enumerate(f_in):
            if i >= n_stop:
                break
            if len(line.split("\t")) == 3:
                if line.split("\t")[0] == '0':
                    f.write("\n")
                f.write(line.split("\t")[1] + " " + line.split("\t")[2])
    f.close()

train_file = wget.download("https://raw.githubusercontent.com/Babelscape/wikineural/master/data/wikineural/it/train.conllu", "train.conllu")
val_file = wget.download("https://raw.githubusercontent.com/Babelscape/wikineural/master/data/wikineural/it/val.conllu", "val.conllu")
test_file = wget.download("https://raw.githubusercontent.com/Babelscape/wikineural/master/data/wikineural/it/test.conllu", "test.conllu")

os.makedirs(os.path.dirname("./CoNLL2003_ita/"), exist_ok=True)
conllu_to_txt("train.conllu", n_stop=200_000)
conllu_to_txt("val.conllu")
conllu_to_txt("test.conllu")
os.remove("train.conllu")
os.remove("val.conllu")
os.remove("test.conllu")