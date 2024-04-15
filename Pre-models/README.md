<p align="center">
  <br>
    <img src="data/CharBert_logo.png" width="500" />  
  <br>
</p>
 
# CharBERT: Character-aware Pre-trained Language Model 

This repository contains resources of the following [COLING 2020](https://www.coling2020.org) paper.  


## Models
We primarily provide two models. Here are the download links:   
pre-trained CharBERT based on BERT [charbert-bert-wiki](https://drive.google.com/file/d/1rF5_LbA2qIHuehnNepGmjz4Mu6OqEzYT/view?usp=sharing)    
pre-trained CharBERT based on RoBERTa [charbert-roberta-wiki](https://drive.google.com/file/d/1tkO7_EH1Px7tXRxNDu6lzr_y8b4Q709f/view?usp=sharing)   

1. In this link, the authors provide pre-trained models based on the Wikipedia dataset. You can download the pre-trained model and use it as your model:

MODEL_DIR = YOUR_MODEL_PARH/Pre-models/charbert-bert-wiki

2. There is other method for training your pre-trained model. Firstly, you could dolwnload the Wikipedia simple-datasets by the commond line:
   
from datasets import load_dataset
datasets = load_dataset("wikipedia", "20220301.simple")

Second, you can use this datasets to help you train pre-trained models.

3. If you prefer to trained your pre-trained model, you could run the following code to help me train your pre-trained models.
```
pip3 install -q datasets
python3 /content/nlp_charbert/datasets/load_wiki_resized.py

import nltk
nltk.download('punkt')

python3 ./CharBERT/run_lm_finetuning.py \
    --model_type bert \
    --model_name_or_path bert-base-cased \
    --do_train \
    --do_eval \
    --train_data_file "./wikipedia_resized/wikil_eng_wiki_ita_train.txt" \
    --eval_data_file  "./wikipedia_resized/wikil_eng_wiki_ita_val.txt" \
    --term_vocab "/content/nlp_charbert/CharBERT/data/dict/term_vocab" \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --char_vocab "/content/nlp_charbert/CharBERT/data/dict/bert_char_vocab" \
    --mlm_probability 0.10 \
    --input_nraws 1000 \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 4 \
    --save_steps 10000 \
    --block_size 384  \
    --mlm \
    --overwrite_output_dir \
    --output_dir  "./output/bert_base_cased_wiki_eng"

```
 
MODEL_DIR = "./output/bert_base_cased_wiki_eng"


## Directory Guide
```
root_directory
    |- modeling    # contains source codes of CharBERT model part
    |- data   # Character attack datasets and the dicts for CharBERT
    |- processors # contains source codes for processing the datasets
    |- shell     # the examples of shell script for training and evaluation
    |- run_*.py  # codes for pre-training or finetuning

```

### MLM && NLM Pre-training
```
DATA_DIR=YOUR_DATA_PATH
MODEL_DIR=YOUR_MODEL_PATH/bert_base_cased #initialized by bert_base_cased model
OUTPUT_DIR=YOUR_OUTUT_PATH/mlm
python3 run_lm_finetuning.py \
    --model_type bert \
    --model_name_or_path ${MODEL_DIR} \
    --do_train \
    --do_eval \
    --train_data_file $DATA_DIR/testdata/mlm_pretrain_enwiki.train.t \
    --eval_data_file $DATA_DIR/testdata/mlm_pretrain_enwiki.test.t \
    --term_vocab ${DATA_DIR}/dict/term_vocab \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --mlm_probability 0.10 \
    --input_nraws 1000 \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 4 \
    --save_steps 10000 \
    --block_size 384 \
    --overwrite_output_dir \
    --mlm \
    --output_dir ${OUTPUT_DIR}

```

### SQuAD
```
MODEL_DIR= './output/bert_base_cased_wiki_eng'
SQUAD2_DIR=YOUR_DATA_PATH/squad 
OUTPUT_DIR=YOUR_OUTPUT_PATH/squad

python run_squad.py \
    --model_type bert \
    --model_name_or_path ${MODEL_DIR} \
    --do_train \
    --do_eval \
    --data_dir $SQUAD2_DIR \
    --train_file $SQUAD2_DIR/train-v1.1.json \
    --predict_file $SQUAD2_DIR/dev-v1.1.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 4 \
    --save_steps 2000 \
    --max_seq_length 384 \
    --overwrite_output_dir \
    --doc_stride 128 \
    --output_dir ${OUTPUT_DIR}
```

### NER
```
DATA_DIR=YOUR_DATA_PATH/CoNLL2003/NER-en
MODEL_DIR='./output/bert_base_cased_wiki_eng'
OUTPUT_DIR=YOUR_OUTPUT_PATH/ner
python run_ner.py --data_dir ${DATA_DIR} \
                  --model_type bert \
                  --model_name_or_path $MODEL_DIR \
                  --output_dir ${OUTPUT_DIR} \
                  --num_train_epochs 3 \
                  --learning_rate 3e-5 \
                  --char_vocab ./data/dict/bert_char_vocab \
                  --per_gpu_train_batch_size 6 \
                  --do_train \
                  --do_predict \
                  --overwrite_output_dir
```
