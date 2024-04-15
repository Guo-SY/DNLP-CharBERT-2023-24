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

In this link, The author have provided a pre-trained model based on wikipedia datasets. You can download the pre-trained models then use them as the: 

MODEL_DIR = YOUR_MODEL_PARH/Pre-models/charbert-bert-wiki

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
MODEL_DIR=YOUR_MODEL_PARH/charbert-bert-pretrain 
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
MODEL_DIR=YOUR_MODEL_PATH/charbert-bert-wiki
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

## Citation
If you use the data or codes in this repository, please cite our paper.
```
@misc{ma2020charbert,
      title={CharBERT: Character-aware Pre-trained Language Model}, 
      author={Wentao Ma and Yiming Cui and Chenglei Si and Ting Liu and Shijin Wang and Guoping Hu},
      year={2020},
      eprint={2011.01513},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Issues
If there is any problem, please submit a GitHub Issue.
