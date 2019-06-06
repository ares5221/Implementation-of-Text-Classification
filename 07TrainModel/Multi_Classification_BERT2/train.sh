#!/usr/bin/env bash

python3 run_classifier.py \
  --data_dir=data \
  --task_name=sim \
  --vocab_file=chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json \
  --output_dir=tmp/sim_model \
  --do_train=true \
  --do_eval=true \
  --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=300 \
  --train_batch_size=16 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0


python run_classifier.py --task_name=sim --vocab_file=G:/downloaddata/bert-chinese/chinese_L
-12_H-768_A-12/vocab.txt --bert_config_file=G:/downloaddata/bert-chinese/chinese_L-12_H-768_A-12/bert_config.json --do_train=true --do_eval=true --init_checkpoint=G:/downloaddata/ber
t-chinese/chinese_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=70 --train_batch_size=32 --learning_rate=5e-5 --num_train_epochs=3.0 --output_dir=sim_modelrain_epochs=3.0 --output
_dir=sim_model --data_dir=G:/tf-start/Implementation-of-Text-Classification/07TrainModel/Multi_Classification_BERT2/data
