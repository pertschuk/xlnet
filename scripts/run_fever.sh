#!/usr/bin/env bash

export TPU_NAME=fever-tpu
export GS_ROOT=gs://poloma-tpu
export FEVER_DIR=fever
export LARGE_DIR=xlnet_cased_L-24_H-1024_A-16

python run_classifier.py \
  --use_tpu=True \
  --tpu=${TPU_NAME} \
  --do_eval=True \
  --predict_dir=${GS_ROOT}/fever_predict \
  --task_name=fever \
  --data_dir=./${FEVER_DIR} \
  --output_dir=${GS_ROOT}/fever \
  --model_dir=${GS_ROOT}/exp/fever \
  --uncased=False \
  --spiece_model_file=${LARGE_DIR}/spiece.model \
  --model_config_path=${GS_ROOT}/${LARGE_DIR}/xlnet_config.json \
  --init_checkpoint=${GS_ROOT}/${LARGE_DIR}/xlnet_model.ckpt \
  --max_seq_length=128 \
  --eval_batch_size=32 \
  --train_batch_size=128 \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --learning_rate=2e-5 \
  --train_steps=100000 \
  --num_passes=3 \
  --save_steps=1000