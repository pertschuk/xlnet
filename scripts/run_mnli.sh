#!/usr/bin/env bash

export TPU_NAME=train-tpu
export GS_ROOT=gs://poloma-tpu
export GLUE_DIR=glue_data
export LARGE_DIR=xlnet_cased_L-24_H-1024_A-16

python run_classifier.py \
  --use_tpu=True \
  --tpu=${TPU_NAME} \
  --do_train=True \
  --do_eval=True \
  --task_name=mnli_matched \
  --data_dir=${GS_ROOT}/${GLUE_DIR}/MNLI \
  --output_dir=${GS_ROOT}/proc_data/mnli \
  --model_dir=${GS_ROOT}/exp/mnli \
  --uncased=False \
  --spiece_model_file=${LARGE_DIR}/spiece.model \
  --model_config_path=${GS_ROOT}/${LARGE_DIR}/xlnet_config.json \
  --init_checkpoint=${GS_ROOT}/${LARGE_DIR}/xlnet_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=128 \
  --num_hosts=1 \
  --num_core_per_host=8 \
  --learning_rate=2e-5 \
  --train_steps=100000 \
  --num_passes=2 \
  --save_steps=1000