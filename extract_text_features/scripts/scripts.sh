#!/bin/bash

# This script is used for batch extract the vae latents for video generation training
# Since the video latent extract is very slow, pre-extract the video vae latents will save the training time

GPUS=1  # The gpu number
MODEL_NAME=pyramid_flux     # The model name, `pyramid_flux` or `pyramid_mmdit`
MODEL_PATH=PATH/txt_path  # The VAE CKPT dir.
# ANNO_FILE=annotation/video_text.jsonl   # The video annotation file path
ANNO_FILE=/content/VideoGen_FineTune/annotation/test_video_annotation.jsonl


torchrun --nproc_per_node=$GPUS \
    /content/videoGen_fineTune/extract_text_features/extract_text_feature.py \
    --batch_size 1 \
    --model_dtype bf16 \
    --model_name $MODEL_NAME \
    --model_path $MODEL_PATH \
    --anno_file $ANNO_FILE