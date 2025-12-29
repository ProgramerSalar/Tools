GPUS=1
ANNO_FILE=/content/Tools/annotation/output.jsonl
VAE_MODEL_PATH=/content/Tools/extract_latent/PATH/vae_model_ckpt
WIDTH=640
HEIGHT=384
NUM_FRAMES=121

torchrun --nproc_per_node $GPUS \
    /content/Tools/extract_latent/extract_video_latent.py \
    --model_dtype bf16 \
    --batch_size 4 \
    --anno_file $ANNO_FILE \
    --model_path $VAE_MODEL_PATH \
    --width $WIDTH \
    --height $HEIGHT \
    --num_frames $NUM_FRAMES \
    
