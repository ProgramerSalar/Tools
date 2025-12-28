

# Extract the latent from video.

1. clone the huggingface repo because in this repo have the pretrained vae weight.
    ```
    cd extract_latent/PATH
    git clone https://huggingface.co/ProgramerSalar/vae_model_ckpt
    ```
    
    put the `duffusion_pytorch_model.safetensors` file in the `PATH/vae_ckpt` folder

2. clone this repo: 
    ```
    git clone https://github.com/ProgramerSalar/Tools.git
    ```

3. Download the Dataset
```
    cd Tools
    hf download ProgramerSalar/clip_video clip_video_part_2.zip --repo-type dataset --local-dir .
    unzip clip_video_part_2.zip
```


4. install the `req.txt` file 
    ```
    cd tools
    pip install -r req.txt 
    ```

5. run the `script`
    ```
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    ```

    ```
    sh extract_latent/script/script.sh
    ```


# Extract the latent from text.
1. clone the huggingface repo because in this repo have the pretrained vae weight.
   
    ```
    cd extract_text_features/PATH
    git clone https://huggingface.co/ProgramerSalar/text_encoder_miniflux
    ```
    
    put the `duffusion_pytorch_model.safetensors` file in the `PATH/vae_ckpt` folder



4. run the `script`
    ```
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    ```

    ```
    sh extract_text_features/scripts/scripts.sh
    ```
