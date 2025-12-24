
# makes sure follow this step 

1. clone the huggingface repo because in this repo have the pretrained vae weight.
    ```
    git clone https://huggingface.co/ProgramerSalar/vae_model_ckpt
    ```
    
    put the `duffusion_pytorch_model.safetensors` file in the `PATH/vae_ckpt` folder

2. clone this repo: 
    ```
    git clone https://github.com/ProgramerSalar/VideoGen_FineTune.git
    ```

3. install the `req.txt` file 
    ```
    cd VideoGen_FineTune
    pip install -r req.txt 
    ```

4. run the `script`
    ```
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    ```

    ```
    sh extract_latent/script/script.sh
    ```
