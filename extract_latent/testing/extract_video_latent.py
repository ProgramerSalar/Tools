import torch 
from torch.utils.data import DataLoader
from concurrent import futures
import sys 
import os

# Ensure path is correct
sys.path.append("/content/Tools")

from args import get_args
from vae import CausalVideoVAELossWrapper
from Dataset.video_dataset import VideoDataset

def build_model(args):
    model_path = args.model_path
    model_dtype = args.model_dtype
    # Load model
    model = CausalVideoVAELossWrapper(model_path, model_dtype=model_dtype, interpolate=False, add_discriminator=False)
    model.vae.enable_tiling(True)
    model = model.eval()
    return model

def build_data_loader(args):
    def collate_fn(batch):
        # Flattens the list of lists if dataset returns chunks, otherwise just processes items
        flat_inputs = []
        flat_outputs = []

        for item in batch:
            # Handle case where dataset returns a list of clips per video
            if isinstance(item, list):
                for subitem in item:
                    flat_inputs.append(subitem['input'])
                    flat_outputs.append(subitem['output'])
            else:
                flat_inputs.append(item['input'])
                flat_outputs.append(item['output'])

        # OPTIMIZATION 1: Stack into a single batch tensor for GPU efficiency
        # Shape becomes [Batch_Size, Channels, Frames, Height, Width]
        return {
            'input': torch.stack(flat_inputs), 
            'output': flat_outputs
        }
    
    dataset = VideoDataset(anno_file=args.anno_file,
                           width=args.width,
                           height=args.height,
                           num_frames=args.num_frames)
    
    # OPTIMIZATION 2: Lower num_workers to reduce RAM usage. 
    # Increase batch_size in your ARGS if you have VRAM headroom.
    loader = DataLoader(dataset=dataset,
                        batch_size=args.batch_size,
                        num_workers=6,  # Reduced from 6 to save RAM
                        pin_memory=True,
                        shuffle=False,
                        collate_fn=collate_fn,
                        drop_last=False,
                        persistent_workers=True # Keeps workers alive to avoid setup overhead
                        )
    
    return loader

def save_tensor(tensor, output_path):
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        torch.save(tensor.clone(), output_path)
    except Exception as e:
        print(f"Failed to save {output_path}: {e}")

def main(args):
    device = torch.device('cuda')
    model = build_model(args).to(device)
    
    if args.model_dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    data_loader = build_data_loader(args)
    
    # Use fewer threads for saving to avoid disk I/O choking
    with futures.ThreadPoolExecutor(max_workers=4) as executor:
        
        # Prefetching setup implicitly handled by DataLoader
        for step, sample in enumerate(data_loader):
            # Move entire batch to GPU at once (Non-blocking speeds up transfer)
            pixel_values = sample['input'].to(device, dtype=torch_dtype, non_blocking=True)
            # --- ADD THIS FIX ---
            # If shape is [Batch, 1, Channels, Frames, Height, Width], remove the '1'
            if pixel_values.dim() == 6:
                pixel_values = pixel_values.squeeze(1)
            # --------------------
            output_path_list = sample['output']

            print(f"Processing Batch {step} | Shape: {pixel_values.shape}")

            with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch_dtype):
                
                # OPTIMIZATION 3: Run inference on the WHOLE batch at once.
                # This drastically increases VRAM usage (good!) and speed.
                # If you get Out Of Memory (OOM), reduce --batch_size in args.
                batch_latents = model.encode_latent(
                    pixel_values, 
                    sample=True, 
                    temporal_chunk=True, 
                    window_size=8, 
                    tile_sample_min_size=256
                )
                
                # Move entire result batch to CPU
                batch_latents_cpu = batch_latents.cpu()

                # Dispatch save tasks
                for i, output_path in enumerate(output_path_list):
                    # executor.submit is non-blocking
                    executor.submit(save_tensor, batch_latents_cpu[i], output_path)

            # Cleanup: Only clear if absolutely necessary. 
            # Routine empty_cache() slows things down. 
            # Let Python garbage collector handle 'pixel_values' and 'batch_latents' automatically.
            
            # Periodically clear internal model cache if the model leaks memory
            if step % 10 == 0:
                for module in model.modules():
                    if hasattr(module, "_clear_context_parallel_cache"):
                        module._clear_context_parallel_cache()

if __name__ == "__main__":
    args = get_args()
    # OPTIONAL: Force set batch size here if you want to override args
    # args.batch_size = 4 
    main(args=args)