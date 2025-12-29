import torch 
from torch.utils.data import DataLoader
from concurrent import futures
import sys 
sys.path.append("/content/VideoGen_FineTune")


from args import get_args

from vae import CausalVideoVAELossWrapper
from Dataset.video_dataset import VideoDataset


def build_model(args):
    model_path = args.model_path
    model_dtype = args.model_dtype
    model = CausalVideoVAELossWrapper(model_path, model_dtype=model_dtype, interpolate=False, add_discriminator=False)
    model.vae.enable_tiling(True)
    model = model.eval()
    return model



def build_data_loader(args):

    def collate_fn(batch):

        return_batch = {
            'input': [],
            'output': []
        }

        for videos_ in batch:
            for video_input in videos_:
                return_batch['input'].append(video_input['input'])
                return_batch['output'].append(video_input['output'])

        return return_batch
    

    dataset = VideoDataset(anno_file=args.anno_file,
                           width=args.width,
                           height=args.height,
                           num_frames=args.num_frames)
    
    loader = DataLoader(dataset=dataset,
                        batch_size=args.batch_size,
                        num_workers=6,
                        pin_memory=True,
                        shuffle=False,
                        collate_fn=collate_fn,
                        drop_last=False,
                        )
    
    return loader




def main(args):

    device = torch.device('cuda')
    model = build_model(args).to(device)
    
    if args.model_dtype == "bf16":
        torch_dtype = torch.bfloat16

    else:
        torch_dtype = torch.float16

    data_loader = build_data_loader(args)
    task_queue = []

    with futures.ThreadPoolExecutor(max_workers=8) as Excecuter:

        for sample in data_loader:
            input_video_list = sample['input']
            output_path_list = sample['output']

            print(f"what is the sample: >>>>>>>>>>>>> {sample}")

            # CLEAR CACHE before processing new heavy batch
            torch.cuda.empty_cache()

            with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch_dtype):
                for video_input, output_path in zip(input_video_list, output_path_list):
                    
                    # Move to device JUST before use to save memory
                    video_input = video_input.to(device)

                    # You might need to check if your model supports a 'tile_sample_min_size'
                    # or explicitly modify 'vae/modeling_causal_vae.py' to set temporal_chunk=True
                    video_latent = model.encode_latent(video_input, sample=True, temporal_chunk=True, window_size=8, tile_sample_min_size=256)
                    
                    # Submit task and IMMEDIATELY delete reference to free VRAM
                    task_queue.append(Excecuter.submit(save_tensor, video_latent.cpu(), output_path))
                    del video_input
                    del video_latent

                    torch.cuda.empty_cache()
    
                    # Force clear the causal cache in the model to prevent memory bloat over 103 videos
                    # This iterates through the model and clears the deque cache in every CausalConv3d layer
                    for module in model.modules():
                        if hasattr(module, "_clear_context_parallel_cache"):
                            module._clear_context_parallel_cache()

                    

        for future in futures.as_completed(task_queue):
            res = future.result()


    


def save_tensor(tensor, output_path):
    print(f"what is the output_path : {output_path}")
    try:
        torch.save(tensor.clone(), output_path)

    except ValueError:
        print("you Tensor is not saved.")



if __name__ == "__main__":
    args = get_args()
    main(args=args)