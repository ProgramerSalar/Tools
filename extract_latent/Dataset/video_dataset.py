import torch, cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as py_transform
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms as pth_transforms
import jsonlines
from tqdm import tqdm

def get_transform(width, height, new_width=None, new_height=None, resize=False,):
    transform_list = []

    if resize:
        # rescale according to the largest ratio
        scale = max(new_width / width, new_height / height)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        
        transform_list.append(pth_transforms.Resize((resized_height, resized_width), InterpolationMode.BICUBIC, antialias=True))
        transform_list.append(pth_transforms.CenterCrop((new_height, new_width)))
    
    transform_list.extend([
        pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_list = pth_transforms.Compose(transform_list)

    return transform_list

def load_video_and_transform(video_path, frame_indexs, frame_number, new_width=None, new_height=None, resize=False):
    video_capture = None
    frame_indexs_set = set(frame_indexs)

    try:
        video_capture = cv2.VideoCapture(video_path)
        frames = []
        frame_index = 0
        while True:
            flag, frame = video_capture.read()
            if not flag:
                break
            if frame_index > frame_indexs[-1]:
                break
            if frame_index in frame_indexs_set:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frame = frame.permute(2, 0, 1)
                frames.append(frame)
            frame_index += 1

        video_capture.release()
        
        if len(frames) == 0:
            print(f"Empty video {video_path}")
            return None

        frames = frames[:frame_number]
        duration = ((len(frames) - 1) // 8) * 8 + 1  # make sure the frames match: f * 8 + 1
        frames = frames[:duration]
        frames = torch.stack(frames).float() / 255
        video_transform = get_transform(frames.shape[-1], frames.shape[-2], new_width, new_height, resize=resize)
        frames = video_transform(frames).permute(1, 0, 2, 3)
        return frames

    except Exception as e:
        print(f"Loading video: {video_path} exception {e}")
        if video_capture is not None:
            video_capture.release()
        return None

class VideoDataset(Dataset):
    def __init__(self, anno_file, width, height, num_frames):
        super().__init__()
        self.annotation = []
        self.width = width
        self.height = height
        self.num_frames = num_frames

        with jsonlines.open(anno_file, 'r') as reader:
            for item in tqdm(reader):
                self.annotation.append(item)

        tot_len = len(self.annotation)
        print(f"Totally {len(self.annotation)} videos")

    def process_one_video(self, video_item):
        videos_per_task = []
        video_path = video_item['video']
        output_latent_path = video_item['video_latent']

        # The sampled frame indexs of a video, if not specified, load frames: [0, num_frames)
        frame_indexs = video_item['frames'] if 'frames' in video_item else list(range(self.num_frames))

        try:
            video_frames_tensors = load_video_and_transform(
                video_path, 
                frame_indexs, 
                frame_number=self.num_frames,    # The num_frames to encode
                new_width=self.width, 
                new_height=self.height, 
                resize=True
            )
            
            if video_frames_tensors is None:
                return videos_per_task

            video_frames_tensors = video_frames_tensors.unsqueeze(0)
            videos_per_task.append({'video': video_path, 'input': video_frames_tensors, 'output': output_latent_path})

        except Exception as e:
            print(f"Load video tensor ERROR: {e}")

        return videos_per_task

    def __getitem__(self, index):
        try:
            video_item = self.annotation[index]
            videos_per_task = self.process_one_video(video_item)
        except Exception as e:
            print(f'Error with {e}')
            videos_per_task = []

        return videos_per_task

    def __len__(self):
        return len(self.annotation)




if __name__ == "__main__":

    # tranform = get_transform(width=512, height=512,
    #                          new_height=256, new_width=256,
    #                          resize=True)
    # print(tranform)
    # -------------------------
    num_frames= 121

    load_video_and_transform_ = load_video_and_transform(video_path="../../vae_from_scratch/Data/webvid101/stock-footage-western-clownfish-swimming-around.mp4",
                                                         frame_indexs=list(range(num_frames)),
                                                         frame_number=num_frames,
                                                         new_width=256,
                                                         new_height=256,
                                                         resize=True
                                                         )
    print(load_video_and_transform_.shape)