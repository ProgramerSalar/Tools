#-------------------------------------------------------#
#                                                       #
#    THIS FILE TAKE VAE `annotation` jsonl              #
#                                                       #
#-------------------------------------------------------#


import json
import os
import re

def write_latent_file(json_path, drive_video_file_path):
    json_list = []
    
    # 1. Load JSON and Create a Lookup Map
    # We create a dictionary where keys are filenames (e.g., "video1.mp4")
    # This makes matching instant later.
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
    
    # Assumption: Your JSON has a key 'file_name' or similar that matches the video file.
    # If your JSON is just a list of texts without filenames, you cannot match them correctly.
    # Here I assume the structure is [{'file_name': 'vid1.mp4', 'text': '...'}, ...]
    text_lookup = {item['file']: item['text'] for item in raw_data if 'file' in item}
    
    print(f"Loaded {len(text_lookup)} annotations.")

    # 2. Walk through folders
    for root, dirs, files in os.walk(drive_video_file_path):
        for file in files:
            if file.lower().endswith(".mp4"):
                # Check if we have an annotation for this specific video
                if file in text_lookup:
                    full_video_path = os.path.join(root, file)
                    
                    # Safe way to get path without extension for the video latent
                    video_latent_path = os.path.splitext(full_video_path)[0] + ".pt"
                    
                    text_content = text_lookup[file]
                    
                    # 3. Text Path Sanitization
                    # Remove special chars to avoid invalid file paths
                    # Keep only alphanumeric, underscores, and spaces
                    safe_text = re.sub(r'[^\w\s]', '', text_content) 
                    text_clean = '_'.join(safe_text.split()) # specific split handles multiple spaces
                    
                    # Construct text latent path
                    text_latent = f'/content/drive/MyDrive/video_text_latent/{text_clean}.pt'
                    
                    entry = {
                        "video": full_video_path.strip('.'),        # Removed strip('.')
                        "video_latent": video_latent_path.strip('.'), 
                        "text": text_content,
                        "text_latent": text_latent
                    }
                    json_list.append(entry)

    # 4. Save JSONL
    jsonl_path = json_path.replace('.json', '_video_text_and_latent.jsonl')
    
    with open(jsonl_path, 'w') as f:
        for entry in json_list:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Successfully dumped {len(json_list)} matched entries to {jsonl_path}")

if __name__ == "__main__":
    vae_json_path = "/home/manish/Desktop/projects/video_Generation/Tools/annotation/class_5.json"
    video_file_path = "./content"
    
    # Ensure the path exists before running
    if os.path.exists(vae_json_path) and os.path.exists(video_file_path):
        write_latent_file(vae_json_path, video_file_path)
    else:
        print("Error: Check your input paths.")