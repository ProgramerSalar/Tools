import jsonlines
import json, os, glob 
from pathlib import Path
import pandas as pd 



    
    


jsonl_file_path = "/home/manish/Desktop/projects/video_Generation_FineTune/Tools/annotation/video_data_files_path.jsonl"
output_file_path = "/home/manish/Desktop/projects/video_Generation_FineTune/Tools/annotation/video_data_files_path.json"

def add_pt_path(jsonl_file_path, output_file_path):

    """ 
        This function take a jsonl file and return the video_path, video_latent
    """

    with jsonlines.open(jsonl_file_path) as files:

        json_list = []
        for i in files:
            # print(i['video'])

            video_path = i['video']

            if os.path.exists(video_path):
                pt_path = video_path.replace('.mp4', '.pt')
                

            new_json = {
                'video':video_path,
                'video_latent': pt_path,
                'text':'',
                'text_latent': ''
            }

            json_list.append(new_json)

        with open(output_file_path, 'w') as f:
            json.dump(json_list, f, indent=4)
            print("successfully dumps the data.")



import json 
import pandas as pd 
import re 

def deep_clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Normalize unicode characters
    # Map smart quotes, ellipses, and dashes to standard ASCII
    replacements = {
        '“': '"', '”': '"', '‘': "'", '’': "'",
        '…': '...', '–': '-', '—': '-'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
        
    # 2. Whitespace cleanup
    # Replace multiple spaces/tabs with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3. Capitalization
    # Ensure the sentence starts with an uppercase letter
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
        
    # 4. Terminal Punctuation
    # If the text ends with a letter, number, or closing parenthesis, add a period.
    if text and (text[-1].isalnum() or text[-1] in [')', ']']):
        text += '.'
        
    return text

def csv_file_take(metadata_file_path):
    """Processes a single CSV and returns a list of dictionaries."""
    local_list = []
    try:
        datas = pd.read_csv(metadata_file_path)
        # Check if 'file_name' and 'text' columns actually exist
        if 'file_name' not in datas.columns or 'text' not in datas.columns:
            print(f"Skipping {metadata_file_path}: Required columns missing.")
            return []

        for _, data in datas.iterrows():
            video_file = data['file_name']
            video_text = deep_clean_text(data['text'])

            video_latent_file = video_file.replace('.mp4', '.pt')
            text_latent_file = video_text.replace(' ', '_')
            text_latent_file = text_latent_file.split('.')[0]
            text_latent_file = text_latent_file.__add__('.pt')
            # print(text_latent_file)

            # print(video_latent_file)
            split_file = video_latent_file.split('/')[:-1]
            split_file = '/'.join(split_file)
            text_latent_new_file = split_file+'/'+text_latent_file
            

            local_list.append({
                "video": video_file,
                "text": video_text,
                "video_latent": video_latent_file,
                "text_latent": text_latent_new_file
            })

    except pd.errors.EmptyDataError:
        print(f"Skipping {metadata_file_path}: File is empty or whitespace only.")
    except Exception as e:
        print(f"Error reading {metadata_file_path}: {e}")
    
    return local_list



    

    # --- Main Execution ---
# video_folder_path = "./Data/clip_video"
# video_text_json_path = "./annotation/annotation_with_text/video_text_json.json"


def main(video_folder_path, video_text_json_path):

    all_video_data = [] # Master list to store EVERYTHING

    # Ensure output directory exists
    os.makedirs(os.path.dirname(video_text_json_path), exist_ok=True)

    for roots, dirs, files in os.walk(video_folder_path):
        for file in files:
            if file.lower().endswith('.csv'):
                new_path = Path(roots) / file
                
                # Use size check and the function return
                if new_path.exists() and new_path.stat().st_size > 0:
                    result = csv_file_take(new_path)
                    all_video_data.extend(result) # Add findings to master list

    # Save everything ONCE at the end
    with open(video_text_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_video_data, f, indent=2)
        print(f"Successfully saved {len(all_video_data)} entries to {video_text_json_path}")


def convert_into_jsonl(json_file_path):



    # Assuming your JSON is an array of objects
    with open(json_file_path, 'r') as f_in:
        data = json.load(f_in) # Loads the entire JSON array

    with open('./output.jsonl', 'w') as f_out:
        for item in data:
            # Dumps each item as a JSON string and writes it followed by a newline
            json.dump(item, f_out)
            f_out.write('\n') # Add the newline character



if __name__ == "__main__":
    # video_folder_path = "./Data/clip_video"
    # video_text_json_path = "./annotation/annotation_with_text/video_text_json.json"
    # main(video_folder_path, video_text_json_path)

    convert_into_jsonl("/home/manish/Desktop/projects/video_Generation/Tools/annotation/annotation_with_text/video_text_json.json")
    