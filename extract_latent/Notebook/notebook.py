import json

# The name of your input file containing the json lines
input_filename = '/home/manish/Desktop/projects/videoGen_fineTune/annotation/test_video_annotation.jsonl'
# The name of the output file to save the changes
output_filename = 'output_data.jsonl'

# 1. Open the input file for reading and output file for writing
with open(input_filename, 'r', encoding='utf-8') as infile, \
     open(output_filename, 'w', encoding='utf-8') as outfile:

    for line in infile:
        # 2. Parse the JSON line into a Python dictionary
        data = json.loads(line)
        
        # 3. Get the existing video path
        video_path = data.get('video', '')
        
        # 4. Add '.pt' to the end of the video path
        # Result example: ".../clip_0000.mp4" becomes ".../clip_0000.mp4.pt"
        pt_path = video_path + ".pt"
        
        # 5. Store this new path. 
        # Here I am saving it to the 'video_latent' field since it was empty.
        data['video_latent'] = pt_path
        
        # 6. Write the modified dictionary back to the new file
        outfile.write(json.dumps(data) + '\n')

print(f"Processing complete. Data saved to {output_filename}")