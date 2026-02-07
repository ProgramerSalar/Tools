import json, os 



if __name__ == "__main__":

    video_file_path = "./content"
    list_json = []
    for root, dirs, files in os.walk(video_file_path):
      
        

        
        for file in files:
            full_file_path = os.path.join(root, file)
            print(full_file_path)
            
        
            new_colab_json = {
                "video": full_file_path           
            }

            list_json.append(new_colab_json)

            # print(list_json)

        new_json_file = "./new_annotation/class_5_vae.jsonl"
        with open(new_json_file, 'w') as rite:
            # json.dumps(list_json, rite, indent=2)
            for entry in list_json:
                rite.write(json.dumps(entry) + '\n')

   




    




            




                




