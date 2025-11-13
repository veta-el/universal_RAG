import os
import pandas as pd
import numpy as np
import pickle
from pynndescent import NNDescent

from config import Config
import preprocessing_lib

def update_base (current_id: str):
    if current_id == '1': #Set id
        new_id = '0'
    else:
        new_id = '1'

    data = pd.DataFrame (columns=['file_id', 'section_id', 'text'])
    data_folder = Config.FOLDER_PATH+'/'+Config.BASE_DIR

    files_names = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
    
    for file in files_names: #Get all text files
        if file [-4:] == '.txt': #Check if txt
            with open(os.path.join(data_folder, file), 'r') as text_file: #read txt file
                text = text_file.read()
            
            blocks = text.split ('\n') #Split on semantic blocks
            blocks = list(set(list(filter(None, blocks)))) #Filter empty rows and copies

            for i, block in enumerate (blocks):
                data.loc[data.shape [0]] = [file [:-4], i, block]
        else:
            continue

    if data.shape != 0: #If data was found
        data ['preprocessed_text'] = data['text'].apply(preprocessing_lib.text_analyze) # Get preprocessing for texts
        embeddings = []
        for i in range (0, data.shape [0]): #Get embeddings
            embeddings.append (preprocessing_lib.get_embeddings (data.at [i, 'preprocessed_text'])) # Get embeddings based on text

        embeddings = np.array([np.array(xi) for xi in embeddings])
        emb_index = NNDescent(embeddings, metric='cosine') #Set vectors base

        with open (Config.FOLDER_PATH+'/'+Config.BASE_DIR+'embeddings_data'+new_id, 'wb') as file_to_write: #Save vectors base
            pickle.dump(emb_index, file_to_write)

        data = data.drop('preprocessed_text', axis=1) #Remove preprocessed_text for making lighter base with original texts

        data.to_pickle (Config.FOLDER_PATH+'/'+Config.BASE_DIR+'data'+new_id) #Save original base
        return new_id
    else:
        print ('No data found')
        return False