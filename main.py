import os
import sys
import dotenv
import asyncio
import pickle
import pandas as pd

import base_lib
import ollama_lib
from config import Config

async def main():
    # Open base file
    try:
        with open(Config.FOLDER_PATH+'/'+Config.BASE_DIR+'embeddings_data'+Config.START_ID, 'rb') as file_to_read:
            embeddings_index = pickle.load(file_to_read)
        base = pd.read_pickle(Config.FOLDER_PATH+'/'+Config.BASE_DIR+'data'+Config.START_ID)
        print ('Base opened')
    except FileNotFoundError:
        print ('No base files')
        print ('Trying to create')
        #Create base files

        dotenv_file = dotenv.find_dotenv(Config.FOLDER_PATH+'/.env') # Get base version from env
        dotenv.load_dotenv(dotenv_file)
        current_id = os.environ['START_ID']
        if current_id == '1': #Make the same id (because base_lib will change it)
            current_id = '0'
        else:
            current_id = '1'
        base_lib.update_base (current_id)
        try:
            with open(Config.FOLDER_PATH+'/'+Config.BASE_DIR+'embeddings_data'+Config.START_ID, 'rb') as file_to_read:
                embeddings_index = pickle.load(file_to_read)
            base = pd.read_pickle(Config.FOLDER_PATH+'/'+Config.BASE_DIR+'data'+Config.START_ID)
            print ('Base opened')
        except FileNotFoundError:
            print ('No base files, unable to create')

    current_id = Config.START_ID
    data_folder = Config.FOLDER_PATH+'/'+Config.BASE_DIR #Set base folder name and modified time
    folder_modified_time = os.path.getmtime(data_folder)
    print (Config.START_PHRASE)
    while True:
        prompt = input ()
        if prompt == Config.STOPPER:
            print (Config.FINAL_PHRASE)
            break
        answer = await ollama_lib.get_response(prompt, embeddings_index, base) # Get response

        print (answer)

        folder_new_modified_time = os.path.getmtime(data_folder) #Check base modification 
        if folder_new_modified_time != folder_modified_time: #Check if new files were added or old ones deleted (does not consider file changes!)
            folder_modified_time = folder_new_modified_time

            dotenv_file = dotenv.find_dotenv(Config.FOLDER_PATH+'/.env') # Get base version from env
            dotenv.load_dotenv(dotenv_file)

            new_id = base_lib.update_base (current_id) #Update base file

            if new_id != False:
                print ('Base updated')
                os.environ['START_ID'] = new_id #Set new id in env
                dotenv.set_key(dotenv_file, 'START_ID', os.environ['START_ID'])

            #Remove old base
            del embeddings_index
            del base
            os.remove (Config.FOLDER_PATH+'/'+Config.BASE_DIR+'embeddings_data'+current_id)
            os.remove (Config.FOLDER_PATH+'/'+Config.BASE_DIR+'data'+current_id)
            current_id = new_id

            #Reopen bases
            try:
                with open(Config.FOLDER_PATH+'/'+Config.BASE_DIR+'embeddings_data'+current_id, 'rb') as file_to_read:
                    embeddings_index = pickle.load(file_to_read)
                base = pd.read_pickle(Config.FOLDER_PATH+'/'+Config.BASE_DIR+'data'+current_id)
                print ('Base opened')
            except FileNotFoundError:
                print ('No base files')

if __name__ == '__main__':
    print ('In Work')
    asyncio.run(main())