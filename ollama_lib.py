import ollama
import pickle
import pynndescent
import numpy as np
import pandas as pd

import preprocessing_lib
from config import Config


def retrieve_fragment (input_embedding, embeddings_index, base): # Get fragments from base using embeddings
    input_embedding = np.array([list(input_embedding)], dtype=np.float32)
    embeddings_search_results = embeddings_index.query(input_embedding, k=3) # Get most similar
    distances = (embeddings_search_results[1]) [0]
    min_dist = float(distances [0]) # Distance is from 0 to 1, where 0 is the closest one

    all_fragment_inds = ((embeddings_search_results[0]) [0]).tolist () # Get fragments indices
    if (all_fragment_inds) and (min_dist < 0.7): # If there are fragments
        fragment_inds = []
        if min_dist <= 0.45: # Remove irrelevant blocks if there are good ones
            for d in range (0, len (distances)):
                if float (distances [d]) <= 0.45:
                    fragment_inds.append (all_fragment_inds [d])
        else:
            fragment_inds = all_fragment_inds
        block_texts = base.loc[fragment_inds, 'text'].tolist()

        texts = ' '.join(block_texts)

        # Return fragments and if there are no fragments
        return texts, False
    else:
        return Config.NO_ANSWER, True

async def get_response(prompt, embeddings_index, base): #Get response from model based on user's prompt
    input_embedding = preprocessing_lib.get_embeddings (preprocessing_lib.transform_unknown(preprocessing_lib.text_analyze (prompt))) # Get input embeddings
    fragment, no_answer = retrieve_fragment (input_embedding, embeddings_index, base) # Get fragments and other info
    if no_answer:
        return Config.GENERATED_QUESTION_PROMPT
    else:
        prompt = Config.CONTEXT_START+fragment+Config.CONTEXT_END+Config.RETRIEVE_PROMPT+Config.USER_START+prompt+Config.USER_END

    try: # Ask for generating answer
        response = ollama.chat(model=Config.MODEL_ID, messages=[{'role': 'user', 'content': prompt,},])
    except ollama.ResponseError as e:
        print('Error:', e.error)

    response = response['message']['content'] # Get response and return it
    return response