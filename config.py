import sys
from os import environ, path
from dotenv import load_dotenv

#Load dotenv file with RAG and model data
try:
    load_dotenv(path.abspath(path.dirname(sys.argv[0]))+'/.env')
except FileNotFoundError:
    print ('No such config file')

class Config:
    FOLDER_PATH = path.abspath(path.dirname(sys.argv[0])) #Absolute path where the program executes
    BASE_DIR = environ.get('BASE_DIR')  #Directory to the folder with base files
    START_ID = environ.get('START_ID')  #Base ID when start program
    EMBEDDINGS_MODEL_DIR = environ.get('EMBEDDINGS_MODEL_DIR') #Embedding model directory
    EMBEDDINGS_DIM = environ.get('EMBEDDINGS_DIM') #Embeddings dimensionality
    EMBEDDINGS_MAX_CONTEXT = environ.get('EMBEDDINGS_MAX_CONTEXT') #Max amount of tokens for embeddings
    MODEL_ID = environ.get('MODEL_ID') #ID/name of the LLM model
    CONTEXT_START = environ.get('CONTEXT_START') #Prompts for getting context
    CONTEXT_END = environ.get('CONTEXT_END') 
    USER_START = environ.get('USER_START') #Prompts for getting user input
    USER_END = environ.get('USER_END')
    RETRIEVE_PROMPT = environ.get('RETRIEVE_PROMPT') #Prompt for generating answer to user question
    NO_ANSWER = environ.get('NO_ANSWER') #Answer with no information found
    GENERATED_QUESTION_PROMPT = environ.get ('GENERATED_QUESTION_PROMPT') #Answer to say that there is no info
    START_PHRASE = environ.get ('START_PHRASE') #Chatbot start phrase
    STOPPER = environ.get ('STOPPER') #Stopper for finishing the chat
    FINAL_PHRASE = environ.get ('FINAL_PHRASE') #Chatbot finish phrase