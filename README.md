# universal_RAG
Universal RAG

A template for creating applications using RAG and LLM models through the Ollama framework.

The data for the vector database is presented in the format .txt with \n split into semantic blocks in the docs folder, vector database based on pynndescent.

Model_file_folder contains a file template for customizing the model for the Ollama framework. The .env provides examples of prompts and some parameters and paths for the functioning of the RAG system.

At the start, database files are read or created. After each request to the model, the docs folder is checked for changes. In case of changes, the database is updated and the file variables in the code are updated (database versions 0 and 1 are changed, deleting the previous one).

Data preprocessing (both for databases and for the user's request) includes checking the text for spelling, tokenization, obtaining the original forms of the word and creating embeddings (the embedder is indicated depending on your task).

ollama_lib implements RAG, with the search for the 3 most relevant blocks of text from the database, taking into account the similarity of the blocks, filtering blocks with too low relevance.
