import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir,"documents","document.txt")
persistent_dir = os.path.join(current_dir,"db","chroma_db")


if not os.path.exists(persistent_dir):
    print("Persistent directory is getting created , initializing the vector store....")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exists")
    
    loader = TextLoader(file_path,encoding="utf-8")
    document = loader.load()
    print("Length of document[0].page_content:", len(document[0].page_content))
    text_splitter = CharacterTextSplitter(chunk_size = 500, chunk_overlap = 50,separator = "\n")
    docs = text_splitter.split_documents(document)

    print("Number of document chunks : ",len(docs))
    print("Sample chunk : ", docs[0].page_content)


    print("----Creating Embeddings and storing in Vector DB -----")
    Chroma.from_documents(documents=docs,embedding=embeddings_model,persist_directory=persistent_dir)
    print("-----Finished creating embeddings and storing it in DB-----")


else :
    print("Vector store already exists")