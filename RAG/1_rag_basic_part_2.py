import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir,"db","chroma_db")

embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

db = Chroma(persist_directory=persistent_dir,embedding_function=embedding_model)

query = "What is TransFusion? What is its usecase ? How does it works ?"

retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"k" : 3, "score_threshold" : 0.5}
)

relevant_docs = retriever.invoke(query)

print("----Relevant docs--------")
for i, doc in enumerate (relevant_docs,1): 
    print(f"Document {i} : \n {doc.page_content} \n")
    if doc.metadata : 
        print(f"Source : {doc.metadata.get('source','Unknown')} \n")