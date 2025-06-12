import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir,"db","chroma_db_metadata")

embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

db = Chroma(persist_directory=persistent_dir,embedding_function=embedding_model)

query = "Explain me mining data streams please"

retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"k" : 5, "score_threshold" : 0.3}
)

relevant_docs = retriever.invoke(query)

print("----Relevant docs--------")
for i, doc in enumerate (relevant_docs,1): 
    print(f"Document {i} : \n {doc.page_content} \n")
    print(f"Source : {doc.metadata.get('source','Unknown')} \n")


# ASKING LLM THE QUERY WITH PROVIDING THE CONTEXT.

llm = ChatGoogleGenerativeAI( model = "gemini-2.0-flash")

llm_input = ("You are a helpful assistant and user will ask a query" 
           + "and here are some related context data/documents for answering the query :"
           + "\n\n".join([doc.page_content for doc in relevant_docs])
           + "Please provide a response for user's query based on the context provided with a proper explanation, and if you don't get any context, then just say 'I don't know'")


messages = [SystemMessage(llm_input),
            HumanMessage(query) ]

result = llm.invoke(messages)
print("GENERATED RESPONSE :")
print(result.content)