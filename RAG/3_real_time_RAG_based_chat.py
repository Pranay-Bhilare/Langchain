from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain.schema.output_parser import StrOutputParser

import os
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir,"db","chroma_db_metadata")

embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

db = Chroma(persist_directory=persistent_dir,embedding_function=embedding_model)

chat_history = []
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant and user will ask a query" 
        + "and here are some related context data/documents for answering the query : {relevant_docs}"
        + "Please provide a response for user's query based on the context provided with a proper explanation, and if you don't get any context, then just say 'I don't know'"
        + "And this is the history of chat with the user for referencing if the user asks any past conversation related question {chat_history}"),
    ("human","{query}")])

chain = prompt_template | llm | StrOutputParser()

while True : 
    query = input("User : ")
    if query == "exit" :
        break
    chat_history.append(HumanMessage(content=query))

    retriever = db.as_retriever(
        search_type = "similarity_score_threshold",
        search_kwargs = {"k" : 10, "score_threshold" : 0.3}
    )

    relevant_docs = retriever.invoke(query)
    relevant_context = "\n\n".join(doc.page_content for doc in relevant_docs)

    result = chain.invoke({"relevant_docs" : relevant_context , "chat_history" : chat_history, "query" : query})
    print("AI : ",result)
    chat_history.append(AIMessage(content=result))


