from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

PROJECT_ID = "langchain-424b0"
COLLECTION_NAME = "chat_history"
SESSION_ID = "uniqueidentifier"

# Initialized the firestore client
client = firestore.Client(project=PROJECT_ID)

# Initialize FirestoreChatMessageHistory
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID, 
    collection=COLLECTION_NAME, 
    client=client
    )

print("Current chat history :", chat_history.messages)
print("Start chatting with AI or type `exit` to quit")

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash"
)

while True : 
    query = input("User : ")
    if query == "exit" :
        break

    chat_history.add_user_message(query);

    result = llm.invoke(chat_history.messages)
    ai_response = result.content
    chat_history.add_ai_message(ai_response);
    print(f"AI : {ai_response}")

print("--------Message History---------")
print(chat_history.messages)