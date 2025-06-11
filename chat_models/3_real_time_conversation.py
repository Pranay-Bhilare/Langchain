from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
load_dotenv()

chat_history = []
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash"
)

while True : 
    query = input("User : ")
    if query == "exit" :
        break
    chat_history.append(HumanMessage(content=query))

    result = llm.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))
    print(f"AI : {response}")

print("--------Message History---------")
print(chat_history)