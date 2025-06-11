from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash"
)
message = [SystemMessage("You are a helpful assistant that translates English to Hinglish. Translate the user sentence "),
           HumanMessage("I love AI programming")]
    
result = llm.invoke(message)
print(result.content)