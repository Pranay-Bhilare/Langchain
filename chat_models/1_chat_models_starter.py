from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash"
)
message = [("system", "You are a helpful assistant that translates English to Hinglish. Translate the user sentence "),
           ("user","I love AI programming")]
    
result = llm.invoke(message)
print(result.content)