from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")

message = [("system","You are a comediam who jokes about this {topic}"),
              ("human","Tell me {jokes_count} jokes")]

prompt_template = ChatPromptTemplate.from_messages(messages=message)

# Creating combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | llm | StrOutputParser()

response = chain.invoke({
    "topic" : "cricket",
    "jokes_count" : 5
})

print(response)