from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnableSequence

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")

message = [("system","You are a comediam who jokes about this {topic}"),
              ("human","Tell me {jokes_count} jokes")]

prompt_template = ChatPromptTemplate.from_messages(messages=message)

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model  = RunnableLambda(lambda x: llm.invoke(x.to_messages()))
parse_output =  RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first = format_prompt, middle=[invoke_model], last=parse_output);

response = chain.invoke({
    "topic" : "cricket",
    "jokes_count" : 5
})

print(response)