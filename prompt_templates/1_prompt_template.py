from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")

template = "Write a {tone} email to {company} for expressing interest for {role} , mentioning {skill} as skill"

prompt_template = ChatPromptTemplate.from_template(template=template)
prompt_template.invoke({
    "tone" : "Enthusiastic",
    "company" : "Google",
    "role" : "AI Dev",
    "skill" : "AI",
})
print(prompt_template);

message = [("system","You are a comediam who jokes about this {topic}"),
              ("human","Tell me {jokes_count} jokes")]

prompt_template_2 = ChatPromptTemplate.from_messages(messages=message)

prompt_template_2.invoke({
    "topic" : "cricket",
    "joke_count" : "5"
})