from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnableSequence

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")

message = [("system","You are a comediam who jokes about this {topic}"),
              ("human","Tell me {jokes_count} jokes")]
message_2 = [("system","You are a translator which translate the provided text in {language}"),
             ("human","Translate the following text into {language} : {text}")]
prompt_template = ChatPromptTemplate.from_messages(messages=message)
prompt_template_2 = ChatPromptTemplate.from_messages(messages=message_2)

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model  = RunnableLambda(lambda x: llm.invoke(x.to_messages()))
parse_output =  RunnableLambda(lambda x: x.content)
prepare_translation = RunnableLambda(lambda input : {"text" : input, "language" : "Hinglish"})
translate_template = RunnableLambda(lambda x: prompt_template_2.format_prompt(**x))

# chain = RunnableSequence(first = format_prompt, middle=[invoke_model,parse_output,prepare_translation,translate_template,invoke_model], last=parse_output);
chain = format_prompt  | invoke_model | StrOutputParser() | prepare_translation | translate_template | invoke_model | StrOutputParser()
response = chain.invoke({
    "topic" : "cricket",
    "jokes_count" : 5
})

print(response)