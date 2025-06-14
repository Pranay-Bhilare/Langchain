from langchain import hub
from dotenv import load_dotenv
from langchain.agents import tool,create_react_agent,AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
import datetime


load_dotenv()
llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")

prompt_template = hub.pull("hwchase17/react")

query = "What is the current time in London ? (You are in India)"

@tool
def time_tool(format : str = "%Y-%m-%d %H:%M:%S"): 
     """Gives the current date and time in the specified format"""
     current_time = datetime.datetime.now()
     formatted_time = current_time.strftime(format)
     return formatted_time

tools = [time_tool]

agent = create_react_agent(llm=llm,prompt=prompt_template,tools=tools)

agent_executor = AgentExecutor(agent=agent, verbose=True,tools=tools)

result = agent_executor.invoke({"input" : query})
print(result)